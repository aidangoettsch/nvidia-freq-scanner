use nvapi_hi::{KilohertzDelta, Gpu, Status, ClockDomain};
use rustacuda::prelude::{Device, Context, ContextFlags, Module, CopyDestination};
use rustacuda::error::CudaError;
use crate::test::TestError::{NvapiError, GenericError};
use std::ffi::{CString, NulError};
use rustacuda::stream::{Stream, StreamFlags};
use nvapi_hi::nvapi::sys::gpu::pstate::PstateId;
use rustacuda::event::{Event, EventFlags};
use rustacuda::memory::{DevicePointer, DeviceCopy, cuda_malloc, cuda_free, DeviceBox, DeviceBuffer, AsyncCopyDestination};

struct TestResultPoint {
    write_bandwidth_mbps: f64,
    read_bandwidth_mbps: f64,
    offset: KilohertzDelta,
}

pub(crate) struct Test {
    idx: usize,
    cuda_device: Device,
    stream: Stream,
    module: Module,
    context: Context,
    bound_lower: KilohertzDelta,
    bound_upper: KilohertzDelta,
    initial_step: KilohertzDelta,
    results: Vec<TestResultPoint>,
    buffer_size: usize,
    samples: u64,
    blocks: u32,
    threads_per_block: u32
}

#[derive(Debug)]
pub(crate) enum TestError {
    CudaError(CudaError),
    NvapiError(Status),
    GenericError(String),
}

impl From<Status> for TestError {
    fn from(e: Status) -> Self {
        NvapiError(e)
    }
}

impl From<CudaError> for TestError {
    fn from(e: CudaError) -> Self {
        TestError::CudaError(e)
    }
}

impl From<NulError> for TestError {
    fn from(_: NulError) -> Self {
        TestError::GenericError("Kernel name cannot include null bytes.".to_string())
    }
}

type TestResult<T> = Result<T, TestError>;

impl Test {
    pub(crate) fn new(idx: u32, buffer_size: usize, samples: u64, blocks: u32, threads_per_block: u32) -> TestResult<Self> {
        let gpus = Gpu::enumerate().unwrap_or(vec![]);

        if gpus.len() <= idx as usize {
            return Err(TestError::GenericError("GPU index out of range".to_string()));
        }
        let cuda_device = Device::get_device(idx)?;

        // Create a context associated to this device
        let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, cuda_device)?;

        // Load the module containing the function we want to call
        let kernel = include_str!("../benchmark-kernel/cmake-build-debug/kernel.ptx");
        let module_data = CString::new(kernel)?;
        let module = Module::load_from_string(&module_data)?;

        // Create a stream to submit work to
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        unsafe {
            Ok(Test {
                idx: idx as usize,
                cuda_device,
                context,
                module,
                stream,
                bound_lower: KilohertzDelta(0),
                bound_upper: KilohertzDelta(0),
                initial_step: KilohertzDelta(0),
                results: vec![],
                buffer_size,
                samples,
                blocks,
                threads_per_block,
            })
        }
    }

    fn benchmark_kernel(&mut self, function_name: &str, args: &[&dyn DeviceCopy]) {}

    fn run_write_benchmark(&mut self, buffer: &mut DeviceBuffer<u8>) -> TestResult<f32> {
        let threads = self.threads_per_block * self.blocks;
        let thread_size = buffer.len() / threads as usize;

        if thread_size * threads as usize > buffer.len() {
            return Err(GenericError("Threads cannot evenly split buffer".to_string()));
        }

        if thread_size % 64 != 0 {
            return Err(GenericError("Thread size must be multiple of 64".to_string()));
        }
        let thread_size_u64 = thread_size / 8;

        let mut thread_size_box = DeviceBox::<usize>::new(&thread_size_u64)?;

        let function_name = CString::new("benchmarkWrite")?;
        let function = self.module.get_function(&function_name)?;
        let stream = &self.stream;

        let start_event = Event::new(EventFlags::DEFAULT)?;
        let stop_event = Event::new(EventFlags::DEFAULT)?;
        start_event.record(&self.stream)?;

        for _ in 0..self.samples {
            unsafe {
                launch!(function<<<self.blocks, self.threads_per_block, 0, stream>>>(
                    buffer.as_device_ptr(),
                    thread_size_box.as_device_ptr()
                ))?;
            }
        }

        stop_event.record(&self.stream)?;
        stop_event.synchronize()?;

        let write_time_ms = stop_event.elapsed_time_f32(&start_event)? / self.samples as f32;

        let total_megabytes = self.buffer_size / 1024 / 1024;
        let write_time_s = write_time_ms / 1000.0;
        let write_speed_gbps = total_megabytes as f32 / 1024.0 / write_time_s;

        println!("wrote {:.2}MB in avg {:.2}ms ({:.5}GB/s)", total_megabytes, write_time_ms, write_speed_gbps);

        Ok(write_speed_gbps)
    }

    fn run_copy_benchmark(&mut self, source: &mut DeviceBuffer<u8>, dest: &mut DeviceBuffer<u8>) -> TestResult<f32> {
        let start_event = Event::new(EventFlags::DEFAULT)?;
        let stop_event = Event::new(EventFlags::DEFAULT)?;
        start_event.record(&self.stream)?;

        for _ in 0..self.samples {
            unsafe {
                source.async_copy_to(dest, &self.stream)?;
            }
        }

        stop_event.record(&self.stream)?;
        stop_event.synchronize()?;

        let copy_time_ms = stop_event.elapsed_time_f32(&start_event)? / self.samples as f32;

        let total_megabytes = self.buffer_size / 1024 / 1024;
        let copy_time_s = copy_time_ms / 1000.0;
        let copy_speed_gbps = 2.0 * total_megabytes as f32 / 1024.0 / copy_time_s;

        println!("copied {:.2}MB in avg {:.2}ms ({:.5}GB/s)", 2 * total_megabytes, copy_time_ms, copy_speed_gbps);

        Ok(copy_speed_gbps)
    }

    fn set_memory_clock_offset(&self, offset: KilohertzDelta) -> TestResult<()> {
        let gpus = Gpu::enumerate().unwrap_or(vec![]);

        let gpu = &gpus[self.idx];
        let inner = gpu.inner();

        inner.set_pstates([(PstateId::P0, ClockDomain::Memory, offset)].iter().cloned())?;
        Ok(())
    }

    pub(crate) fn test_at_point(&mut self, offset: KilohertzDelta) -> TestResult<()> {
        println!("testing at +{:?}", offset);
        // self.set_memory_clock_offset(offset)?;
        let mut source = unsafe {
            DeviceBuffer::<u8>::zeroed(self.buffer_size)?
        };

        let mut dest = unsafe {
            DeviceBuffer::<u8>::zeroed(self.buffer_size)?
        };

        self.run_write_benchmark(&mut source)?;
        self.run_copy_benchmark(&mut source, &mut dest)?;

        Ok(())
    }
}
