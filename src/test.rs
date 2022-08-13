use crate::test::TestError::{GenericError, NvapiError};
use nvapi_hi::nvapi::sys::gpu::pstate::PstateId;
use nvapi_hi::{ClockDomain, Gpu, KilohertzDelta, Status};
use rustacuda::error::CudaError;
use rustacuda::event::{Event, EventFlags};
use rustacuda::memory::{AsyncCopyDestination, DeviceBox, DeviceBuffer, DeviceCopy};
use rustacuda::prelude::{Context, ContextFlags, CopyDestination, Device, Module};
use rustacuda::stream::{Stream, StreamFlags};
use std::ffi::{CString, NulError};
use std::hash::{Hash, Hasher};
use iced::futures;

#[derive(Debug, Clone, Copy)]
pub(crate) struct TestResultPoint {
    pub(crate) write_bandwidth_gbps: f32,
    pub(crate) copy_bandwidth_gbps: f32,
    pub(crate) read_bandwidth_gbps: f32,
    pub(crate) read_errors: u64,
    pub(crate) offset: KilohertzDelta,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Test {
    pub(crate) idx: usize,
    bound_lower: KilohertzDelta,
    bound_upper: KilohertzDelta,
    initial_step: KilohertzDelta,
    buffer_size: usize,
    samples: u64,
    blocks: u32,
    threads_per_block: u32,
}

#[derive(Debug, Clone)]
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
    pub(crate) fn new(idx: usize, bound_lower: KilohertzDelta, bound_upper: KilohertzDelta, initial_step: KilohertzDelta) -> TestResult<Self> {
        let gpus = Gpu::enumerate().unwrap_or(vec![]);

        if gpus.len() <= idx {
            return Err(TestError::GenericError(
                "GPU index out of range".to_string(),
            ));
        }

        Ok(Test {
            idx,
            bound_lower,
            bound_upper,
            initial_step,
            buffer_size: 2048 * 2usize.pow(20),
            samples: 50,
            blocks: 32,
            threads_per_block: 256,
        })
    }

    fn benchmark_kernel(
        blocks: u32,
        threads_per_block: u32,
        samples: u64,
        function_name: &str,
        args: &[&dyn DeviceCopy],
    ) -> TestResult<f32> {
        // Load the module containing the function we want to call
        let kernel = include_str!("../benchmark-kernel/cmake-build-debug/kernel.ptx");
        let module_data = CString::new(kernel)?;
        let module = Module::load_from_string(&module_data)?;

        // Create a stream to submit work to
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let function_cstr = CString::new(function_name)?;
        let function = module.get_function(&function_cstr)?;

        let start_event = Event::new(EventFlags::DEFAULT)?;
        let stop_event = Event::new(EventFlags::DEFAULT)?;
        start_event.record(&stream)?;

        let args_vec: Vec<_> = args
            .iter()
            .map(|a| *a as *const _ as *mut ::std::ffi::c_void)
            .collect();

        for _ in 0..samples {
            unsafe {
                stream.launch(&function, blocks, threads_per_block, 0, &args_vec)?;
            }
        }

        stop_event.record(&stream)?;
        stop_event.synchronize()?;

        Ok(stop_event.elapsed_time_f32(&start_event)? / samples as f32)
    }

    fn run_write_benchmark(buffer: &mut DeviceBuffer<u8>,
                           blocks: u32,
                           threads_per_block: u32,
                           samples: u64,) -> TestResult<f32> {
        let threads = threads_per_block * blocks;
        let thread_size = buffer.len() / threads as usize;

        if thread_size * threads as usize > buffer.len() {
            return Err(GenericError(
                "Threads cannot evenly split buffer".to_string(),
            ));
        }

        if thread_size % 64 != 0 {
            return Err(GenericError(
                "Thread size must be multiple of 64".to_string(),
            ));
        }
        let thread_size_u64 = thread_size / 8;

        let mut thread_size_box = DeviceBox::<usize>::new(&thread_size_u64)?;

        let time_ms = Self::benchmark_kernel(
            blocks,
            threads_per_block,
            samples,
            "benchmarkWrite",
            &[&buffer.as_device_ptr(), &thread_size_box.as_device_ptr()],
        )?;

        let total_megabytes = buffer.len() / 1024 / 1024;
        let speed_gbps = total_megabytes as f32 * 1000.0 / 1024.0 / time_ms;

        println!(
            "wrote {:.2}MB in avg {:.2}ms ({:.5}GB/s)",
            total_megabytes, time_ms, speed_gbps
        );

        Ok(speed_gbps)
    }

    fn run_copy_benchmark(
        source: &mut DeviceBuffer<u8>,
        dest: &mut DeviceBuffer<u8>,
        samples: u64,
    ) -> TestResult<f32> {
        // Create a stream to submit work to
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let start_event = Event::new(EventFlags::DEFAULT)?;
        let stop_event = Event::new(EventFlags::DEFAULT)?;
        start_event.record(&stream)?;

        for _ in 0..samples {
            unsafe {
                source.async_copy_to(dest, &stream)?;
            }
        }

        stop_event.record(&stream)?;
        stop_event.synchronize()?;

        let copy_time_ms = stop_event.elapsed_time_f32(&start_event)? / samples as f32;

        let total_megabytes = dest.len() / 1024 / 1024;
        let copy_time_s = copy_time_ms / 1000.0;
        let copy_speed_gbps = 2.0 * total_megabytes as f32 / 1024.0 / copy_time_s;

        println!(
            "copied {:.2}MB in avg {:.2}ms ({:.5}GB/s)",
            2 * total_megabytes,
            copy_time_ms,
            copy_speed_gbps
        );

        Ok(copy_speed_gbps)
    }

    fn run_read_benchmark(buffer: &mut DeviceBuffer<u8>,
                          blocks: u32,
                          threads_per_block: u32,
                          samples: u64,) -> TestResult<(f32, u64)> {
        let threads = threads_per_block * blocks;
        let thread_size = buffer.len() / threads as usize;

        if thread_size * threads as usize > buffer.len() {
            return Err(GenericError(
                "Threads cannot evenly split buffer".to_string(),
            ));
        }

        if thread_size % 64 != 0 {
            return Err(GenericError(
                "Thread size must be multiple of 64".to_string(),
            ));
        }
        let thread_size_u64 = thread_size / 8;

        let mut thread_size_box = DeviceBox::new(&thread_size_u64)?;
        let mut errors_box = DeviceBox::new(&0u64)?;

        let time_ms = Self::benchmark_kernel(
            blocks,
            threads_per_block,
            samples,
            "benchmarkRead",
            &[
                &buffer.as_device_ptr(),
                &thread_size_box.as_device_ptr(),
                &errors_box.as_device_ptr(),
            ],
        )?;

        let total_megabytes = buffer.len() / 1024 / 1024;
        let speed_gbps = total_megabytes as f32 * 1000.0 / 1024.0 / time_ms;

        let mut errors = 0u64;
        errors_box.copy_to(&mut errors)?;

        println!(
            "read {:.2}MB in avg {:.2}ms ({:.5}GB/s) with {} errors",
            total_megabytes, time_ms, speed_gbps, errors
        );

        Ok((speed_gbps, errors))
    }

    fn set_memory_clock_offset(gpu_idx: usize, offset: KilohertzDelta) -> TestResult<()> {
        let gpus = Gpu::enumerate().unwrap_or(vec![]);

        let gpu = &gpus[gpu_idx];
        let inner = gpu.inner();

        inner.set_pstates(
            [(PstateId::P0, ClockDomain::Memory, offset)]
                .iter()
                .cloned(),
        )?;
        Ok(())
    }

    pub(crate) fn test_at_point(
        offset: KilohertzDelta,
        gpu_idx: usize,
        blocks: u32,
        threads_per_block: u32,
        samples: u64,
        buffer_size: usize
    ) -> TestResult<TestResultPoint> {
        let cuda_device = Device::get_device(gpu_idx as u32)?;
        let _context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            cuda_device,
        )?;

        println!("testing at +{:?}", offset);
        let mut source = unsafe { DeviceBuffer::<u8>::zeroed(buffer_size)? };
        let mut dest = unsafe { DeviceBuffer::<u8>::zeroed(buffer_size)? };

        Self::set_memory_clock_offset(gpu_idx, offset)?;

        let write_bandwidth_gbps = Self::run_write_benchmark(
            &mut source,
            blocks,
            threads_per_block,
            samples,
        )?;
        let copy_bandwidth_gbps = Self::run_copy_benchmark(&mut source, &mut dest, samples)?;
        let (read_bandwidth_gbps, read_errors) = Self::run_read_benchmark(
            &mut dest,
            blocks,
            threads_per_block,
            samples,
        )?;

        Ok(TestResultPoint {
            write_bandwidth_gbps,
            copy_bandwidth_gbps,
            read_bandwidth_gbps,
            read_errors,
            offset,
        })
    }
}

impl<H, I> iced_native::subscription::Recipe<H, I> for Test
    where
        H: Hasher,
{
    type Output = Progress;

    fn hash(&self, state: &mut H) {
        struct Marker;
        std::any::TypeId::of::<Marker>().hash(state);
    }

    fn stream(
        self: Box<Self>,
        _input: futures::stream::BoxStream<'static, I>,
    ) -> futures::stream::BoxStream<'static, Self::Output> {
        let bound_upper = self.bound_upper;
        let initial_step = self.initial_step;
        let gpu_idx = self.idx;
        let blocks = self.blocks;
        let threads_per_block = self.threads_per_block;
        let samples = self.samples;
        let buffer_size = self.buffer_size;

        Box::pin(futures::stream::unfold(
            State::TestPoint(self.bound_lower),
            move |state| async move {
                match state {
                    State::TestPoint(point) => {
                        if point <= bound_upper {
                            match Self::test_at_point(
                                point,
                                gpu_idx,
                                blocks,
                                threads_per_block,
                                samples,
                                buffer_size
                            ) {
                                Ok(res) => Some((
                                    Progress::Result(res),
                                    State::TestPoint(point + initial_step),
                                )),
                                Err(e) => Some((
                                    Progress::Errored(e),
                                    State::Finished,
                                )),
                            }
                        } else {
                            Some((
                                Progress::Finished,
                                State::Finished,
                            ))
                        }
                    }
                    State::Finished => {
                        let _: () = iced::futures::future::pending().await;

                        None
                    }
                }
            },
        ))
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Progress {
    Finished,
    Result(TestResultPoint),
    Errored(TestError),
}

pub(crate) enum State {
    TestPoint(KilohertzDelta),
    Finished,
}
