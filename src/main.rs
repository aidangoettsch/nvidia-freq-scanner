mod test;

#[macro_use]
extern crate rustacuda;
extern crate rustacuda_derive;
extern crate rustacuda_core;

use rustacuda::prelude::{CudaFlags};
use rustacuda::memory::DeviceCopy;
use rustacuda_core::DevicePointer;
use nvapi_hi::KilohertzDelta;

fn main() {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty()).unwrap();

    let mut test = test::Test::new(0, 2048 * 2usize.pow(20), 10, 32, 256).unwrap();

    for mem_clock in (0..10000).step_by(10000) {
        test.test_at_point(KilohertzDelta(mem_clock)).unwrap();
    }
}
