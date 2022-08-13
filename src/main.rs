mod gui;
mod number_input;
mod test;
mod plotters_iced;

extern crate rustacuda;
extern crate rustacuda_core;
extern crate rustacuda_derive;

use iced::{Application, Settings};
use rustacuda::CudaFlags;

fn main() {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty()).expect("Failed to init CUDA");

    gui::TestGui::run(Settings::default()).expect("Failed to start GUI");
}
