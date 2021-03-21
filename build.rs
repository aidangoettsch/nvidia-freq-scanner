use std::process::Command;

fn main() {
    let mut run_cmake = Command::new("cmake");
    run_cmake.args(vec![
        "--build",
        "benchmark-kernel/cmake-build-debug",
        "--target",
        "benchmark_kernel"
    ]);
    assert!(run_cmake.status().unwrap().success());
    println!("cargo:rustc-link-search=native={}", std::env::var("CUDA_LIBRARY_PATH").unwrap());
}