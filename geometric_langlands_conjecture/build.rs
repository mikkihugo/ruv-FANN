//! Build script for geometric-langlands
//! 
//! This script handles CUDA compilation when the cuda feature is enabled.

fn main() {
    #[cfg(feature = "cuda")] {
    use std::env;
    use std::path::PathBuf;

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda/kernels/");

    // Check for CUDA installation
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cuda_include = PathBuf::from(&cuda_path).join("include");
    let cuda_lib = PathBuf::from(&cuda_path).join("lib64");

    // Set up paths for CUDA
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cusolver");

    // TODO: Duke Performance Engineer - Add CUDA kernel compilation here
    // cuda_builder::CudaBuilder::new("src/cuda/kernels")
    //     .copy_to("target/cuda/kernels")
    //     .build()
    //     .unwrap();
    }
}