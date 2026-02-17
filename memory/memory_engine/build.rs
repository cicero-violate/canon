use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    println!("cargo:rerun-if-changed=src/hash/merkle.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj_path = out_dir.join("merkle.o");

    let status = Command::new("/opt/cuda/bin/nvcc")
        .args([
            "src/hash/merkle.cu",
            "-c",
            "-o",
        ])
        .arg(&obj_path)
        .args([
            "-Xcompiler", "-fPIC",
            "-std=c++17",
            "-ccbin", "/usr/bin/g++",
        ])
        .status()
        .expect("failed to execute nvcc");

    if !status.success() {
        panic!("nvcc failed");
    }

    // Link object file
    println!("cargo:rustc-link-arg={}", obj_path.display());

    // CUDA runtime
    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");

    // 🔥 REQUIRED for C++ symbols
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
