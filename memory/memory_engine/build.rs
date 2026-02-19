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
    let lib_path = out_dir.join("libmerkle.a");

    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/opt/cuda".into());
    let nvcc_path = PathBuf::from(&cuda_home).join("bin/nvcc");

    let status = Command::new(nvcc_path)
        .args(["src/hash/merkle.cu", "-c", "-o"])
        .arg(&obj_path)
        .args([
            "-Xcompiler",
            "-fPIC",
            "-std=c++17",
            "-ccbin",
            "/usr/bin/g++-11",
        ])
        .status()
        .expect("failed to execute nvcc");

    if !status.success() {
        panic!("nvcc failed");
    }

    // Archive into static library so downstream crates can link it.
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let status = Command::new(ar)
        .args(["crus", lib_path.to_str().unwrap(), obj_path.to_str().unwrap()])
        .status()
        .expect("failed to run ar");
    if !status.success() {
        panic!("ar failed");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=merkle");

    // CUDA runtime
    println!(
        "cargo:rustc-link-search=native={}/lib64",
        cuda_home
    );
    println!("cargo:rustc-link-lib=dylib=cudart");

    // 🔥 REQUIRED for C++ symbols
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
