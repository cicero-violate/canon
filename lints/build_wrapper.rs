//! Build script for lint_rustc wrapper
//! Usage: rustc build_wrapper.rs && ./build_wrapper

use std::process::Command;
use std::path::PathBuf;

fn main() {
    // Get sysroot
    let sysroot_output = Command::new("rustc")
        .args(&["--print", "sysroot"])
        .output()
        .expect("failed to get rustc sysroot");
    let sysroot = String::from_utf8(sysroot_output.stdout)
        .expect("invalid utf8 from rustc")
        .trim()
        .to_string();

    // Build lints library first
    let status = Command::new("cargo")
        .args(["build", "-p", "lints"])
        .env("LINT_RUSTC_BUILDING", "1")
        .current_dir(".")
        .status()
        .expect("failed to build lints");
    if !status.success() {
        eprintln!("cargo build failed");
        std::process::exit(1);
    }

    // Find libraries
    let target_dir = PathBuf::from("../target/debug");
    let lints_rlib = target_dir.join("liblints.rlib");

    let deps_dir = target_dir.join("deps");
    
    // Helper to find a dependency artifact in `target/debug/deps`
    let find_dep = |prefix: &str, suffix: &str| {
        std::fs::read_dir(&deps_dir)
        .expect("failed to read deps dir")
        .filter_map(Result::ok)
        .find(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.starts_with(prefix) && name.ends_with(suffix)
        })
        .map(|e| e.path())
        .unwrap_or_else(|| panic!("{}", format!("missing dependency {prefix}*{suffix} in deps")))
    };

    // Find lazy_static from deps (lints was compiled against this version)
    let lazy_static = find_dep("liblazy_static-", ".rlib");
    let serde_dep = find_dep("libserde-", ".rlib");
    let serde_derive = find_dep("libserde_derive-", ".so");
    let serde_json_dep = find_dep("libserde_json-", ".rlib");

    let rustc_driver = std::fs::read_dir(format!("{}/lib", sysroot))
        .expect("failed to read sysroot/lib")
        .filter_map(Result::ok)
        .find(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("librustc_driver-")
                && e.file_name().to_string_lossy().ends_with(".so")
        })
        .expect("no rustc_driver.so found")
        .path();

    // Compile lint_rustc
    let status = Command::new("rustc")
        .args(&[
            "-Z", "unstable-options",
            "-C", "prefer-dynamic",
            "-L", target_dir.to_str().unwrap(),
            "-L", deps_dir.to_str().unwrap(),
            "-L", &format!("{}/lib", sysroot),
            "-L", &format!("{}/lib/rustlib/x86_64-unknown-linux-gnu/lib", sysroot),
            "--extern", &format!("lints={}", lints_rlib.display()),
            "--extern", &format!("rustc_driver={}", rustc_driver.display()),
            "--extern", &format!("lazy_static={}", lazy_static.display()),
            "--extern", &format!("serde={}", serde_dep.display()),
            "--extern", &format!("serde_derive={}", serde_derive.display()),
            "--extern", &format!("serde_json={}", serde_json_dep.display()),
            "--edition", "2021",
            "lint_rustc/src/main.rs",
            "-o", target_dir.join("lint_rustc").to_str().unwrap(),
        ])
        .status()
        .expect("failed to compile lint_rustc");

    if status.success() {
        println!("✓ Built lint_rustc at ../target/debug/lint_rustc");
    } else {
        eprintln!("rustc compilation failed");
        std::process::exit(1);
    }
}
