# Develop with cargo normally
LINT_RUSTC_BUILDING=1 cargo build
LINT_RUSTC_BUILDING=1 cargo check
LINT_RUSTC_BUILDING=1 cargo test

# When ready, rebuild the wrapper
./build_wrapper  # or: rustc build_wrapper.rs && ./build_wrapper
