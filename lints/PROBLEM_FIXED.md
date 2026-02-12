## Core Features Wanted
1. Rustc wrapper intercepts cargo builds via RUSTC_WRAPPER
2. Runs `rustc_driver::run_compiler` with custom lint callbacks
3. Lint callbacks register `API_TRAITS_ONLY` policy
4. After compilation, emit structured JSON judgment signals to stdout
5. Auto-bootstrap: gate builds wrapper if missing

## Problems
1. **CRITICAL**: `rustc_driver::run_compiler(&args, &mut callbacks)` fails with "multiple input filenames provided (first two filenames are `graph_kernel` and `src/lib.rs`)"
   - Cargo passes: `["--crate-name", "graph_kernel", "--edition=2021", "src/lib.rs", ...]`
   - rustc_driver misinterprets `graph_kernel` as a second input file
   - Need to find correct rustc_driver API or preprocess args differently

2. Serde version conflicts during wrapper build (SOLVED via explicit --extern)

3. Library output naming: rustc_driver may write `liblib-*.rlib` instead of `libgraph_kernel-*.rlib` (unsolved, may be related to #1)

**Status**: Wrapper compiles and runs, but immediately hits rustc_driver arg parsing error, preventing lint execution.
