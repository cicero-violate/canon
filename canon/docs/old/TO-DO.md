IMPLEMENTATION INSTRUCTIONS FOR 10 REMAINING ITEMS

1. FUNCTION BODY EXECUTION (Priority 1 - Blocks everything)
   Add bytecode IR: Create `src/runtime/bytecode.rs` with enum Instruction { LoadConst(Value), Add, Sub, Mul, Call(FunctionId), Return }. Convert function signatures to bytecode in `materialize.rs`. Implement interpreter in `executor.rs::execute()` that walks bytecode, maintains stack, executes ops. Replace `generate_stub_outputs` with `interpret_bytecode(function, inputs) -> outputs`. Support basic ops: arithmetic, field access, function calls. Emit deltas when bytecode hits `EmitDelta(payload)` instruction.
2. ACTUAL COMPUTATION (Priority 1 - Depends on #1)
   In bytecode interpreter, implement real operations: Add pops two Values, adds them, pushes result. Field access extracts from StructValue. Function calls push frame, bind inputs, execute callee bytecode, pop frame. Store intermediate values in ExecutionContext bindings. For now, hardcode simple functions like "add two numbers" or "concat strings" as bytecode sequences for testing.
3. TEST FIXTURES (Priority 2 - Enables validation)
   Create `fixtures/` directory with sample IR JSONs. Build `fixtures/simple_add.json` with: 2 modules, 1 struct (State), 1 trait (Add), 1 impl (StateAdd), 1 function (add), 1 tick graph. Generate bytecode for add function. Write integration test that loads fixture, executes tick, verifies outputs match expected. Add fixtures for: recursion detection, delta emission, multi-function composition.
4. PARALLEL EXECUTION (Priority 3 - Phase 2)
   Create `src/runtime/parallel.rs`. In `TickExecutor`, after topological sort, identify independent nodes (no shared dependencies). Use rayon::spawn for each independent subgraph. Collect results with channels. Before merging, run sequential version and assert outputs match (batch == sequential). Store both timing for benchmarks. Add `--parallel` flag to CLI.
5. SMT PROOF VALIDATION (Priority 3 - Phase 2, research-heavy)
   Add `z3 = "0.12"` to Cargo.toml. Create `src/proof/smt_bridge.rs`. For each FunctionContract, generate Z3 constraints: inputs are bounded, outputs satisfy postcondition, no division by zero (totality). Submit to Z3 solver. If SAT, extract counterexample and reject. If UNSAT, function is proven total. Attach Z3 proof certificate hash to Delta.proof_object_hash. Start with simple contracts (non-negative outputs) before complex invariants.
6. GPU RUNTIME (Priority 4 - Phase 3, optional)
   Add `wgpu = "0.19"` dependency. Create `src/gpu/codegen.rs` that converts bytecode to WGSL shader code. Only allow GpuFunction with pure math ops (no branching per GpuProperties). In `src/gpu/dispatch.rs`, initialize wgpu device, compile shader, allocate buffers, copy inputs to GPU, execute kernel, copy outputs back. Run same function on CPU and GPU, assert bit-identical results. Benchmark speedup.
7. KERNEL FUSION (Priority 5 - Phase 3, research)
   In `src/gpu/fusion_analyzer.rs`, detect pattern: kernel A outputs → kernel B inputs with no other uses. Merge WGSL code: inline kernel A into B, eliminate intermediate buffer. Prove equivalence: run unfused and fused, assert outputs match. Only fuse if both are pure (GpuProperties checks). Measure reduced memory bandwidth. This is PhD-level compiler optimization - defer until GPU runtime works.
8. REAL FUNCTION CODE (Priority 1 - Depends on #1)
   In `materialize.rs::render_impl_function()`, instead of `todo!()`, emit: `canon_runtime::execute_function(function_id, inputs)` which calls your interpreter. Or generate bytecode as Rust arrays: `const ADD_BYTECODE: &[Instruction] = &[LoadConst(0), LoadConst(1), Add, Return];`. Link generated code to runtime. Functions become thin wrappers around bytecode execution.
9. BYTECODE/AST (Priority 1 - Foundation)
   Design AST in `src/runtime/ast.rs`: `enum Expr { Literal(Value), BinOp(Box<Expr>, Op, Box<Expr>), FieldAccess(Box<Expr>, String), Call(FunctionId, Vec<Expr>) }`. Parser converts function IR (from proposal DSL) into AST. Compiler lowers AST to bytecode. Interpreter executes bytecode (see #1). Store bytecode in Function.metadata as serialized bytes. This lets you ship IR without source code.
10. PERFORMANCE (Priority 5 - After everything works)
    Profile with `cargo flamegraph`. Optimize hot paths: cache state hashes (don't recompute on every delta), use Arc<Function> to avoid clones, replace HashMap with FxHashMap, arena-allocate Values to reduce GC pressure. Add `#[inline]` to small functions. Use criterion for benchmarks. Parallelize independent tick executions (separate ticks can run concurrently). Only optimize after correctness is proven - premature optimization breaks Canon laws.

EXECUTION ORDER: 1,2,8,9,3 (get basic execution working), then 4,5 (phase 2), then 6,7,10 (phase 3 optional).
START WITH: Implement bytecode interpreter (#1), add simple test fixture (#3), replace todo!() (#8).
