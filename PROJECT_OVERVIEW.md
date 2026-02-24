## Session Handoff — ModelIR Completeness + Emit Correctness

### What we built

A pipeline: `model_ir.json → ModelIR → derive() → solve() → emit() → *.rs files`

### Core files to know

```
model/src/ir/
  mod.rs          — re-exports all IR modules
  node.rs         — NodeId, NodeKind, Body, BasicBlock, Stmt, Terminator, Field, Param, GenericParam, Visibility, TypeAlias
  edge.rs         — EdgeKind, EdgeHint
  csr_graph.rs    — CsrGraph<ND,ED> with from_edges(), neighbours()
  model_ir.rs     — ModelIR struct: nodes arena, emit_order, edge_hints, five CsrGraphs

analyzer/src/
  lib.rs          — analyze(ir) = derive() + solve()
  derive.rs       — routes edge_hints into five graph builders
  graph/
    mod.rs
    name_graph.rs      — NameGraphBuilder
    type_graph.rs      — TypeGraphBuilder
    call_graph.rs      — CallGraphBuilder
    module_graph.rs    — ModuleGraphBuilder
    cfg_graph.rs       — CfgGraphBuilder
  solver/
    mod.rs             — solve() calls all five solvers in order
    name_solver.rs     — topo sort -> rename propagation
    type_solver.rs     — Kosaraju SCC -> cycle detection
    call_solver.rs     — DFS -> dead function detection
    module_solver.rs   — topo sort -> emit_order
    cfg_solver.rs      — DFS reachability + Cooper dominators

projection/src/
  lib.rs          — project(ir) -> Plan, emit_to_disk(plan, root)
  emit.rs         — trait Emit, one emitter struct per NodeKind
                    ModuleEmitter, StructEmitter, TraitEmitter,
                    ImplEmitter, FnEmitter, TypeAliasEmitter

orchestration/src/main.rs
  — args: <model_ir.json> <output_dir>
  — stages: load JSON -> analyze -> project -> emit_to_disk -> write snapshot

test_projects/test_rust_project/model_ir.json
  — hand-written ModelIR for the test project
  — nodes: Crate, Module x7, Struct, Impl, Method, Function x6, TypeAlias
  — edge_hints: Contains edges for module tree, Calls edges for call graph
```

### Three known gaps to fix next session

**Gap 1 — Use declarations not emitted.**

The IR has no concept of `use` statements. `src/core/engine.rs` uses `User` but has no `use crate::data::model::User;`. The model needs either:
- A `NodeKind::Use { path: String, alias: Option<String> }`, or
- The emitter resolves unqualified names via the `name_graph` and emits fully-qualified paths automatically.

The graph-correct approach: add `Resolves` edges in `edge_hints` from each use-site node to its definition node, then the name solver qualifies the path and the emitter generates the `use` declaration.

**Gap 2 — Derives not emitted.**

`NodeKind::Struct` has no `derives` field. `#[derive(Debug, Clone)]` is structural metadata that must be part of the IR. Fix:

```rust
NodeKind::Struct {
    name:     String,
    vis:      Visibility,
    generics: Vec<GenericParam>,
    fields:   Vec<Field>,
    derives:  Vec<String>,   // add this
}
```

Then `StructEmitter` emits `#[derive(...)]` before the struct if non-empty.

**Gap 3 — Entrypoint imports not emitted.**

`src/main.rs` calls `run()` but has no `use test_rust_project::core::engine::run;`. Same root cause as Gap 1 — use declarations must be nodes or inferred from `Resolves` edges in `name_graph`.

### Goal for next session

Make the emitted project compile from scratch with `cargo build` with zero manual fixes. The test is:

```bash
cargo run -p orchestration -- \
  test_projects/test_rust_project/model_ir.json \
  /tmp/test_emit

cd /tmp/test_emit && cargo build
```

Both commands must succeed with no errors. Fix in IR-first order — no ad hoc string patches in the emitter.
