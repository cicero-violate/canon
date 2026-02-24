## Session Handoff — Solver Pipeline + Mutation Crate

### What we built this session

Full solver pipeline (15 solvers), 3 new constraint graphs, mutation crate,
and orchestration `--mutate` flag. Pipeline compiles and runs end-to-end:

```bash
cargo run -p orchestration -- \
  test_projects/test_rust_project/model_ir.json \
  /tmp/test_emit
cd /tmp/test_emit && cargo build && cargo run
# prints: User: User { name: "Cheese", score: 42 }
# solver output:
#   WARN  provenance_solver: name "describe" shadowed in module 5
#   DIAG  cycle_diag_solver: type cycle detected [2]: ref u32 -> ref u32
#   INFO  liveness_solver: pruned 1 dead function(s) from emit_order
```

---

### Core files to know

```
model/src/ir/
  mod.rs          — re-exports all IR modules
  node.rs         — NodeId, NodeKind, Body, BasicBlock, Stmt, Terminator,
                    Field, Param, GenericParam, Visibility, TypeAlias
  edge.rs         — EdgeKind: Contains, Calls, Resolves, Renames, TypeOf,
                    TypeUnifies, CfgEdge, CfgBranch,
                    Outlives (NEW), ConstDep (NEW), Expands (NEW)
  csr_graph.rs    — CsrGraph<ND,ED>: from_edges(), neighbours(), Default
  model_ir.rs     — ModelIR: nodes, emit_order, edge_hints,
                    8 CsrGraphs (5 original + region, value, macro)
  model_diff.rs   — diff_semantic covers all 8 graphs + edge_hints + emit_order

analyzer/src/
  lib.rs               — analyze(ir) = derive() + solve()
  derive.rs            — routes all 11 EdgeKinds into 8 graph builders
  graph/
    name_graph.rs      — NameGraphBuilder   (Renames, Resolves)
    type_graph.rs      — TypeGraphBuilder   (TypeOf, TypeUnifies)
    call_graph.rs      — CallGraphBuilder   (Calls)
    module_graph.rs    — ModuleGraphBuilder (Contains, ImplFor)
    cfg_graph.rs       — CfgGraphBuilder    (CfgEdge, CfgBranch)
    region_graph.rs    — RegionGraphBuilder (Outlives)          NEW
    value_graph.rs     — ValueGraphBuilder  (ConstDep)          NEW
    macro_graph.rs     — MacroGraphBuilder  (Expands)           NEW
  solver/
    mod.rs             — solve() chains all 15 solvers + 6 originals
    module_solver.rs   — topo sort → emit_order
    name_solver.rs     — topo sort → rename propagation
    type_solver.rs     — Kosaraju SCC → cycle detection
    call_solver.rs     — DFS → dead function detection
    cfg_solver.rs      — DFS reachability + Cooper dominators
    use_solver.rs      — DFS on inv_module_graph → inject Use nodes
    invariant_solver.rs— dangling edges, impl targets, acyclic module graph
    visibility_solver.rs— pub/private enforcement across modules
    impl_solver.rs     — impl target existence + duplicate impl detection
    trait_solver.rs    — trait method completeness
    generic_solver.rs  — TypeUnifies concrete conflict detection via SCC
    provenance_solver.rs— name shadowing + symbol origin chains
    cycle_diag_solver.rs— structured diagnostics for type SCC cycles
    liveness_solver.rs — prune dead functions from emit_order
    stability_solver.rs— deterministic emit_order sort
    borrow_solver.rs   — stub: awaits IR lifetime nodes (gap E9)
    const_solver.rs    — stub: awaits NodeKind::Const/Static (gap E5)
    macro_solver.rs    — stub: awaits NodeKind::MacroCall (gap E14)
    exhaustiveness_solver.rs — stub: awaits NodeKind::Enum (gap E6)
    drop_solver.rs     — stub: awaits ownership/scope IR nodes
    unsafe_solver.rs   — stub: awaits unsafe_ flag on nodes (gap E12)

algorithms/src/graph/
  dfs.rs              — dfs(adj, start) -> Vec<usize>
  topological_sort.rs — Kahn's algorithm
  scc.rs              — Kosaraju SCC
  reachability.rs     — reachability(adj, roots) -> Vec<bool>  NEW
                        is_acyclic(adj) -> bool                 NEW

projection/src/emit/
  emitters.rs  — ImplEmitter: trait impl methods suppress pub vis  FIXED
  fmt.rs       — fmt_trait_method: trait decl methods suppress pub vis  FIXED

mutation/
  Cargo.toml
  src/
    lib.rs     — MutationOp, ChangeSet, apply/diff/verify re-exports
    apply.rs   — apply(ir, op) -> Result<NodeId>  (tombstone strategy)
    diff.rs    — diff(before, after) -> ChangeSet
    verify.rs  — verify(ir) = analyze(clone) + invariant_solver

orchestration/src/main.rs
  — args: <model_ir.json> <output_dir> [--mutate <mutation.json>]
  — --mutate: snapshot_A → apply_mutations → verify → diff → emit
              → snapshot_B + diff_report.json

test_projects/test_rust_project/model_ir.json
  — 30 nodes: Crate, Module x8, Struct x2, Trait, Impl x3, Method x3,
              Function x8, TypeAlias, TypeRef x2
  — exercises: trait impls, Describable trait, dead function,
               TypeUnifies cycle, cross-module Resolves, private access
```

---

### Gaps closed this session

| # | Gap | Fix |
|---|-----|-----|
| 1 | 15 solvers missing | All implemented (S1–S8 full, S9–S15 stubs wired to graphs) |
| 2 | Only 5 graphs, no 1:1 for stub solvers | +3 graphs: region, value, macro |
| 3 | No mutation pipeline | mutation crate: MutationOp, ChangeSet, apply/diff/verify |
| 4 | orchestration had no --mutate flag | Extended with snapshot_A/B + diff_report.json |
| 5 | Trait impl methods emitting `pub` | ImplEmitter + fmt_trait_method both fixed |
| 6 | CsrGraph had no Default impl | Added — enables #[serde(default)] on new graph fields |
| 7 | model_diff.rs only diffed nodes | Now covers all 8 graphs + edge_hints + emit_order |

---

### Remaining IR gaps (emit coverage)

| Gap | What is missing | IR fix needed |
|-----|----------------|---------------|
| E1  | `#[attribute]` on any item | `attrs: Vec<String>` on Struct, Fn, Impl, Trait |
| E2  | `where` clauses | `where_clauses: Vec<String>` on Fn, Impl, Struct, Trait |
| E3  | `pub use` re-exports | `vis: Visibility` on `NodeKind::Use` |
| E4  | `extern crate` | `NodeKind::ExternCrate { name, alias }` |
| E5  | `const` / `static` | `NodeKind::Const`, `NodeKind::Static` — unblocks S11 |
| E6  | `enum` variants | `NodeKind::Enum { variants }` — unblocks S13 |
| E7  | Tuple/unit structs | `StructKind` enum on `NodeKind::Struct` |
| E8  | `impl Trait` / `dyn Trait` edges | TypeOf edges from Fn → Trait nodes |
| E9  | Lifetime annotations | fix `fmt_params` for `&'a T`; add Outlives edges — unblocks S9 |
| E10 | Inline `mod` blocks | `inline: bool` on `NodeKind::Module` |
| E11 | Trait bounds on impl | verify `fmt_generics` round-trips `impl<T: Clone> Foo<T>` |
| E12 | `unsafe` flag | `unsafe_: bool` on Fn, Impl, Trait — unblocks S15 |
| E13 | `async` flag | `async_: bool` on Function/Method |
| E14 | Macro invocations | `NodeKind::MacroCall { path, tokens }` — unblocks S12 |
| E15 | Glob imports | `glob: bool` on `NodeKind::Use` |

### Remaining solver gaps

| Gap | What is missing |
|-----|----------------|
| S1  | use_solver: only one level of Resolves — transitive re-exports not followed |
| S2  | type_solver: SCC cycles detected but no diagnostic node emitted into IR |
| S3  | call_solver: dead functions marked but not removed from emit_order (liveness_solver now does this) |
| S4  | invariant_solver: Impl.for_struct check is warn-only, not hard error |

---

### Next session goals

**Priority 1 — unblock stub solvers (IR gaps):**
Close E5, E6, E12 in that order — each unblocks one full solver (S11, S13, S15).
Pattern: add NodeKind variant → add emitter → add JSON test nodes → wire solver.

**Priority 2 — mutation pipeline end-to-end test:**
Write a `test_mutation.json` that exercises AddNode + AddEdge + RemoveNode.
Run: `orchestration model_ir.json /tmp/out --mutate test_mutation.json`
Verify diff_report.json is correct and emitted Rust compiles.

**Priority 3 — capture_rustc integration:**
`capture_rustc` should emit `model_ir.json` from real Rust source.
Current status: unknown — needs audit of capture_rustc/src to see what it produces.
Goal: real Rust file → capture_rustc → model_ir.json → orchestration → identical .rs files.

**Priority 4 — IR gap E2 (where clauses):**
Needed for any non-trivial generic code. Add `where_clauses: Vec<String>` and
update `ImplEmitter` + `FnEmitter` to emit `where T: Foo` blocks.
