## Session Handoff — ModelIR Emit Completeness + Mutation Pipeline

### What we built

A pipeline: `model_ir.json → ModelIR → derive() → solve() → emit() → *.rs files`

All three original gaps are closed. The pipeline now compiles and runs
from scratch with zero manual fixes:

```bash
cargo run -p orchestration -- \
  test_projects/test_rust_project/model_ir.json \
  /tmp/test_emit
cd /tmp/test_emit && cargo build && cargo run
# prints: User: User { name: "Cheese", score: 42 }
```

---

### Core files to know

```
model/src/ir/
  mod.rs          — re-exports all IR modules
  node.rs         — NodeId, NodeKind, Body, BasicBlock, Stmt, Terminator,
                    Field, Param, GenericParam, Visibility, TypeAlias
                    ** NodeKind::Use { path, alias }  (added this session)
                    ** NodeKind::Struct.derives: Vec<String>  (added this session)
  edge.rs         — EdgeKind (Contains, Calls, Resolves, Renames, TypeOf,
                    TypeUnifies, CfgEdge, CfgBranch), EdgeHint
  csr_graph.rs    — CsrGraph<ND,ED>: from_edges(), neighbours()
  model_ir.rs     — ModelIR struct: nodes arena, emit_order, edge_hints,
                    five CsrGraphs

analyzer/src/
  lib.rs               — analyze(ir) = derive() + solve()
  derive.rs            — routes edge_hints into five graph builders
  graph/
    name_graph.rs      — NameGraphBuilder (Renames, Resolves)
    type_graph.rs      — TypeGraphBuilder (TypeOf, TypeUnifies)
    call_graph.rs      — CallGraphBuilder (Calls)
    module_graph.rs    — ModuleGraphBuilder (Contains, ImplFor)
    cfg_graph.rs       — CfgGraphBuilder (CfgEdge, CfgBranch)
  solver/
    mod.rs             — solve() runs all solvers in order
    module_solver.rs   — topo sort → emit_order
    name_solver.rs     — topo sort → rename propagation
    type_solver.rs     — Kosaraju SCC → cycle detection
    call_solver.rs     — DFS → dead function detection
    cfg_solver.rs      — DFS reachability + Cooper dominators
    use_solver.rs      — DFS on inv_module_graph → inject Use nodes  (NEW)

projection/src/
  lib.rs               — project(ir) -> Plan, emit_to_disk(plan, root)
  emit/
    mod.rs             — re-exports emit_files, emit_node, emit_cargo_toml
    emitters.rs        — ModuleEmitter, StructEmitter, TraitEmitter,
                         ImplEmitter, FnEmitter, TypeAliasEmitter,
                         UseEmitter  (NEW)
    body.rs            — emit_blocks, indent_raw
    fmt.rs             — fmt_generics, fmt_params, fmt_field, fmt_trait_method
    cargo.rs           — emit_cargo_toml (Cargo.toml from NodeKind::Crate)  (NEW)

orchestration/src/main.rs
  — args: <model_ir.json> <output_dir>
  — stages: load JSON → analyze → project → emit_to_disk → write snapshot

test_projects/test_rust_project/model_ir.json
  — nodes: Crate, Module x7, Struct (+ derives), Impl, Method,
           Function x6, TypeAlias
  — edge_hints: Contains (module tree), Calls (call graph),
                Resolves (use-site → definition)  (NEW)

algorithms/src/graph/
  dfs.rs             — dfs(adj, start) -> Vec<usize>  (used by use_solver)
  topological_sort.rs — Kahn's algorithm (used by module_solver, name_solver)
  scc.rs             — Kosaraju (used by type_solver)
```

---

### Gaps closed this session

| # | Gap                                            | Fix                                                                   |
|---+------------------------------------------------+-----------------------------------------------------------------------|
| 1 | `use` declarations not emitted in module files | `NodeKind::Use` + `use_solver` + `Resolves` edges                     |
| 2 | `#[derive(...)]` not emitted on structs        | `derives: Vec<String>` on `NodeKind::Struct` + `StructEmitter`        |
| 3 | Binary entrypoint `use` not emitted            | Same `use_solver`; binary modules get crate-name prefix not `crate::` |
| 4 | No `Cargo.toml` emitted                        | `cargo.rs` emitter reads `NodeKind::Crate { name, edition }`          |

---

### Remaining emit gaps — full Rust source coverage

These are the NodeKind/feature combinations not yet represented in the IR
or emitter. Fix in IR-first order (IR → solver → emitter → JSON).

**IR structure gaps:**

| Gap | What is missing                                                                | IR fix needed                                                                          |                                                                                                                 |
|-----+--------------------------------------------------------------------------------+----------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------|
| E1  | `#[attribute]` on any item (derive is now handled but arbitrary attrs are not) | `NodeKind` fields: `attrs: Vec<String>` on Struct, Fn, Impl, Trait                     |                                                                                                                 |
| E2  | `where` clauses                                                                | `GenericParam` only has bounds inline; no `where T: Foo + Bar` clause node             | add `where_clauses: Vec<String>` to Fn, Impl, Struct, Trait                                                     |
| E3  | `pub use` re-exports                                                           | No `NodeKind::UseReexport` or `vis` on `NodeKind::Use`                                 | add `vis: Visibility` to `NodeKind::Use`                                                                        |
| E4  | `extern crate` declarations                                                    | Not representable                                                                      | add `NodeKind::ExternCrate { name: String, alias: Option<String> }`                                             |
| E5  | `const` and `static` items                                                     | Not representable                                                                      | add `NodeKind::Const { name, vis, ty, value: String }` and `NodeKind::Static { name, vis, ty, value, mutable }` |
| E6  | `enum` variants                                                                | No `NodeKind::Enum`                                                                    | add `NodeKind::Enum { name, vis, generics, variants: Vec<Variant> }`                                            |
| E7  | Tuple structs and unit structs                                                 | `Field.name: Option<String>` exists but `StructKind` (named/tuple/unit) is not encoded | add `StructKind` enum to `NodeKind::Struct`                                                                     |
| E8  | `impl Trait` / `dyn Trait` in signatures                                       | Stored as raw strings in `ty`/`ret` — no graph edges                                   | add `TypeOf` edges from Fn nodes to Trait nodes                                                                 |
| E9  | Lifetime annotations on functions                                              | `GenericParam.is_lifetime` exists but fn signatures don't emit `'a` on refs            | fix `fmt_params` to emit `&'a T` when lifetime present                                                          |
| E10 | `mod` blocks declared inline (not as files)                                    | `NodeKind::Module` always maps to a file                                               | add `inline: bool` to `NodeKind::Module`; `ModuleEmitter` emits `mod name { ... }`                              |
| E11 | Trait bounds on `impl` blocks (`impl<T: Clone> Foo<T>`)                        | `generics` exists but bound emit in `ImplEmitter` is not tested                        | verify `fmt_generics` round-trips correctly                                                                     |
| E12 | `unsafe` functions, impls, traits                                              | No `unsafe` flag                                                                       | add `unsafe_: bool` to Fn, Impl, Trait NodeKinds                                                                |
| E13 | `async` functions                                                              | No `async` flag                                                                        | add `async_: bool` to `NodeKind::Function/Method`                                                               |
| E14 | Macro invocations as items                                                     | Not representable                                                                      | add `NodeKind::MacroCall { path: String, tokens: String }`                                                      |
| E15 | `use` glob imports (`use foo::*`)                                              | `NodeKind::Use.path` has no glob variant                                               | add `glob: bool` to `NodeKind::Use`                                                                             |

**Solver gaps:**

| Gap | What is missing                                                                       |
|-----+---------------------------------------------------------------------------------------|
| S1  | `use_solver` only handles one level of Resolves — transitive re-exports not followed  |
| S2  | `type_solver` detects SCC cycles but does not emit a compiler error / diagnostic node |
| S3  | `call_solver` marks dead functions but does not remove them from emit_order           |
| S4  | No solver enforces that `NodeKind::Impl.for_struct` resolves to an actual Struct node |

---

### Next goal — Mutation pipeline

Target workflow:

```
load → snapshot → mutate → diff → verify → emit → snapshot
```

**Stages:**

```
load:     read model_ir.json → ModelIR
snapshot: serialize ModelIR → snapshot_A.json  (baseline)
mutate:   apply a MutationOp to ModelIR in memory
diff:     compare snapshot_A vs current IR → ChangeSet
verify:   run analyze() on mutated IR; check solver invariants
emit:     project() → emit_to_disk()
snapshot: serialize mutated IR → snapshot_B.json
```

**New types needed:**

```rust
// mutation/src/lib.rs

/// A single atomic mutation on ModelIR.
enum MutationOp {
    AddNode    { kind: NodeKind, span: Option<String> },
    RemoveNode { id: NodeId },
    UpdateNode { id: NodeId, kind: NodeKind },
    AddEdge    { hint: EdgeHint },
    RemoveEdge { src: NodeId, dst: NodeId, kind: EdgeKind },
}

/// Result of diffing two ModelIR snapshots.
struct ChangeSet {
    added_nodes:   Vec<NodeId>,
    removed_nodes: Vec<NodeId>,
    changed_nodes: Vec<(NodeId, NodeKind, NodeKind)>,  // (id, before, after)
    added_edges:   Vec<EdgeHint>,
    removed_edges: Vec<EdgeHint>,
}

fn apply(ir: &mut ModelIR, op: MutationOp) -> Result<NodeId>;
fn diff(before: &ModelIR, after: &ModelIR) -> ChangeSet;
fn verify(ir: &ModelIR) -> Result<()>;   // re-runs analyze(), checks invariants
```

**New crate:** `mutation/` (peer to `analyzer/`, `projection/`)

**Orchestration extension:**

```
orchestration <model_ir.json> <output_dir> [--mutate <mutation.json>]
```

A `mutation.json` file describes a sequence of `MutationOp`s to apply
before emit. The orchestration pipeline becomes:

```
load → analyze → snapshot_A → apply_mutations → diff(A, current)
     → verify → emit → snapshot_B → write diff report
```

**Files to create next session:**

```
mutation/
  Cargo.toml
  src/
    lib.rs       — MutationOp, ChangeSet, apply(), diff(), verify()
    apply.rs     — apply(ir, op) -> Result<NodeId>
    diff.rs      — diff(before, after) -> ChangeSet
    verify.rs    — verify(ir) calls analyze() + invariant checks

orchestration/src/main.rs  — extend with --mutate flag and snapshot_B write
```
