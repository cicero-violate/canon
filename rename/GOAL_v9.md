# GOAL v9 — Rename: Stateful Graph Intelligence

## Mission

Transform the rename crate from a stateless AST rewriter into a
stateful, graph-intelligent refactoring engine. Every captured
Rust codebase becomes a persistent, queryable, GPU-traversable
knowledge graph.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    rename crate                      │
│                                                      │
│  RustcFrontend                                       │
│      │ capture_snapshot()                            │
│      ▼                                               │
│  GraphNormalizer                                     │
│      │ emit_deltas() → Vec<GraphDelta>               │
│      ▼                                               │
│  ┌──────────────┐      ┌─────────────────────────┐  │
│  │ database│      │      graph_gpu           │  │
│  │              │      │                          │  │
│  │ commit_graph │      │ GraphSnapshot            │  │
│  │  _delta()    │      │   .to_csr()              │  │
│  │              │      │      │                   │  │
│  │ WAL (.tlog)  │      │  CsrGraph                │  │
│  │ GraphDelta   │      │  (unified memory)        │  │
│  │  Log         │      │      │                   │  │
│  │ (.graph.log) │      │  bfs(source, filter)     │  │
│  │              │      │      │                   │  │
│  │ Merkle root  │      │  BfsResult               │  │
│  │  (GPU SHA256)│      │  reachable / dist        │  │
│  └──────────────┘      └─────────────────────────┘  │
│         │                        ▲                   │
│         │ materialized_graph()   │                   │
│         └────────────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

---

## State Layer — database

**Responsibility:** durability, integrity, history.

### What gets persisted

| Store | File | Contents |
|-------|------|----------|
| WAL | `<path>.tlog` | Page deltas (MmapLog, 64MB mmap) |
| Graph log | `<path>.graph.log` | GraphDelta stream (length-prefixed bincode) |
| Checkpoint | `<path>.checkpoint` | Full page store snapshot + root hash |
| Merkle tree | GPU unified memory | SHA256 tree, root = cryptographic state proof |

### Key APIs

```rust
// Persist one graph mutation
engine.commit_graph_delta(GraphDelta::AddNode(node))?;
engine.commit_graph_delta(GraphDelta::AddEdge(edge))?;

// Reconstruct full graph from log
let snapshot: GraphSnapshot = engine.materialized_graph()?;

// Reconstruct graph at any past point
let snapshot_at_k: GraphSnapshot = engine.graph_snapshot_at(k)?;

// Total deltas written
let n: u64 = engine.graph_delta_count();

// Cryptographic proof of current state
let root: Hash = engine.current_root_hash();
```

### Startup / Recovery

1. Open WAL → replay page deltas → rebuild Merkle tree
2. Open GraphDeltaLog → available for replay on demand
3. Root hash verified against last written RootHeader

---

## Query Layer — graph_gpu

**Responsibility:** zero-copy traversal, GPU-parallel graph queries.

### CSR layout (unified memory)

```
row_offsets : [u32; N+1]   neighbors of node i = col_indices[row_offsets[i]..row_offsets[i+1]]
col_indices : [u32; E]     neighbor node indices
edge_kinds  : [u8;  E]     Contains=0, Call=1, ControlFlow=2, Reference=3
node_ids    : [u64; N]     stable external node IDs
```

All arrays in `cudaMallocManaged` — CPU builds, GPU traverses,
no `cudaMemcpy` ever.

### Key APIs

```rust
// Build CSR from snapshot (zero-copy into unified memory)
let csr: CsrGraph = snapshot.to_csr();

// BFS — GPU when available, CPU fallback, same API
let result: BfsResult = bfs(&csr, source_idx, Some(EdgeKind::Call));

// Reachable nodes from source via Call edges
let impacted: Vec<u32> = result.reachable();

// Distance to every node (-1 = unreachable)
let dist: &[i32] = &result.dist;

// Lookup internal index for external node id
let idx: Option<u32> = csr.node_index(external_id);
```

### CsrGraph is stateless

- Built from a `GraphSnapshot`, lives as long as needed, then dropped
- Unified memory freed on `Drop`
- No writes back to `database`
- Rebuild it whenever the snapshot changes

---

## Capture → Persist → Query Pipeline

```
1. RustcFrontend::capture_snapshot(entry, args, env)
       │
       ▼
2. GraphNormalizer::normalize(items) → GraphSnapshot
   GraphNormalizer::emit_deltas(item) → Vec<GraphDelta>
       │
       ▼
3. for delta in deltas:
       engine.commit_graph_delta(delta)   ← persisted to .graph.log
       │
       ▼
4. let snapshot = engine.materialized_graph()   ← replay from log
       │
       ▼
5. let csr = snapshot.to_csr()                  ← CSR in unified mem
       │
       ▼
6. let result = bfs(&csr, source, edge_filter)  ← GPU traversal
       │
       ▼
7. use result.reachable() for:
   - impact_of(symbol_id)         → impacted symbols
   - cross_crate_users(symbol_id) → callers across crates
   - propagate_rename(id, name)   → symbols to rewrite
```

---

## Agent Instructions

You are operating on the `rename` crate in the canon workspace.
The following crates are available as dependencies:

- `database` — state, persistence, WAL, Merkle integrity
- `graph_gpu` — CSR builder, GPU BFS, zero-copy unified memory

### Rules

1. **Never store graph state in memory only.** Every `AddNode` and
   `AddEdge` must go through `engine.commit_graph_delta()`.

2. **Never traverse the raw `GraphSnapshot` in a loop for reachability.**
   Always build a `CsrGraph` via `snapshot.to_csr()` and use `bfs()`.

3. **Replay is free.** Use `engine.graph_snapshot_at(k)` to inspect
   any past state. Do not maintain your own history.

4. **The Merkle root is ground truth.** Two states with the same
   `root_hash` are identical. Use it for caching and deduplication.

5. **`graph_gpu` is stateless.** Build `CsrGraph`, query it, drop it.
   Do not hold it across mutations.

6. **Edge kinds are typed.** Filter BFS by `EdgeKind::Call` for
   call graphs, `EdgeKind::ControlFlow` for CFG traversal,
   `EdgeKind::Contains` for module containment, `EdgeKind::Reference`
   for type dependencies.

### Entry points for new work

| Task | Where to start |
|------|---------------|
| Capture a crate | `rustc_integration/frontends/rustc/frontend_driver.rs` |
| Normalize to graph | `rustc_integration/transform/normalizer.rs` |
| Persist graph deltas | `database::MemoryEngine::commit_graph_delta` |
| Query reachability | `graph_gpu::traversal::bfs` |
| Rename with impact | `rename/core/project_editor/propagate.rs` |
| Oracle implementation | `rename/core/oracle.rs` — replace NullOracle with GPU-backed impl |

---

## Immediate Next Steps

- [ ] Wire `GraphNormalizer::emit_deltas` output into `engine.commit_graph_delta`
      in `multi_capture.rs`
- [ ] Replace `OracleData::impact_of` CPU hash lookup with GPU BFS
      over Call edges
- [ ] Replace `OracleData::cross_crate_users` with GPU BFS filtered
      by Reference + Call edges across module boundaries
- [ ] Implement `graph_snapshot_at` diffing — two snapshots → changed
      node set → minimal rename propagation
- [ ] Add DFS and dominator tree traversal to `graph_gpu`
- [ ] Wire `MemoryEngine::compact()` with configurable checkpoint path
