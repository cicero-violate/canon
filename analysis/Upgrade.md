### GPU Offload Evaluation

Let

* ( V = |nodes| )
* ( E = |edges| )
* ( I = iterations )
* ( T = I(V+E) )

If
[
T \gg 10^6
]
→ GPU likely beneficial.

---

## Analysis Modules → GPU Suitability

| Module                         | Core Operation       | Complexity     | GPU Suitability                   | Reason                                                        |
| ------------------------------ | -------------------- | -------------- | --------------------------------- | ------------------------------------------------------------- |
| `graph/scc` (Kosaraju)         | DFS ×2               | (O(V+E))       | ❌ (as written) / ✔ (if rewritten) | DFS is sequential; must convert to FW-BW or label propagation |
| `control_flow/dominators`      | Iterative meet       | (O(I(V+E)))    | ✔✔                                | Bitset intersections parallelize well                         |
| `control_flow/dataflow`        | Fixpoint propagation | (O(I(V+E)))    | ✔✔                                | Frontier-based propagation maps to GPU                        |
| `deadcode` (DFS)               | Reachability         | (O(V+E))       | ✔ (if BFS)                        | Replace DFS with parallel frontier expansion                  |
| `cfg` (dominators)             | Graph meet           | (O(I(V+E)))    | ✔✔                                | Same as dominators                                            |
| `call_graph` (Dijkstra)        | Shortest path        | (O(E \log V))  | ✔ (large graphs)                  | Use delta-stepping for GPU                                    |
| `alias` / `taint` / `lifetime` | Graph reachability   | (O(I(V+E)))    | ✔✔                                | Dataflow-style propagation                                    |
| `concurrency/lockset`          | Set intersection     | (O(N \cdot L)) | ✔ (large L)                       | Parallel bitset ops help only if large                        |
| `usedef` (merge_sort)          | Sorting              | (O(n \log n))  | ⚠                                 | GPU only helps if very large input                            |
| `effect` (hash_lookup)         | Hash table           | (O(1)) avg     | ❌                                 | Random memory access; CPU cache better                        |
| `escape` (linear_search)       | Scan                 | (O(n))         | ❌                                 | Memory-bound, low arithmetic intensity                        |

---

## Highest ROI

| Priority | Target                    |
| -------- | ------------------------- |
| 🔥 1     | Dataflow engine           |
| 🔥 2     | Dominators                |
| 🔥 3     | SCC (rewrite)             |
| 🔥 4     | All reachability analyses |

---

## Structural Summary

If analysis dominated by:

[
I(V+E)
]

→ Build one GPU CSR + frontier engine and move all graph passes onto it.

If dominated by:

[
n \log n \text{ or memory lookups}
]

→ Stay on CPU.

---

**Conclusion:**
Your slow components are almost certainly the iterative graph analyses. Those are strong GPU candidates once rewritten into frontier-based kernels.
