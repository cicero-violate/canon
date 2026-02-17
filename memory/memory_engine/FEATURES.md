1️⃣ Canonical State & Merkle Layer
| Feature                              | Status | Category     | Pure GPU Implementation              | Lose / Gain                    |
| ------------------------------------ | ------ | ------------ | -----------------------------------  | -----------------------------  |
| Page-based canonical state           | ✅     | Core         | ❌ Hard (GPU memory paging required) | ❌ Likely lose flexibility     |
| Flat array Merkle tree (`Vec<Hash>`) | ✅     | Core         | ⚠️ Needs device memory layout         | ➖ Neutral                     |
| Deterministic root hash              | ✅     | Core         | ✅ Yes                               | ➕ Gain performance            |
| Domain-separated hashing             | ✅     | Core         | ✅ Yes                               | ➖ Neutral                     |
| Incremental bubbling update          | ✅     | Core         | ⚠️ Hard (branch divergence)           | ❌ Likely lose simplicity      |
| Full tree recompute                  | ✅     | Core         | ✅ Ideal for GPU                     | ➕ Major gain                  |
| Dirty leaf batching                  | ⚠️     | Optimization | ⚠️ Present but not fully exploited   | ➖ Partial                     |
| Parallel parent hashing (Rayon)      | ❌     | Optimization | ❌ Not implemented                   | ➖ N/A                         |
| Deterministic full recompute check   | ✅     | Safety       | ✅ Yes                               | ➖ Neutral                     |
| Inclusion proofs                     | ❌     | Core         | ❌ Not implemented                    | ➖ N/A                         |
| Stateless proof verification         | ❌     | Core         | ❌ Not implemented                    | ➖ N/A                         |
| O(log n) update                      | ❌     | Optimization | ❌ Not implemented                    | ➖ N/A                         |

2️⃣ Hashing Backend
| Feature                             | Status | Category      | Pure GPU Implementation | Lose / Gain         |
| ----------------------------------- | ------ | ------------- | ----------------------- | ------------------  |
| CPU SHA256                          | ✅     | Legacy        | ❌ Removed              | ➖ Neutral          |
| GPU SHA256                          | ✅     | Upgrade       | ✅ Native               | ➕ Massive gain     |
| Backend abstraction (`HashBackend`) | ✅     | Architecture  | ❌ Not needed           | ❌ Lose abstraction |
| CPU/GPU hybrid                      | ✅     | Upgrade       | ❌ Not applicable       | ➖ Neutral          |
| GPU batched Merkle rebuild          | ✅     | Upgrade       | ✅ Native               | ➕ Massive gain     |
| CUDA feature flag                   | ✅     | Compatibility | ❌ Remove               | ➖ Simpler build    |

3️⃣ Storage Layer
| Feature                         | Status | Category       | Pure GPU Implementation     | Lose / Gain               |
| ------------------------------- | ------ | -------------- | --------------------------  | ------------------------  |
| In-memory PageStore             | ✅     | Core           | ❌ CPU-side only            | ➖ Neutral                |
| mmap PageStore                  | ⚠️     | Persistence    | ⚠️ Struct exists, not wired | ➖ Partial                |
| Flush-on-commit                 | ✅     | Durability     | ❌ GPU not involved         | ➖ Neutral                |
| Zero-copy leaf hashing          | ✅     | Optimization   | ❌ GPU needs copy to device | ❌ Lose                   |
| Delta apply with scratch buffer | ❌     | Technical debt | ❌ Not implemented          | ➖ N/A                    |

4️⃣ Engine Layer (Public API Boundary)
| Feature                 | Status | Category          | Pure GPU Implementation     | Lose / Gain |
| ----------------------- | ------ | ----------------- | --------------------------  | ----------- |
| `Engine` trait boundary | ✅     | Core Architecture | ✅ Still valid              | ➖ Neutral  |
| Admission logic         | ✅     | Core              | ❌ CPU                      | ➖ Neutral  |
| Delta registry          | ✅     | Core              | ❌ CPU                      | ➖ Neutral  |
| Commit delta            | ✅     | Core              | ⚠️ State mutation CPU        | ➖ Neutral  |
| Commit batch            | ✅     | Core              | ✅ GPU-backed state rebuild | ➕ Gain     |
| Event hash computation  | ✅     | Core              | ❌ CPU fine                 | ➖ Neutral  |
| Graph delta log         | ✅     | Core              | ❌ CPU                      | ➖ Neutral  |

5️⃣ Logging & Persistence
| Feature                     | Status | Category   | Pure GPU Implementation | Lose / Gain      |
| --------------------------- | ------ | ---------- | ----------------------- | ---------------  |
| Append-only transaction log | ✅     | Core       | ❌ CPU only             | ➖ Neutral       |
| Replay verification         | ✅     | Core       | ⚠️ GPU rebuild root      | ➕ Faster replay |
| Graph delta log             | ✅     | Core       | ❌ CPU                  | ➖ Neutral       |
| Journal recovery            | ❌     | Incomplete | ❌ Not implemented      | ➖ N/A           |

6️⃣ Performance Model
| Feature                       | Status | Category         | Pure GPU Implementation | Lose / Gain              |
| ----------------------------- | ------ | ---------------- | ----------------------- | -----------------------  |
| Flat contiguous layout        | ✅     | Core             | ✅ Required             | ➕ Essential             |
| Batched delta commit          | ✅     | Core             | ✅ Implemented          | ➕ Major gain            |
| Incremental updates           | ⚠️     | CPU optimization | ⚠️ Partial (dirty list) | ➖ Partial               |
| Full tree rebuild             | ✅     | Core             | ✅ Primary mode         | ➕ Major gain            |
| Massive parallel hashing      | ✅     | Upgrade          | ✅ Native               | ➕ Massive gain          |
| Deterministic reproducibility | ✅     | Core             | ✅ Yes                  | ➖ Neutral               |
