1️⃣ Canonical State & Merkle Layer
| Feature                              | Status | Category     | Pure GPU Implementation             | Lose / Gain                   |
| ------------------------------------ | ------ | ------------ | ----------------------------------- | ----------------------------- |
| Page-based canonical state           | ✅     | Core         | ⚠️ GPU memory layout                 | ➕ Zero-copy hashing          |
| Device-resident Merkle tree          | ✅     | Core         | ✅ Fully GPU-native                 | ➕ Major simplification       |
| Flat array Merkle tree               | ❌     | Core         | ❌ Removed host Vec                 | ➕ Removes duplication        |
| Deterministic root hash              | ✅     | Core         | ✅ Yes                              | ➕ Gain performance           |
| Domain-separated hashing             | ✅     | Core         | ✅ Yes                              | ➖ Neutral                    |
| Incremental bubbling update          | ❌     | Core         | ❌ Removed                          | ➕ Simplified batch model     |
| Full tree recompute                  | ✅     | Core         | ✅ Primary GPU mode                 | ➕ Major gain                 |
| Dirty leaf batching                  | ⚠️      | Optimization | ⚠️ Partial                           | ➖ Partial                    |
| Parallel parent hashing (Rayon)      | ❌     | Optimization | ❌ Not implemented                  | ➖ N/A                        |
| Deterministic full recompute check   | ⚠️      | Safety       | ⚠️ Debug only                        | ➖ Production simplified      |
| Inclusion proofs                     | ❌     | Core         | ❌ Not implemented                  | ➖ N/A                        |
| Stateless proof verification         | ❌     | Core         | ❌ Not implemented                  | ➖ N/A                        |
| O(log n) update                      | ❌     | Optimization | ❌ Not implemented                  | ➖ N/A                        |

2️⃣ Hashing Backend
| Feature                             | Status | Category      | Pure GPU Implementation | Lose / Gain          |
| ----------------------------------  | ------ | ------------- | ----------------------- | -------------------- |
| CPU SHA256                          | ❌     | Legacy        | ❌ Removed              | ➕ Removes dead code |
| GPU SHA256                          | ✅     | Upgrade       | ✅ Native               | ➕ Massive gain      |
| Backend abstraction (`HashBackend`) | ✅     | Architecture  | ✅ Still used           | ➖ Neutral           |
| Hybrid execution                    | ❌     | Optimization  | ❌ Not used             | ➕ Simpler model     |
| GPU batched Merkle rebuild          | ✅     | Upgrade       | ✅ Fully GPU-native     | ➕ Major gain        |
| Persistent CUDA streams             | ✅     | Upgrade       | ⚠️ Single stream now     | ➕ Performance ready |

3️⃣ Storage Layer
| Feature                         | Status  | Category       | Pure GPU Implementation    | Lose / Gain              |
| ------------------------------- | ------- | -------------- | -------------------------- | ------------------------ |
| Unified managed memory store    | ✅      | Core           | ✅ GPU-visible             | ➕ Zero-copy hashing     |
| In-memory PageStore             | ✅      | Core           | ⚠️ Fallback alloc only      | ➕ Passes tests          |
| mmap Write-Ahead Log (WAL)      | ✅      | Persistence    | ❌ CPU-backed              | ➕ Crash-safe root       |
| Double-buffered root header     | ✅      | Durability     | ❌ CPU-backed              | ➕ Atomic root updates   |
| Deterministic WAL replay        | ✅      | Durability     | ⚠️ CPU deserialize         | ➕ Verified recovery     |
| Zero-copy leaf hashing          | ✅      | Optimization   | ✅ Direct device pointer   | ➕ Major gain            |
| GPU-side capacity growth        | ✅      | Upgrade        | ⚠️ Partial                  | ➕ Removes hard cap      |

4️⃣ Engine Layer (Public API Boundary)
| Feature                 | Status | Category          | Pure GPU Implementation     | Lose / Gain         |
| ----------------------- | ------ | ----------------- | --------------------------- | ------------------- |
| `Engine` trait boundary | ✅     | Architecture      | ✅ Still valid              | ➖ Neutral          |
| Admission logic         | ✅     | Core              | ❌ CPU                      | ➖ Neutral          |
| Delta registry          | ✅     | Core              | ❌ CPU                      | ➖ Neutral          |
| Commit delta            | ⚠️      | Core              | ⚠️ CPU mutation              | ➖ Neutral          |
| Commit batch            | ✅     | Core              | ✅ Single rebuild per batch | ➕ Major gain       |
| Per-delta rebuild       | ❌     | Core              | ❌ Removed                  | ➕ Removes thrash   |
| Event hash computation  | ✅     | Core              | ❌ CPU fine                 | ➖ Neutral          |
| Graph delta log         | ✅     | Core              | ❌ CPU                      | ➖ Neutral          |

5️⃣ Logging & Persistence
| Feature                     | Status | Category   | Pure GPU Implementation | Lose / Gain      |
| --------------------------- | ------ | ---------- | ----------------------- | ---------------- |
| mmap WAL (append-only)      | ✅     | Core       | ❌ CPU                  | ➕ Deterministic |
| Crash-safe fsync discipline | ✅     | Core       | ❌ CPU                  | ➕ Durability    |
| Replay verification         | ✅     | Core       | ⚠️ CPU deserialize + GPU rebuild | ➕ Strong safety |
| Graph delta log             | ✅     | Core       | ❌ CPU                  | ➖ Neutral       |
| Journal recovery            | ✅     | Core       | ⚠️ Hybrid               | ➕ Root-validated |

6️⃣ Performance Model
| Feature                       | Status | Category         | Pure GPU Implementation | Lose / Gain             |
| ----------------------------- | ------ | ---------------- | ----------------------- | ----------------------- |
| Flat contiguous layout        | ✅     | Core             | ✅ Required             | ➕ Essential            |
| Batched delta commit          | ✅     | Core             | ✅ Implemented          | ➕ Major gain           |
| Incremental updates           | ❌     | CPU optimization | ❌ Removed              | ➕ Simplified GPU path  |
| Full tree rebuild             | ✅     | Core             | ✅ Primary mode         | ➕ Major gain           |
| Massive parallel hashing      | ✅     | Upgrade          | ✅ Native               | ➕ Massive gain         |
| Deterministic reproducibility | ✅     | Core             | ✅ Yes                  | ➖ Neutral              |
| Single-rebuild replay model   | ✅     | Core             | ✅ GPU rebuild once     | ➕ Predictable recovery |
