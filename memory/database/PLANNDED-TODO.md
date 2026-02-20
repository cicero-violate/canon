9️⃣ Planned Future Features

| Feature                          | Category        | Goal                                      | Expected Impact              |
| -------------------------------- | --------------- | ----------------------------------------- | ---------------------------- |
| Multi-stream Merkle scheduling   | Performance     | Overlap leaf + parent level hashing       | ➕ Higher GPU utilization     |
| Stream-ordered async rebuild     | Performance     | Remove implicit synchronization           | ➕ Lower latency              |
| True device-only tree API        | Architecture    | Eliminate host root copies entirely       | ➕ Zero host dependency       |
| GPU memory pool allocator        | Optimization    | Avoid repeated cudaMalloc calls           | ➕ Lower allocation overhead  |
| CUDA graph execution             | Performance     | Pre-record Merkle rebuild pipeline        | ➕ Kernel launch efficiency   |
| Inclusion proof generation (GPU) | Core            | Parallel Merkle path extraction           | ➕ Proof scalability          |
| Stateless proof verification     | Core            | On-device proof checking                  | ➕ Trustless verification     |
| Sparse leaf compression          | Optimization    | Avoid hashing empty leaves                | ➕ Reduced memory bandwidth   |
| Pinned host staging buffers      | I/O Optimization| Faster host↔device transfers              | ➕ Lower PCIe overhead        |
| Deterministic replay harness     | Safety          | Cross-device reproducibility validation   | ➕ Strong correctness model   |
| mmap-backed PageStore (GPU)      | Persistence     | Hybrid persistent + managed memory model  | ➕ Durable GPU state          |
| WAL compaction with snapshot     | Persistence     | Atomic snapshot + truncate WAL            | ➕ Bounded log growth         |
| Header generation fencing        | Durability      | Strict monotonic generation enforcement    | ➕ Stronger crash ordering    |
| WAL CRC32 per-record             | Safety          | Detect partial/corrupt records            | ➕ Hardened recovery          |
| Kernel-level domain separation   | Security        | Enforce hash domains inside device code   | ➕ Harder misuse surface      |
| Batched inclusion proof export   | Performance     | Generate N proofs in parallel             | ➕ Throughput gain            |
| O(log n) selective rebuild mode  | Advanced Mode   | Optional incremental path for small sets  | ➕ Lower latency option       |
