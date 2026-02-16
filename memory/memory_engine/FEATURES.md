# memory_engine – Feature Matrix (Accurate)

---

## State Model

| Feature                      | Status | Notes                                      |
|------------------------------+--------+--------------------------------------------|
| Page-based canonical state   | ✅     | Logical pages, mutable via deltas          |
| Flat array Merkle tree       | ✅     | Full binary tree stored in `Vec<Hash>`     |
| Deterministic root hash      | ✅     | Root stored at index `1`                   |
| Configurable capacity        | ✅     | `with_capacity(max_leaves)`                |
| Auto-resize + full rebuild   | ✅     | Grows via `next_power_of_two`              |
| Domain-separated hashing     | ✅     | LEAF / NODE / DELTA / EVENT prefixes       |
| Incremental leaf updates     | ✅     | Dirty leaf tracking + bubbling             |
| Batched rehash (dirty set)   | ✅     | Deduped dirty leaves per update cycle      |
| Parallel level hashing       | ✅     | Rayon used during parent layer computation |
| Deterministic full recompute | ✅     | `full_recompute_root()`                    |
| Merkle inclusion proofs      | ✅     | Proof path generation                      |
| Stateless proof verification | ✅     | External verification API                  |
| Internal consistency check   | ✅     | Root compared to recomputed tree           |
| O(log n) update complexity   | ✅     | Per delta after leaf insertion             |

---

## Storage Layer

| Feature                    | Status | Notes                                  |
|----------------------------+--------+----------------------------------------|
| In-memory PageStore        | ✅     | Default backend                        |
| Memory-mapped PageStore    | ✅     | `memmap2` backend                      |
| Automatic in-memory growth | ✅     | `Vec` resize fallback                  |
| mmap bounds guard          | ✅     | Assert on overflow                     |
| Flush on commit            | ✅     | Ensures mmap durability                |
| Zero-copy hashing          | ❌     | Page snapshot is cloned before hashing |

---

## Logging

| Feature                     | Status | Notes                                    |
|-----------------------------+--------+------------------------------------------|
| Append-only transaction log | ✅     | Durable delta commits                    |
| Tlog replay (manager path)  | ✅     | Root verified during replay              |
| Graph delta log             | ✅     | Append-only graph layer                  |
| Journal structure           | ⚠️      | Present but not integrated with recovery |
| Crash recovery guarantees   | ⚠️      | Partial; relies on replay correctness    |

---

## Performance Characteristics

| Feature                       | Status | Notes                                |
|-------------------------------+--------+--------------------------------------|
| Flat contiguous Merkle layout | ✅     | Cache-friendly                       |
| Batched dirty rehash          | ✅     | Multiple leaf updates deduped        |
| Parallel parent hashing       | ✅     | Level-isolated mutation              |
| Parallel commit batch         | ❌     | Current implementation is sequential |
| GPU-ready structure           | ❌     | No GPU implementation present        |

---

## What Is Not Implemented

- True zero-copy page hashing
- Parallel delta commit execution
- GPU compute path
- Journal-backed recovery
- Crash-consistent multi-layer durability protocol

---

## What Is Architecturally Solid

- Deterministic Merkle model
- Domain separation discipline
- Incremental + full recompute equivalence
- Clean proof object layering
- Append-only log design

