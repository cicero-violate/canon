| Category        | Feature                       | Status | Notes                                |
| --------------- | ----------------------------- | ------ | ------------------------------------ |
| **State Model** | Page-based canonical state    | ✅      | 4096-byte logical pages              |
|                 | Flat array Merkle tree        | ✅      | Full binary tree in `Vec<Hash>`      |
|                 | Configurable capacity         | ✅      | `with_capacity(max_leaves)`          |
|                 | Auto-resize + rehash          | ✅      | `next_power_of_two` growth           |
|                 | Deterministic root hash       | ✅      | Index 1 in flat tree                 |
|                 | Zero-copy page hashing        | ✅      | Hash directly from `PageStore`       |
|                 | Domain-separated hashing      | ✅      | LEAF / NODE / DELTA / EVENT prefixes |
|                 | Incremental leaf updates      | ✅      | Dirty-leaf marking                   |
|                 | Batched level rebuild         | ✅      | Level-by-level rebuild logic         |
|                 | Level-sliced parallel rebuild | ✅      | Rayon-based parallel layer hashing   |
|                 | Merkle inclusion proofs       | ✅      | Proof path generation                |
|                 | Stateless proof verification  | ✅      | External verifier API                |
|                 | Internal consistency check    | ✅      | Root re-derived validation           |


| Feature                       | Status | Notes                   |
| ----------------------------- | ------ | ----------------------- |
| In-memory PageStore           | ✅      | Default mode            |
| Memory-mapped PageStore       | ✅      | `memmap2` backend       |
| Capacity guard for mmap       | ✅      | Bounds assertion        |
| Automatic in-memory growth    | ✅      | `Vec` resize fallback   |
| Flush on commit               | ✅      | Ensures mmap durability |
| Transaction log (append-only) | ✅      | Durable delta commits   |
| Graph delta log               | ✅      | Append-only graph layer |
| Journal structure             | ⚠️     | Present but not active  |

| Feature                            | Status | Notes                         |
| ---------------------------------- | ------ | ----------------------------- |
| Flat contiguous Merkle node layout | ✅      | Cache-friendly                |
| GPU-transition-ready structure     | ✅      | Level-sliced design           |
| O(log n) update complexity         | ✅      | Per delta                     |
| Batched rehash support             | ✅      | Dirty leaf accumulation       |
| Parallel commit batch              | ✅      | Rayon parallel delta commit   |
| Parallel level hashing             | ✅      | Safe, level-isolated mutation |
