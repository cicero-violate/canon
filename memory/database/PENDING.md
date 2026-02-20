🚀 What’s Coming Next
🔥 Immediate Upgrade (Next Logical Step)

1️⃣ True Incremental Parallel Rehash
Current: rebuild full upper tree when dirty
Next:

Track dirty subtrees
Recompute only affected branches
Maintain level buckets
Enable true parallel level hashing without full rebuild

2️⃣ SIMD Hash Acceleration
Switch SHA backend to SIMD-optimized implementation
Auto-detect CPU SHA extensions
Prepare abstraction layer for GPU hash backend

3️⃣ Merkle Subtree Persistence
Persist internal nodes to disk
Separate node file from page file
Enable fast cold-start recovery

4️⃣ Journal Replay Boot Recovery
Replay TLog on startup
Reconstruct canonical state
Validate final root against persisted value

5️⃣ GPU Hash Pipeline (Architecture Phase)
Abstract hash function behind trait
Level-sliced buffer upload
Batch hash kernel per level
Maintain deterministic ordering

6️⃣ Proof Compression
Compress proof vectors
Bitmask sibling presence encoding
Potential recursive proof scheme

7️⃣ Lock-Free Delta Ingestion Queue
Separate ingestion from commit
Concurrent delta buffering
Single writer canonical commit loop

