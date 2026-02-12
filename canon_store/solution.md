**Terms**

Epoch					— a fixed time window after which state is sealed and hashed
Epoch Chain				— a sequence where each epoch root includes the previous, forming a tamper-evident history
Partition				— a named slice of memory with its own independent Merkle tree
Leaf					— a single hashed data entry at the bottom of a Merkle tree

Merkle Tree				— a binary tree where every parent is the hash of its two children
Merkle Root				— the single top hash that summarizes an entire tree

Root-of-Roots			— a Merkle tree whose leaves are other Merkle roots

Sibling Hash			— the partner hash needed alongside a leaf to recompute the path to the root
Proof Path				— the minimal set of sibling hashes needed to verify one leaf against the root
Partial Verification	— checking one partition without loading any other partition
Tamper Evidence			— the property that any change anywhere propagates up to invalidate the root
Checkpoint				— a saved epoch root file that lets you verify state without replaying all tlogs
Replay					— reconstructing current state by re-applying all transaction logs from a snapshot forward
Snapshot				— a full frozen copy of state at a point in time, used as a replay starting point
Tlog					— a transaction log recording every delta after the last snapshot

---

**Steps**

Step 1 — Every write inside a partition gets hashed into that partition's own Merkle tree during the epoch
Step 2 — At epoch end each partition produces one root hash summarizing everything inside it
Step 3 — Those partition roots become the leaves of a second smaller tree called the root-of-roots
Step 4 — The root-of-roots is hashed into one final value that fingerprints the entire system at that moment
Step 5 — That final hash is written to a small epoch root file such as epoch_0042.root
Step 6 — The next epoch root takes the previous epoch root as an input so history cannot be silently rewritten
Step 7 — Any pipeline can verify one partition by reading only its root, its sibling hashes, and the epoch root file
Step 8 — No pipeline needs to load any other partition to confirm its own partition is untampered
