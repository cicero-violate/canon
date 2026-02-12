sovereign-memory/
├── Cargo.toml                          ← workspace root
│
├── crates/
│   │
│   ├── merkle-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  ← pub mod declarations
│   │       ├── hash.rs                 ← SHA-256 wrappers
│   │       ├── tree.rs                 ← tree construction, leaf insertion
│   │       ├── root.rs                 ← root computation
│   │       └── proof.rs                ← proof path generation and verification
│   │
│   ├── partition-store/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── partition.rs            ← named partition open/read/write
│   │       ├── schema.rs               ← per-partition data schema definitions
│   │       └── access.rs               ← permission scoping per pipeline
│   │
│   ├── tlog-writer/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── writer.rs               ← append-only delta recording
│   │       ├── entry.rs                ← tlog entry format and serialization
│   │       └── rotate.rs               ← tlog file rotation policy
│   │
│   ├── snapshot-manager/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── freeze.rs               ← freezes partition state to canonical bin
│   │       ├── compact.rs              ← compaction policy, tlog absorption
│   │       └── restore.rs              ← reconstruct partition from snapshot + tlogs
│   │
│   ├── epoch-sealer/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── sealer.rs               ← collects partition roots, triggers seal
│   │       ├── root_of_roots.rs        ← builds second-level merkle from partition roots
│   │       └── epoch_file.rs           ← writes epoch_NNNN.root files
│   │
│   ├── proof-store/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── store.rs                ← save and retrieve proof path files
│   │       └── shard.rs                ← 2-hex-prefix sharded directory management
│   │
│   ├── verifier/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── verify.rs               ← stateless leaf + proof + root verification
│   │       └── result.rs               ← VerifyResult type, error kinds
│   │
│   └── root-chain/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── chain.rs                ← epoch root chain maintenance
│           ├── detect.rs               ← chain break detection
│           └── trusted_root.rs         ← exposes current trusted root to other crates
│
└── data/
    │
    ├── snapshots/
    │   ├── governance/
    │   │   └── canonical_state.bin
    │   ├── episodic/
    │   │   └── canonical_state.bin
    │   ├── semantic/
    │   │   └── canonical_state.bin
    │   └── working/
    │       └── canonical_state.bin
    │
    ├── tlogs/
    │   ├── governance/
    │   │   └── 2026-02-12T00/
    │   │       ├── 000001.tlog
    │   │       └── 000002.tlog
    │   ├── episodic/
    │   │   └── 2026-02-12T00/
    │   │       └── 000001.tlog
    │   ├── semantic/
    │   │   └── 2026-02-12T00/
    │   │       └── 000001.tlog
    │   └── working/
    │       └── 2026-02-12T00/
    │           └── 000001.tlog
    │
    ├── proofs/
    │   ├── 00/
    │   │   └── a7f9e4...json
    │   ├── 06/
    │   │   └── 336fbd...json
    │   └── 14/
    │       └── d7b2e5...json
    │
    └── epochs/
        ├── epoch_0001.root
        ├── epoch_0002.root
        └── epoch_0042.root
