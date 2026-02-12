| Crate              | File               | Structs / Types Defined                                                           |
| ------------------ | ------------------ | --------------------------------------------------------------------------------- |
| **kernel**         | `lib.rs`           | `pub use state::CanonicalState`, `pub use delta::Delta`, `pub use kernel::Kernel` |
|                    | `state.rs`         | `CanonicalState`, `StateSlice`                                                    |
|                    | `delta.rs`         | `Delta`, `DeltaId`, `DeltaMask`, `DeltaError`                                     |
|                    | `apply.rs`         | `ApplyResult`, `StateTransitionError`                                             |
|                    | `invariant.rs`     | `InvariantSet`, `InvariantViolation`                                              |
|                    | `hash.rs`          | `RootHash`, `HashError`                                                           |
|                    | `kernel.rs`        | `Kernel`, `KernelConfig`                                                          |
|                    | `error.rs`         | `KernelError`                                                                     |
| **memory**         | `lib.rs`           | `pub use ledger::DeltaLedger`, `pub use snapshot::SnapshotManager`                |
|                    | `ledger.rs`        | `DeltaLedger`, `LedgerEntry`                                                      |
|                    | `snapshot.rs`      | `SnapshotManager`, `SnapshotMetadata`                                             |
|                    | `replay.rs`        | `ReplayEngine`, `ReplayError`                                                     |
|                    | `storage.rs`       | `StorageBackend`, `StorageConfig`                                                 |
|                    | `memory_error.rs`  | `MemoryError`                                                                     |
| **judgment**       | `lib.rs`           | `pub use validator::JudgmentValidator`                                            |
|                    | `radius.rs`        | `IntentRadius`, `RadiusCalculator`                                                |
|                    | `policy.rs`        | `Policy`, `PolicyDecision`                                                        |
|                    | `proof.rs`         | `JudgmentProof`, `ProofScope`                                                     |
|                    | `validator.rs`     | `JudgmentValidator`, `JudgmentResult`                                             |
|                    | `error.rs`         | `JudgmentError`                                                                   |
| **planner**        | `lib.rs`           | `pub use planner::Planner`                                                        |
|                    | `planner.rs`       | `Planner`, `Plan`, `PlanStep`                                                     |
|                    | `intent.rs`        | `Intent`, `IntentPayload`                                                         |
|                    | `llm_adapter.rs`   | `LLMAdapter`, `LLMConfig`                                                         |
|                    | `heuristics.rs`    | `HeuristicEngine`                                                                 |
|                    | `planner_error.rs` | `PlannerError`                                                                    |
| **runtime**        | `lib.rs`           | `pub use coordinator::RuntimeCoordinator`                                         |
|                    | `boot.rs`          | `BootLoader`, `BootConfig`                                                        |
|                    | `loop.rs`          | `ExecutionLoop`                                                                   |
|                    | `coordinator.rs`   | `RuntimeCoordinator`, `ExecutionContext`                                          |
|                    | `runtime_error.rs` | `RuntimeError`                                                                    |
| **projection**     | `lib.rs`           | `pub use graph::GraphProjection`                                                  |
|                    | `graph.rs`         | `GraphProjection`, `ProjectionNode`, `ProjectionEdge`                             |
|                    | `metrics.rs`       | `MetricSnapshot`, `MetricCollector`                                               |
|                    | `export.rs`        | `GraphExporter`, `ExportFormat`                                                   |
| **substrate**      | `shell.rs`         | `ShellExecutor`, `ShellCommand`                                                   |
|                    | `gpu.rs`           | `GpuExecutor`, `GpuLaunchConfig`                                                  |
|                    | `network.rs`       | `NetworkClient`, `NetworkRequest`                                                 |
|                    | `fs.rs`            | `FilesystemAdapter`                                                               |
| **proof_gate**     | `bridge.rs`        | `LeanBridge`, `ProofCertificate`                                                  |
|                    | `certificate.rs`   | `CertificateStore`, `CertificateHash`                                             |
|                    | `obligation.rs`    | `ProofObligation`                                                                 |
|                    | `proof_error.rs`   | `ProofGateError`                                                                  |
| **bin/canon-node** | `main.rs`          | `fn main()`, `NodeConfig`                                                         |
