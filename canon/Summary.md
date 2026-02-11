Here is the **clean summary table** of what now exists, with **no narrative drift** and **no forward pressure**:

| Layer                    | Responsibility             | What It Can Do                                              | What It Cannot Do                                  | Determinism Status    |
| ------------------------ | -------------------------- | ----------------------------------------------------------- | -------------------------------------------------- | --------------------- |
| **Kernel**               | Admission gate             | Accept / reject admissions based on judgment outcome        | Interpret deltas, infer meaning, execute semantics | **Absolute**          |
| **Judgment**             | State predicate evaluation | Compare hashes, check invariant flags, allow meta-decisions | Modify state, emit deltas, reinterpret history     | **Absolute**          |
| **StateLog**             | Append-only history        | Record deltas, compute state hash, replay history           | Mutate past, reorder effects                       | **Absolute**          |
| **InvariantRegistry**    | Law activation             | Track invariant status, gate meta-decisions                 | Invent laws, bypass proof checks                   | **Absolute**          |
| **ProofArtifact**        | External attestation       | Bind deltas to scopes and hashes                            | Execute logic, alter outcomes                      | **Absolute**          |
| **Admission**            | Change bundle              | Group deltas under a single judgment                        | Partially apply changes                            | **Absolute**          |
| **Canon (outer system)** | Semantics & evolution      | Define meaning, pipelines, learning, planning               | Apply changes directly                             | **Non-authoritative** |
| **Learning**             | Rule proposal              | Emit candidate invariants, drafts                           | Activate invariants unilaterally                   | **Gated**             |
| **Execution world**      | Effects                    | Produce observations                                        | Alter state without admission                      | **Observed only**     |


> *All state change flows through judgment-gated admission into an append-only log.*

Nothing in this table is aspirational.
Everything listed already exists in your code.
