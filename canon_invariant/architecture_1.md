| Category              | Domain  | Rule / Constraint                         | Enforced By Function(s)                                                                                 |
| --------------------- | ------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Policy / Doctrine     | Kernel  | Intent Radius Bound                       | (Not implemented in `validate` module)                                                                  |
| Policy / Doctrine     | Law     | Single Source of Truth                    | `helpers::index_by_id`, `helpers::build_indexes`                                                        |
| Policy / Doctrine     | Law     | Judgment Decides Policy                   | `check_proposals::check_judgments`, `check_execution::check_plans`                                      |
| Policy / Doctrine     | Law     | Law Violations Are Hard Errors            | `validate_ir` (returns `ValidationErrors`)                                                              |
| Policy / Doctrine     | Meta    | Performance < Correctness                 | (Architectural doctrine, not validated here)                                                            |
| Policy / Doctrine     | Meta    | Correctness < Law                         | `check_deltas::check_deltas` (proof_scope_allows), `helpers::proof_scope_allows`                        |
| Policy / Doctrine     | Meta    | System Must Choose Rightly                | `check_proposals::check_judgments`, `check_execution::check_plans`                                      |
| --------------------- | ------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Category              | Domain  | Rule / Constraint                         | Enforced By Function(s)                                                                                 |
| --------------------- | ------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Pure State Invariant  | Planner | DAG Acyclicity                            | `check_graphs::check_call_graph`, `check_graphs::check_module_graph`, `check_graphs::check_tick_graphs` |
| Pure State Invariant  | Planner | No Orphan Nodes                           | `check_execution::check_plans`, `check_execution::check_executions`                                     |
| Pure State Invariant  | Planner | Unique Task IDs                           | `helpers::index_by_id`                                                                                  |
| Pure State Invariant  | Planner | Bounded Depth                             | `check_execution::check_epochs` (cycle prevention)                                                      |
| Pure State Invariant  | Planner | Bounded Branching                         | (Not explicitly enforced)                                                                               |
| Pure State Invariant  | IR      | Unique Node IDs                           | `helpers::index_by_id`                                                                                  |
| Pure State Invariant  | IR      | Unique Edge IDs                           | (Implicit via indexing; no explicit edge-id uniqueness check)                                           |
| Pure State Invariant  | IR      | No Dangling References                    | All `check_*` functions referencing `idx.*.get(...)`                                                    |
| Pure State Invariant  | IR      | Schema Validity                           | `validate_ir` orchestration + structural checks                                                         |
| Pure State Invariant  | IR      | No Cyclic Module Imports                  | `check_graphs::check_module_graph`                                                                      |
| Pure State Invariant  | IR      | Directory Width Bound                     | (Not implemented)                                                                                       |
| Pure State Invariant  | Memory  | Merkle Root Valid                         | (Not implemented in this module)                                                                        |
| Pure State Invariant  | Memory  | Hash Integrity                            | (Not implemented here)                                                                                  |
| Pure State Invariant  | Memory  | Unique Delta IDs                          | `helpers::index_by_id` (deltas)                                                                         |
| Pure State Invariant  | Kernel  | Non-Negative Balance                      | (Not implemented here)                                                                                  |
| Pure State Invariant  | Runtime | No Dangling Shell Sessions                | (Not implemented here)                                                                                  |
| Pure State Invariant  | Meta    | No Hidden State                           | All structural checks requiring explicit references                                                     |
| --------------------- | ------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Category              | Domain  | Rule / Constraint                         | Enforced By Function(s)                                                                                 |
| --------------------- | ------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Transition Constraint | Planner | Dependency Satisfaction (Exec ⇒ Deps Met) | `check_execution::check_plans`, `check_execution::check_executions`                                     |
| Transition Constraint | Planner | Monotonic Completion                      | `check_deltas::check_applied_records`                                                                   |
| Transition Constraint | IR      | Type Soundness Enforcement                | `check_artifacts::check_impls`, `check_artifacts::check_functions`                                      |
| Transition Constraint | Memory  | Append-Only                               | `check_deltas::check_deltas` (`delta.append_only`)                                                      |
| Transition Constraint | Memory  | Replay Determinism                        | `check_deltas::check_applied_records`                                                                   |
| Transition Constraint | Memory  | Epoch Ordering                            | `check_execution::check_epochs`                                                                         |
| Transition Constraint | Kernel  | Capability Satisfaction                   | `check_artifacts::check_impls`, `check_artifacts::check_functions`                                      |
| Transition Constraint | Kernel  | Deterministic Transition                  | `check_artifacts::check_functions` (contract.deterministic)                                             |
| Transition Constraint | Kernel  | State Immutability                        | `check_deltas::check_deltas` (append_only)                                                              |
| Transition Constraint | Kernel  | Resource Bound                            | `check_graphs::check_call_graph` (acyclic), `check_execution::check_gpu` (lane > 0)                     |
| Transition Constraint | Law     | Proof Required for Critical Delta         | `check_deltas::check_deltas`                                                                            |
| Transition Constraint | Law     | Signature Required for Mutation           | (Not implemented here)                                                                                  |
| Transition Constraint | Law     | Cross-Layer Dependency Rule               | `check_graphs::check_call_graph` (module_has_permission)                                                |
| Transition Constraint | LLM     | Schema Valid Output                       | `check_proposals::check_proposals`                                                                      |
| Transition Constraint | LLM     | No Direct Mutation                        | `check_deltas::check_deltas` (EffectsAreDeltas)                                                         |
| Transition Constraint | LLM     | Snapshot Grounding                        | `check_execution::check_ticks`                                                                          |
| Transition Constraint | LLM     | Bounded Recursion                         | `check_graphs::check_call_graph` (acyclic)                                                              |
| Transition Constraint | LLM     | Cost Bound                                | (Not implemented here)                                                                                  |
| Transition Constraint | Runtime | Controlled Side Effects                   | `check_artifacts::check_functions` (effects_are_deltas)                                                 |
| Transition Constraint | Runtime | Deterministic IO Capture                  | `check_artifacts::check_functions` (explicit IO + deterministic)                                        |
| Transition Constraint | Runtime | Idempotent Commands                       | `check_deltas::check_deltas` (append-only discipline)                                                   |
| Transition Constraint | Runtime | PTY Isolation                             | (Not implemented here)                                                                                  |
| Transition Constraint | Meta    | Closed Loop (Observe → Commit → Log)      | `helpers::pipeline_stage_allows`, `check_deltas::check_deltas`                                          |
| Transition Constraint | Meta    | Replayable                                | `check_deltas::check_applied_records`, `check_execution::check_epochs`                                  |
| --------------------- | ------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------- |
