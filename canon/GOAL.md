### Layer 0 — Existing Infrastructure (already built)

The `canon` IR already provides:

- `CanonicalIr` — the central state graph
- `Delta` / `DeltaAdmission` / `AppliedDeltaRecord` — append-only mutation ledger
- `Tick` / `TickEpoch` / `TickGraph` — execution scheduling
- `ExecutionRecord` + `ExecutionEvent` — runtime outcome capture
- `Proof` / `Judgment` / `Predicate` — formal verification hooks
- `FunctionExecutor` / `TickExecutor` — bytecode runtime
- `Plan` — goal-directed execution steps
- `Learning` — proposal-coupled learning artifacts

---

### Layer 1 — Foundation: Scalar Utility

| #  | Task                                                                  | IR Target                     | Notes                              |
|----+-----------------------------------------------------------------------+-------------------------------+------------------------------------|
| F1 | Define `UtilityFunction` struct with explicit scalar formula          | `ir/utility.rs` (new)         | Must be serializable and versioned | DONE
| F2 | Add `reward: f64` field to `ExecutionRecord`                          | `ir/timeline.rs`              | Logged after every tick            |
| F3 | Add `reward_deltas: Vec<RewardDelta>` to `CanonicalIr`                | `ir/core.rs`                  | Append-only, delta-tracked         | DONE
| F4 | Implement `compute_reward()` in `runtime/tick_executor.rs`            | Runs post-tick                | Returns `f64`, stored on record    | DONE
| F5 | Add `MonotonicityCheck` — enforce $U(s_{t+1}) \geq U(s_t) - \epsilon$ | `validate/check_execution.rs` | New rule in `CanonRule`            | DONE

---

### Layer 2 — World Model: Predictive State

| #  | Task                                                                    | IR Target                          | Notes                                 |
|----+-------------------------------------------------------------------------+------------------------------------+---------------------------------------|
| W1 | Define `WorldModel` struct with state snapshot + prediction head        | `ir/world_model.rs` (new)          | Stores $\hat{s}_{t+k}$ rollouts       |
| W2 | Add `PredictionRecord` — stores $\hat{s}$ vs $s$, computes $\epsilon_t$ | `ir/world_model.rs`                | Per-tick                              | DONE
| W3 | Implement multi-step rollout in `runtime/rollout.rs` (new)              | Calls `TickExecutor` speculatively | Depth-limited                         | 
| W4 | Add world-model update step post-execution                              | `runtime/tick_executor.rs`         | Updates `WorldModel` in `CanonicalIr` |
| W5 | Track entropy reduction $H_\tau$ per epoch                              | `ir/timeline.rs` (`TickEpoch`)     | Aggregate of $\log \epsilon_t$        |

---

### Layer 3 — Planning: Reward-Optimizing Search

| #  | Task                                                                         | IR Target                            | Notes                              |
|----+------------------------------------------------------------------------------+--------------------------------------+------------------------------------|
| P1 | Add `search_depth: u32` and `utility_estimate: f64` to `Plan`                | `ir/timeline.rs`                     | Replaces correctness-only planning |
| P2 | Implement candidate action scoring via rollout in `runtime/planner.rs` (new) | Scores $\mathbb{E}[R]$ per candidate | Uses `WorldModel`                  |
| P3 | Record planner decision rationale with utility estimate on `Plan`            | `ir/timeline.rs`                     | Full audit trail                   |
| P4 | Integrate planner into `TickExecutor` pre-execution step                     | `runtime/tick_executor.rs`           | Plan selection before dispatch     |

---

### Layer 4 — Learning: Policy Parameter Updates

| #  | Task                                                          | IR Target                         | Notes                                |
|----+---------------------------------------------------------------+-----------------------------------+--------------------------------------|
| L1 | Define `PolicyParameters` struct — separate from static rules | `ir/policy.rs` (new)              | Versioned, append-only               |
| L2 | Add `PolicySnapshot` per `TickEpoch`                          | `ir/timeline.rs`                  | Stores $\Pi_\tau$                    |
| L3 | Implement `update_policy()` — gradient or rule-based step     | `runtime/policy_updater.rs` (new) | $\Pi_{\tau+1} = f(\Pi_\tau, R_\tau)$ |
| L4 | Add `RewardCollapseDetector` — regression check across epochs | `validate/check_execution.rs`     | Alert if $R_\tau \ll R_{\tau-1}$     |

---

### Layer 5 — Goal Mutation: Constrained Self-Modification

| #  | Task                                                                 | IR Target                          | Notes                         |
|----+----------------------------------------------------------------------+------------------------------------+-------------------------------|
| G1 | Define `GoalMutation` struct with proposed $g'$ and invariant proofs | `ir/goals.rs` (new)                | Must cite proof IDs           |
| G2 | Implement `mutate_goal()` — constrained by `canon_invariant/`        | `evolution/goal_mutation.rs` (new) | Rejects if any $I(g') = \bot$ |
| G3 | Add `GoalDriftMetric` — measures alignment between $g'$ and $g$      | `ir/goals.rs`                      | Prevents silent misalignment  |
| G4 | Wire goal mutation through the `Judgment` / `Proposal` pipeline      | `decision/`                        | Requires formal acceptance    |

---

### Cross-Cutting Concerns

| #  | Task                                                                        | Notes                                                                                        |
|----+-----------------------------------------------------------------------------+----------------------------------------------------------------------------------------------|
| X1 | New `CanonRule` entries for each layer                                      | `validate/rules.rs` — `RewardLogging`, `WorldModelUpdate`, `PolicySnapshot`, `GoalInvariant` |
| X2 | Schema evolution — all new structs need `#[derive(JsonSchema)]`             | Required for `validate` + `ingest`                                                           |
| X3 | `delta_emitter.rs` — new `DeltaPayload` variants for each new artifact type | `ir/delta.rs`                                                                                |
| X4 | `repomap` — currently at 632 symbols; expect +80–120 new symbols            | Monitor token budget                                                                         |
