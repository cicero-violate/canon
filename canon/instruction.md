Goal : Implement two capability layers: (1) **Layer 4** introduces a learnable policy that updates across `TickEpoch` boundaries using reward feedback, and (2) **Layer 5** adds constrained goal self-modification gated by invariant proofs and the existing Judgment/Proposal pipeline.

---
## Instruction Plan

---
### Layer 4 — Learning: Policy Parameter Updates

---
#### L1 — Define `PolicyParameters` in `ir/policy.rs` (new file)

```
Action: CreateFile
file: src/ir/policy.rs
```

```
Action: WriteStruct
file: src/ir/policy.rs
struct_name: PolicyParameters
fields: [
  "id: PolicyParameterId",
  "version: u32",
  "epoch: TickEpochId",
  "learning_rate: f64",
  "discount_factor: f64",
  "entropy_weight: f64",
  "reward_baseline: f64",
  "proof_id: Option<ProofId>"
]
derives: [Debug, Clone, Serialize, Deserialize, JsonSchema]
```

```
Action: WriteTypeAlias
file: src/ir/ids.rs
alias_name: PolicyParameterId
target_type: String
```

**Constraint:** `PolicyParameters` is append-only — no mutation of existing records, only new versions pushed. This mirrors the `Delta` append-only invariant already enforced by `DeltaAppendOnly` in `validate/rules.rs`.

---

#### L2 — Add `PolicySnapshot` to `ir/timeline.rs`

```
Action: AddField
file: src/ir/timeline.rs
struct_name: TickEpoch
field_name: policy_snapshot
field_type: Option<PolicySnapshot>
visibility: pub
```

```
Action: WriteStruct
file: src/ir/timeline.rs
struct_name: PolicySnapshot
fields: [
  "epoch: TickEpochId",
  "parameters: PolicyParameters",
  "reward_at_snapshot: f64",
  "delta_ids: Vec<DeltaId>"
]
derives: [Debug, Clone, Serialize, Deserialize, JsonSchema]
```

**Rationale:** $\Pi_\tau$ is captured at each epoch boundary so the audit trail is complete. The `delta_ids` field links the snapshot back to the structural evidence that produced $R_\tau$.

---

#### L3 — Implement `update_policy()` in `runtime/policy_updater.rs` (new file)

```
Action: CreateFile
file: src/runtime/policy_updater.rs
```

```
Action: WriteStruct
file: src/runtime/policy_updater.rs
struct_name: PolicyUpdater
fields: ["ir: *mut CanonicalIr"]
```

```
Action: WriteFunction
file: src/runtime/policy_updater.rs
signature: pub fn update_policy(current: &PolicyParameters, reward: f64) -> PolicyParameters
context_symbols: [PolicyParameters, TickEpochId]
```

The function body encodes:

$$\Pi_{\tau+1} = \Pi_\tau + \alpha \cdot \nabla R_\tau$$

where $\alpha$ = `learning_rate`. For the initial implementation use a rule-based linear step: each parameter is nudged by `learning_rate * (reward - reward_baseline)`. The result is a **new** `PolicyParameters` with `version` incremented — never mutated in place.

```
Action: WriteEnum
file: src/runtime/policy_updater.rs
enum_name: PolicyUpdateError
variants: [EpochNotFound, NoPriorPolicy, InvalidLearningRate]
derives: [Debug]
```

---

#### L4 — Add `RewardCollapseDetector` to `validate/check_execution.rs`

```
Action: WriteFunction
file: src/validate/check_execution.rs
signature: fn check_reward_collapse(ir: &CanonicalIr, violations: &mut Vec<Violation>)
context_symbols: [CanonicalIr, Violation, CanonRule, TickEpoch, PolicySnapshot]
```

**Logic:** Iterate over consecutive epoch pairs $(\tau-1, \tau)$. Extract $R_{\tau-1}$ and $R_\tau$ from their `PolicySnapshot`. Emit a `Violation` with `CanonRule::RewardMonotonicity` if:

$$\delta_\tau = R_\tau - R_{\tau-1} < -\epsilon, \quad \epsilon = 0.05$$

```
Action: AddMatchArm
file: src/validate/check_execution.rs
fn_name: check
variant: RewardCollapse
handler_description: delegate to check_reward_collapse
```

```
Action: AddEnumVariant
file: src/validate/rules.rs
enum_name: CanonRule
variant: RewardCollapseDetected
```

Wire `RewardCollapseDetected` into `CanonRule::code()` and `CanonRule::text()`.

---

#### L4 (wiring) — Register `ir/policy.rs` in `ir/core.rs` and `ir/mod.rs`

```
Action: AddField
file: src/ir/core.rs
struct_name: CanonicalIr
field_name: policy_parameters
field_type: Vec<PolicyParameters>
visibility: pub
```

```
Action: AddModDecl
file: src/ir/mod.rs
mod_name: policy
visibility: pub
```

---

### Layer 5 — Goal Mutation: Constrained Self-Modification

---

#### G1 — Define `GoalMutation` and `GoalDriftMetric` in `ir/goals.rs` (new file)

```
Action: CreateFile
file: src/ir/goals.rs
```

```
Action: WriteStruct
file: src/ir/goals.rs
struct_name: GoalMutation
fields: [
  "id: GoalMutationId",
  "original_goal: String",
  "proposed_goal: String",
  "invariant_proof_ids: Vec<ProofId>",
  "proposal_id: ProposalId",
  "judgment_id: Option<JudgmentId>",
  "status: GoalMutationStatus"
]
derives: [Debug, Clone, Serialize, Deserialize, JsonSchema]
```

The invariant requirement encodes:

$$\text{accept}(g') \iff \forall i \in \text{invariant\_proof\_ids}: I_i(g') = \top$$

```
Action: WriteEnum
file: src/ir/goals.rs
enum_name: GoalMutationStatus
variants: [Proposed, Accepted, Rejected]
derives: [Debug, Clone, Serialize, Deserialize, JsonSchema]
```

```
Action: WriteStruct
file: src/ir/goals.rs
struct_name: GoalDriftMetric
fields: [
  "mutation_id: GoalMutationId",
  "cosine_distance: f64",
  "keyword_overlap: f64",
  "within_bound: bool",
  "bound_theta: f64"
]
derives: [Debug, Clone, Serialize, Deserialize, JsonSchema]
```

`GoalDriftMetric` encodes:

$$\text{drift}(g, g') \leq \theta \implies \text{within\_bound} = \top$$

```
Action: WriteTypeAlias
file: src/ir/ids.rs
alias_name: GoalMutationId
target_type: String
```

---

#### G2 — Implement `mutate_goal()` in `evolution/goal_mutation.rs` (new file)

```
Action: CreateFile
file: src/evolution/goal_mutation.rs
```

```
Action: WriteFunction
file: src/evolution/goal_mutation.rs
signature: pub fn mutate_goal(ir: &CanonicalIr, mutation: &GoalMutation) -> Result<CanonicalIr, GoalMutationError>
context_symbols: [CanonicalIr, GoalMutation, GoalMutationError, Proof, ProofScope]
```

**Logic flow:**

1. For each `proof_id` in `mutation.invariant_proof_ids`, call `ensure_proof_exists()` (already in `decision/accept/proposal_checks.rs`).
2. Compute `GoalDriftMetric`. If `within_bound == false`, return `Err(GoalMutationError::DriftExceeded)`.
3. Check `mutation.status == GoalMutationStatus::Accepted` (requires a completed `Judgment`). If not, return `Err(GoalMutationError::NotAccepted)`.
4. Apply the mutation to `ir` and return the new IR.

$$\text{mutate\_goal}(g, \Pi) = g' \iff I(g') = \top \wedge \text{Judgment}(g') = \text{Accept} \wedge \text{drift}(g,g') \leq \theta$$

```
Action: WriteEnum
file: src/evolution/goal_mutation.rs
enum_name: GoalMutationError
variants: [MissingProof, InvariantViolated, DriftExceeded, NotAccepted, UnknownMutation]
derives: [Debug]
```

---

#### G3 — Add `compute_goal_drift()` helper in `ir/goals.rs`

```
Action: WriteFunction
file: src/ir/goals.rs
signature: pub fn compute_goal_drift(original: &str, proposed: &str, theta: f64) -> GoalDriftMetric
context_symbols: [GoalDriftMetric, GoalMutationId]
```

Uses keyword-overlap as a proxy for cosine distance (no embedding dependency). Overlap:

$$\text{keyword\_overlap}(g, g') = \frac{|W(g) \cap W(g')|}{|W(g) \cup W(g')|}$$

where $W(x)$ is the word-token set of $x$. Sets `within_bound = keyword_overlap >= 1.0 - \theta`.

---

#### G4 — Wire goal mutation through `Judgment` / `Proposal` pipeline

**Step 1:** Register `GoalMutation` collection in `CanonicalIr`:

```
Action: AddField
file: src/ir/core.rs
struct_name: CanonicalIr
field_name: goal_mutations
field_type: Vec<GoalMutation>
visibility: pub
```

**Step 2:** Add `GoalMutationId` to `ir/ids.rs` (done in G1 above).

**Step 3:** Add a validation check:

```
Action: WriteFunction
file: src/validate/check_proposals.rs
signature: fn check_goal_mutations(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>)
context_symbols: [CanonicalIr, Indexes, Violation, CanonRule, GoalMutation, GoalMutationStatus]
```

Emits a violation if any `GoalMutation` has `status == Accepted` but `judgment_id == None`, or if any cited `invariant_proof_ids` do not exist in `ir.proofs`.

**Step 4:** Add new `CanonRule` variants:

```
Action: AddEnumVariant
file: src/validate/rules.rs
enum_name: CanonRule
variant: GoalMutationRequiresJudgment
```

```
Action: AddEnumVariant
file: src/validate/rules.rs
enum_name: CanonRule
variant: GoalMutationInvariantMissing
```

**Step 5:** Register `ir/goals.rs` module:

```
Action: AddModDecl
file: src/ir/mod.rs
mod_name: goals
visibility: pub
```

**Step 6:** Register `evolution/goal_mutation.rs` module:

```
Action: AddModDecl
file: src/evolution/mod.rs
mod_name: goal_mutation
visibility: pub
```

---

### Execution Order

$$\text{L1} \to \text{L2} \to \text{L3} \to \text{L4} \to \text{G1} \to \text{G2} \to \text{G3} \to \text{G4}$$

Each step has a hard dependency on the prior: `PolicyParameters` must exist before `PolicySnapshot` references it; `GoalMutation` must exist before `mutate_goal()` can be written; all new `CanonRule` variants must land before the check functions that emit them.

After all file edits are complete, the final action is:

```
Action: CargoCheck
```

Resolve any `FixCompileError` children before marking the plan complete. No tests, no summary files, no documentation files are to be created.
