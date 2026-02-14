# Tier 7 Architecture — Development Plan

## What Was Built (Phases 1–5 Complete)

### Phase 1 — L3 Capability Graph (stateless call surface)
- `src/agent/capability.rs` — CapabilityGraph, CapabilityNode, IrField, entropy(), trust_scores()
- `src/agent/slice.rs`      — build_ir_slice(): enforces read contracts at extraction time
- `src/agent/call.rs`       — AgentCallInput, AgentCallOutput, AgentCallError
- `src/agent/dispatcher.rs` — AgentCallDispatcher: trust-gated topological call builder

### Phase 2 — Lyapunov Safety Gate
- `src/evolution/lyapunov.rs` — TopologyFingerprint, check_topology_drift(), LyapunovError
- Wired into `apply_deltas()` — Structure deltas gated by ||Φ(G)-G||_F ≤ θ (θ=0.15)

### Phase 3 — RefactorProposal Pipeline
- `src/agent/refactor.rs`  — RefactorProposal, RefactorKind, RefactorTarget
- `src/agent/pipeline.rs`  — run_pipeline(): Observe→Reason→Prove→Judge→Mutate

### Phase 4 — Reward Learning Loop
- `src/agent/reward.rs` — RewardLedger, NodeOutcome, EMA reward per node
- trust_threshold_for(): positive EMA → permissive, negative EMA → scrutiny
- Feeds update_policy() → PolicyParameters

### Phase 5 — L7 Meta-Tick (Φ: G → G')
- `src/agent/meta.rs` — run_meta_tick(): CapabilityGraph self-rewrite
- GraphMutation: RemoveNode, AddEdge, RemoveEdge, PromoteToMetaAgent
- Safety: entropy bound, MIN_NODES=3, connectivity check

---

## The Full Tier 7 Loop (current state)
```
every tick:
  AgentCallDispatcher::dispatch_order()
    → for each node: dispatch() → AgentCallInput  (hand to LLM provider)
    → record_output() per completed call

  run_pipeline(ir, layout, proposal, stage_outputs)
    → Lyapunov gate on Structure deltas
    → PipelineResult or PipelineError

  record_pipeline_outcome(ledger, node_id, result)
    → RewardLedger::record() → EMA update
    → trust_threshold_for() feeds next dispatcher

  every N ticks:
    ledger.update_policy(current_policy) → new PolicyParameters
    run_meta_tick(graph, ledger)         → Φ(G) if safety checks pass
```

---

## What To Build Next

### Next Session Priority Order

#### 1. Real Reward Signal (high priority)
**File:** `src/agent/pipeline.rs`
**Problem:** `PipelineResult::reward` is hardcoded to `1.0`
**Fix:** Compute reward from IR diff after mutation:
- count new capabilities unlocked (new functions, traits, modules)
- measure entropy reduction in CapabilityGraph
- penalise delta count above a threshold (too many changes = risky)
- wire into `compute_reward()` in `src/runtime/reward.rs`

#### 2. CapabilityGraph Persistence (high priority)
**Problem:** CapabilityGraph is not serialisable to disk alongside CanonicalIr
**Fix:**
- Add `#[derive(Serialize, Deserialize, JsonSchema)]` to CapabilityGraph
- Add `capability_graph: Option<CapabilityGraph>` to CanonicalIr (serde default)
- Add `load_capability_graph` / `save_capability_graph` to `src/io_utils.rs`

#### 3. CLI Commands for Agent Layer (medium priority)
**File:** `src/cli.rs` + `src/commands.rs`
**Add commands:**
- `RunPipeline`   — drives one refactor pipeline run from CLI
- `MetaTick`      — fires one meta-tick and prints graph diff
- `ShowLedger`    — prints ranked_nodes() with EMA rewards
- `ShowGraph`     — prints capability graph topology

#### 4. Tick-Driven Meta-Tick Trigger (medium priority)
**File:** `src/runtime/tick_executor/mod.rs`
**Problem:** meta-tick fires manually — needs to be wired into tick epoch boundaries
**Fix:** After every N `TickEpoch` completions, auto-fire `run_meta_tick()`
- N configurable via `PolicyParameters` (new field: `meta_tick_interval`)
- Result stored as a new IR artifact (new field: `graph_mutations: Vec<GraphMutationRecord>`)

#### 5. Observer Node Implementation (medium priority)
**Problem:** Observer capability node has no concrete IR analysis logic
**Fix:** Add `src/agent/observe.rs`:
- Reads declared IrFields from slice
- Produces structured observation: hottest modules, largest structs, deepest call chains
- Output feeds Reasoner as structured JSON (not free text)

#### 6. Prover ↔ SMT Bridge Integration (lower priority)
**File:** `src/proof/smt_bridge.rs`
**Problem:** Prover stage populates `proof_id` from LLM output payload
but does not call `attach_function_proofs()` or verify via SMT
**Fix:** After Prover stage output arrives, run `verify_function_postconditions()`
on any functions touched by the proposal before Judge fires

#### 7. Graph Partition Export (lower priority)
**Problem:** No way to visualise the capability graph alongside the DOT IR export
**Fix:** Add `export_capability_dot(graph: &CapabilityGraph) -> String`
to `src/dot_export.rs` — renders capability nodes as DOT clusters

---

## Key Invariants To Never Break

| Invariant                                       | Where enforced                        |
|-------------------------------------------------+---------------------------------------|
| Structure deltas gated by Lyapunov bound θ=0.15 | `evolution/mod.rs` apply_deltas       |
| Capability graph retains >= 3 nodes             | `agent/meta.rs` MIN_NODES             |
| Entropy deviation <= 0.5 per meta-tick          | `agent/meta.rs` MAX_ENTROPY_DELTA     |
| IR slice contains only declared IrFields        | `agent/slice.rs` build_ir_slice       |
| Trust threshold clamped [0.1, 0.95]             | `agent/reward.rs` trust_threshold_for |
| No LLM calls inside agent layer                 | all agent/* files                     |
| CapabilityGraph lives beside IR, not inside it  | architecture invariant                |

---

## How To Start Next Conversation

Paste this into the first message:

> We are building a Tier 7 self-modifying AI architecture on top of canon/canon.
> Phases 1-5 are complete (see PLAN.md).
> Today we are tackling: [pick item from "What To Build Next" above].
> Repo is at ~/git/canon/canon. Start by reading the relevant files with bat -n.

