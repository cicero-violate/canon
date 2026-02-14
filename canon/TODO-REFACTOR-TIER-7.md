```bash
git add src/agent/meta.rs src/agent/mod.rs src/lib.rs && git commit -m "feat(agent): L7 meta-tick — CapabilityGraph self-rewrite driven by reward signal

- Add run_meta_tick(graph, ledger) -> Result<MetaTickResult, MetaTickError>
- Add GraphMutation: RemoveNode, AddEdge, RemoveEdge, PromoteToMetaAgent
- Add propose_mutations(): reads ranked_nodes() EMA to identify underperformers
- Add apply_mutation(): clones graph, applies mutation, returns candidate
- Safety check 1: entropy bound |H(G') - H(G)| <= MAX_ENTROPY_DELTA (0.5)
- Safety check 2: graph must retain >= MIN_NODES (3) after mutation
- Safety check 3: no disconnected nodes after mutation
- Underperforming nodes (EMA < -0.1) proposed for removal if not MetaAgent
- Top-ranked node promoted to MetaAgent if none exists and EMA > 0
- MetaTickResult: applied + rejected mutations with entropy before/after
- MetaTickError: typed errors for all safety violations
- Wired into agent::mod and lib.rs public surface

Phase 5 complete — Φ: G → G' operator is live.
System can now rewrite its own capability graph topology."
```

**All 5 phases are complete:**

```
Phase 1 ✓  CapabilityGraph + AgentCallDispatcher + IR slice enforcement
Phase 2 ✓  Lyapunov gate wired into apply_deltas (Structure deltas)
Phase 3 ✓  run_pipeline() Observe→Reason→Prove→Judge→Mutate
Phase 4 ✓  RewardLedger → PolicyUpdater → trust_threshold feedback
Phase 5 ✓  run_meta_tick() — CapabilityGraph rewrites itself
```

**The full Tier 7 loop now looks like this:**

```
every tick:
  AgentCallDispatcher::dispatch_order()
    → for each node: dispatch() → AgentCallInput (hand to LLM)
    → record_output() per completed call

  run_pipeline(ir, layout, proposal, stage_outputs)
    → Lyapunov gate on Structure deltas
    → PipelineResult or PipelineError

  record_pipeline_outcome(ledger, node_id, result)
    → RewardLedger::record() → EMA update
    → trust_threshold_for() feeds next dispatcher

  every N ticks:
    ledger.update_policy(current_policy)  → new PolicyParameters
    run_meta_tick(graph, ledger)          → Φ(G) if safety checks pass
```

**What remains to make this production-ready:**

The `reward: 1.0` hardcoded in `PipelineResult` needs to be replaced with a real signal derived from the IR diff after mutation — how many new capabilities were unlocked, how much entropy reduced, etc. That is a natural next step when you wire in real LLM calls.

Want to tackle that, or move to something else?
