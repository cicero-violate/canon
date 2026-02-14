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
