GOAL

Construct a recursively self-optimizing system that:
Maintains a persistent world model
Simulates counterfactual futures
Optimizes an explicit scalar utility
Learns from execution outcomes
Updates its own policy
Mutates its own goals under constraint
Preserves formal invariants during self-modification

Foundation Layer

Define a formal scalar utility function 
Add a reward logging mechanism to execution records.
Ensure reward is computed after every tick.
Add reward delta tracking to CanonicalIr.
Define utility monotonicity checks.

World Model Layer

Introduce a predictive state model 
Implement multi-step rollout simulation.
Store prediction error metrics.
Add world-model update step after each execution.
Track entropy reduction over model error.

Planning Layer

Modify planner to optimize expected reward, not just correctness.
Add search depth parameter.
Implement candidate action scoring via simulated rollouts.
Record planner decision rationale with utility estimate.
Surface plan/execution wiring in the timeline IR and verify it with a planner fixture that drives ticks/rewards.

Learning Layer

Introduce policy parameters 
separate from static rules.
Add parameter update rule 
Store policy snapshots across epochs.
Add regression detection for reward collapse.

Goal Layer

Implement goal mutation function constrained by invariants.
Add goal evaluation metric to prevent misalignment drift.
