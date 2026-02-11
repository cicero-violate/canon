Below are exactly 30 lines of instructions to bootstrap Lean from inside without breaking determinism or authority:
Do not modify the kernel; it remains admission-only and proof-hash–based.
Freeze Lean as an authoring language, not an authority.
Define a restricted proof object format (AST / bytecode) output by Lean.
Limit the format to a decidable subset (propositional + equality).
Add a Canon module proof_object to store and hash these objects.
Make proof objects pure data, no tactics, no IO, no execution.
Canon records (InvariantDraft, ProofObjectHash) instead of Lean success.
Implement a minimal proof checker inside Canon (not kernel).
Checker verifies reduction + equality only, no inference search.
Checker must be total, deterministic, and panic-free.
Checker output is (valid: bool, derived_hash).
Require derived_hash == stored ProofObjectHash.
Add a Meta-scope invariant: “proof checker rules are sound.”
Admit this invariant manually as the system’s root of trust.
Forbid learning from emitting invariants without proof objects.
Remove all runtime dependence on Lean binaries.
Lean is now used only to compile proofs → proof objects.
Store Lean source immutably for audit, not for authority.
Canon validates proof objects during learning, before delta emission.
Successful validation allows emitting CandidateInvariant deltas.
Failed validation emits no deltas and no state change.
Kernel continues to see only proof IDs and hashes.
Kernel never inspects proof objects or checker logic.
Replay uses stored hashes, never re-runs Lean.
Proof objects are append-only and content-addressed.
Upgrading the proof checker requires a new Meta invariant.
Old proof objects remain valid under old admitted laws.
Judgment predicates may reference “internally verified proof.”
Learning still cannot activate invariants without judgment.
Stop once proof validity depends only on admitted system laws.
