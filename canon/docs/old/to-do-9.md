1. [x] Parse CanonicalIr and locate Proposal by ProposalId.
2. [x] Assert Proposal.status == Submitted.
3. [x] Validate proposal declarative completeness (nodes, apis, edges).
4. [x] Deterministically map ProposedNode::Module → AddModule delta.
5. [x] Deterministically map ProposedNode::Struct → AddStruct delta.
6. [x] Deterministically map ProposedNode::Trait → AddTrait delta.
7. [x] For each ProposedApi, emit AddTraitFunction deltas.
8. [x] Emit AddImpl deltas binding structs to traits.
9. [x] Emit AddFunction deltas for each trait function implementation.
10. [x] Emit AddModuleEdge deltas for proposal edges.
11. [x] Assign all emitted deltas kind = Structure.
12. [x] Assign stage = Decide for all emitted deltas.
13. [x] Set append_only = true for all deltas.
14. [x] Attach proof_id referencing proposal proof artifact.
15. [x] Insert emitted deltas into CanonicalIr.deltas.
16. [x] Create Judgment referencing proposal with decision = Accept.
17. [x] Insert Judgment into CanonicalIr.judgments.
18. [x] Create DeltaAdmission linking judgment to emitted deltas.
19. [x] Assign admission to current TickId.
20. [x] Insert DeltaAdmission into CanonicalIr.admissions.
21. [x] Call apply_deltas(ir, [admission_id]).
22. [x] Kernel verifies state hash, proofs, invariants.
23. [x] AppliedDeltaRecords are appended in order.
24. [x] CanonicalIr state advances deterministically.
25. [x] Validate updated CanonicalIr via validate_ir.
26. [x] Call materialize(updated_ir).
27. [x] Generate FileTree from structural artifacts.
28. [x] Emit src/ directories and Rust modules.
29. [x] Emit structs, traits, impls, functions as todo stubs.
30. [x] Write FileTree to disk as canonical source output.
