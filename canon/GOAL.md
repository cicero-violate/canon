| **Tier** | **ID** | **Condition**               | **Formal Constraint (Latent)**   | **Operational Meaning**                    | **Implemented?** |
| -------- | ------ | --------------------------- | -------------------------------- | ------------------------------------------ | --------------- |
| **7**    | **A**  | Append-Only Audit           | ∀δ∈Δ: prov(δ) ∧ 𝓜←𝓜∥⟨δ,prov⟩     | All proposals & decisions immutably logged | ✅              |
| **7**    | **B**  | Two-Phase Commit            | apply(s,δ) ⇔ admit(δ)=accept     | No inline mutation                         | ✅              |
| **7**    | **C**  | Invariant Preservation      | s⊨Π ∧ accept ⇒ apply(s,δ)⊨Π      | Safety monotonicity                        | ❌              |
| **7**    | **D**  | Risk Budget (Intent Radius) | accept ⇒ risk(δ) ≤ Θ             | Bounded blast radius                       | ✅              |
| **7**    | **E**  | Spec-Bounded Rewrite        | δ∈Δ_self ⇒ δ∈Spec                | Restricted self-edit DSL                   | ❌              |
| **7**    | **F**  | Proof / Check Carrying      | accept ⇒ Proof ∨ Verify          | Formal safety evidence                     | ❌              |
| **7**    | **G**  | Rewind / Rollback           | ∀δ accepted, ∃undo ∨ checkpoint  | Deterministic recovery                     | ✅              |
| **7**    | **H**  | Gate Non-Bypass             | Δ_self ∩ Δ_TCB = ∅               | Judge cannot be rewritten                  | ✅              |
| **7**    | **I**  | Fail-Closed Admission       | uncertainty ⇒ reject             | No permission from ambiguity               | ✅              |
| ---      | ---    | ---                         | ---                              | ---                                        | ---             |
| **8**    | **J**  | Global Invariant Closure    | ∀s reachable, s ⊨ Π_global       | No hidden invariant violations             | ❌              |
| **8**    | **K**  | Invariant Self-Reference    | Π ⊢ sound(Π)                     | Invariants reason about themselves         | ❌              |
| **8**    | **L**  | Judgment Fixpoint           | Decide(Π,State) is idempotent    | Same judgment ⇒ same result                | ❌              |
| **8**    | **M**  | Goal Consistency            | ∀g∈G, g ⊨ Π                      | Goals cannot contradict invariants         | ❌              |
| **8**    | **N**  | Self-Rewrite Stability      | rewrite(rewrite(S)) ≈ rewrite(S) | No oscillation / drift                     | ❌              |
| **8**    | **O**  | Unified State Identity      | Thought ≡ State ≡ Action         | No representational gap                    | ❌              |
