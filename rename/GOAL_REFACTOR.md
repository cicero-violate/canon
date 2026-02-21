1. Parse the source workspace and build a complete GraphSnapshot (Model₀) as the single authoritative state.
2. Validate Model₀ for structural completeness and invariant correctness before applying any changes.
3. Apply all refactor operations as pure graph mutations to produce Model₁ (no text edits allowed).
4. Recompute all derived edges and semantic relationships in Model₁ (calls, imports, visibility, ordering).
5. Project Model₁ into a deterministic Plan₁ describing full file layout and contents.
6. Emit Plan₁ to regenerate the entire source workspace as Source₁.
7. Re-parse Source₁ and rebuild a new GraphSnapshot (Model₂).
8. Compare Model₂ to Model₁ for structural equality (fixpoint verification).
9. If Model₂ ≠ Model₁, correct projection rules and repeat from step 5.
10. If Model₂ == Model₁, commit Source₁ as the stable refactored state.
