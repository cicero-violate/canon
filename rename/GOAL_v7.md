# GOAL_PROMPT.md — Structural AST Editor with rustc Integration

## Status: MoveSymbol Working — Verification and Hardening Remaining

---

## What Is Fully Built and Verified

### Core Editor
- NodeRegistry, NodeHandle, NodeOp, FieldMutation — complete
- ProjectEditor: load, load_with_rustc, queue, queue_by_id,
  apply, validate, commit, preview — complete
- preview() — real unified diff per file (original_sources vs render_file)
- ChangeReport { touched_files, conflicts, file_moves } — complete
- commit() — executes file_moves via fs::rename after writing ASTs

### Propagation (propagate.rs)
- propagate_rename — oracle primary, occurrence visitor fallback
- propagate_delete — all sites → conflicts
- propagate_remove_field / propagate_remove_variant — conflicts
- propagate_visibility — leak analysis
- propagate_signature — satisfies_bounds conflicts
- propagate_add_field / propagate_add_variant — exhaustiveness conflicts
- apply_rewrites — SymbolEdit → registry.asts
- propagate_move — file layout change + use path updates

### Use Path Wiring
- run_use_path_rewrite — runs UsePathRewritePass on live ASTs
- collect_use_path_updates — extracts mapping from RenameIdent + MoveSymbol ops
- Fires in apply() after structural edits and rewrites

### MoveSymbol Op
- NodeOp::MoveSymbol { handle, new_module_path, new_crate }
- ModuleMovePlan — all layout transitions handled
- Same-crate: file_moves populated, use paths rewritten
- Cross-crate: EditConflict per use site

### Verified
- Smoke test: nodeop_movesymbol — file moves planned, real diff shown
- Symbol ID join key: syn == rustc confirmed
- preview() shows real unified diff

---

## Known Behavior: prettyplease Whitespace

preview() diff shows whitespace/blank-line changes from prettyplease
normalizing the source. This is expected — prettyplease canonicalizes
formatting on render. Not a bug. Expected behavior for any file that
passes through render_file.

If this is undesirable, only render files where the AST actually
changed structurally. Track mutation at apply_node_op level (see
Spurious Write Check below).

---

## Remaining Work

### Priority 1 — Spurious Write Check

touched: [] in the smoke test is correct for a file-only move.
But in rename smoke tests, 5 files were written when the target
was one symbol. Confirm touched_files only contains files where
AST actually changed.

Fix location: apply_node_op returns bool (already does in ops.rs).
apply() only inserts into touched_files when apply_node_op returns true.
Same for apply_rewrites — already returns HashSet of changed files.

The prettyplease whitespace issue above is a symptom of this —
files that were only iterated (not mutated) are being rendered
and written, showing whitespace-only diffs.

### Priority 2 — impl Block Co-Movement

Moving a struct does not automatically move its impl blocks.
Moving a trait does not move its impl blocks either.

Current behavior: impl blocks stay in original file. Code still
compiles (same crate), but organizational intent is not fulfilled.

Fix: in propagate_move, after computing new_module_path,
query oracle.impact_of(symbol_id) for associated impl blocks.
If any impl block has kind == "impl" and references the moved
symbol as its target type, queue a secondary MoveSymbol for
each impl block to follow the primary symbol.

This requires oracle adjacency to track impl-of relationships,
which the rustc frontend captures in capture_impl via traits.rs.

### Priority 3 — mod Declaration Fixup

After a MoveSymbol that changes a module's file path, the parent
module's `mod old_name;` declaration needs updating.

Current status: update_mod_declarations exists in mod_decls.rs
but operates on disk files, not registry.asts.

Options:
  A. Post-commit pass: after commit() writes files, run
     update_mod_declarations on the written files. Simpler,
     breaks atomicity slightly.
  B. Pre-commit pass: adapt update_mod_declarations to operate
     on registry.asts directly before render.

Recommended: option A for now. Mark as known limitation.

### Priority 4 — Real Propagation Smoke Test

Current smoke tests verify structure but not semantic correctness.
Need a test that:
  1. Renames a symbol with 5+ real reference sites
  2. Confirms every reference site is rewritten
  3. Runs cargo check on the modified project
  4. Confirms zero errors

This is the definitive correctness check. Until this passes,
the propagation layer is unverified for real projects.

### Priority 5 — Cross-Module Use Path Verification

run_use_path_rewrite fires but has not been verified to correctly
rewrite use statements across module boundaries. Specifically:
  - `use crate::old::path::Symbol` → `use crate::new::path::Symbol`
  - `use super::old::Symbol` → relative path needs recalculation
  - glob imports: `use crate::old::*` may need manual review

Add a smoke test that moves a symbol used via qualified path
in another module and confirms the use statement updates.

---

## Two-Mode Operation (Both Working)

Offline:
  ProjectEditor::load(project, Box::new(NullOracle))
  No compile step. propagate_rename uses occurrence visitor fallback.

Full validation:
  ProjectEditor::load_with_rustc(project)
  Oracle uses real HIR/MIR. propagate uses oracle.impact_of.

---

## What Is NOT Changing

- NodeOp / FieldMutation existing variants — unchanged
- OracleData / CargoProject — unchanged
- All existing rename CLI — unchanged
- NullOracle — unchanged
- AstEdit + text-level passes — unchanged
