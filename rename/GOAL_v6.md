```markdown
# GOAL_PROMPT.md — Structural AST Editor with rustc Integration

## Status: Propagation Layer Built — MoveSymbol + Use Path Wiring Next

---

## What Is Fully Built and Verified

### Core Editor
- NodeRegistry, NodeHandle, NodeKind — complete
- NodeOp, FieldMutation — complete
- ProjectEditor: load, load_with_rustc, queue, queue_by_id,
  apply, validate, commit, preview — complete
- apply() wires: structural edit → propagate → apply rewrites → merge conflicts

### Propagation Layer (propagate.rs)
- PropagationResult { rewrites, conflicts }
- propagate() dispatch on all op variants
- propagate_rename — oracle.impact_of primary, occurrence visitor fallback
- propagate_delete — all sites become conflicts
- propagate_remove_field — field access site conflicts
- propagate_remove_variant — match arm conflicts
- propagate_visibility — leak analysis via AliasGraph
- propagate_signature — satisfies_bounds per call site
- propagate_add_field — constructor completeness conflicts
- propagate_add_variant — exhaustiveness conflicts
- apply_rewrites — applies SymbolEdit list to registry.asts
- build_symbol_index_and_occurrences — syn fallback path
- build_visibility_map — VisibilityScope per symbol

### Oracle
- OracleData { adjacency, macro_generated, crate_by_key, signature_by_key }
- impact_of, cross_crate_users, satisfies_bounds, is_macro_generated — real
- NullOracle — offline path
- Symbol ID join key — verified (syn == rustc)

---

## What Is NOT Yet Built

### 1. MoveSymbol Op

Add to NodeOp enum in rename/structured/ops.rs:

```rust
MoveSymbol {
    handle: NodeHandle,
    new_module_path: String,   // e.g. "crate::new::location"
    new_crate: Option<String>, // None = same crate, Some = cross-crate
}
```

Add to propagate.rs:

```rust
fn propagate_move(
    symbol_id: &str,
    new_module_path: &str,
    new_crate: Option<&str>,
    registry: &NodeRegistry,
    oracle: &dyn StructuralEditOracle,
) -> Result<PropagationResult>
```

Same-crate move logic:
  1. ModuleMovePlan::new(old_path, new_path, file, project_root)
     → determines file layout change (Inline/File/Directory)
  2. oracle.impact_of(symbol_id)
     → internal reference sites
  3. build path_updates map: { old_module_path → new_module_path }
  4. UsePathRewritePass or update_use_paths against registry.asts
     → rewrites all use statements across all files
  5. update_mod_declarations against registry.asts
     → fixes parent module mod decls
  6. oracle.cross_crate_users(symbol_id)
     → EditConflict per external user (cannot auto-resolve)
  7. file_renames: Vec<FileRename> added to ChangeReport

Cross-crate move:
  - EditConflict on every use site
  - EditConflict noting Cargo.toml change needed
  - Definition site still moves (apply_node_op executes)

ChangeReport needs new field:
  file_moves: Vec<(PathBuf, PathBuf)>  // (from, to)
commit() executes file_moves via std::fs::rename after writing ASTs.

---

### 2. Wire update_use_paths Into apply()

Currently apply() rewrites identifier occurrences via SymbolEdit
but does NOT rewrite use statement paths. This means after a rename:
  use crate::old_module::OldName  →  not updated

The fix: after propagate_rename produces rewrites, also run
UsePathRewritePass on every file in registry.asts with the
mapping { old_id → new_name }.

Location: rename/core/project_editor/mod.rs — inside apply()
after apply_rewrites call.

```rust
// after apply_rewrites(registry, rewrites)
if let Some(mapping) = extract_rename_mapping(&changesets) {
    run_use_path_rewrite(&mut self.registry, &mapping)?;
}
```

run_use_path_rewrite iterates registry.asts, runs UsePathRewritePass
on each file's AST directly (not on disk — on the live syn::File).

UsePathRewritePass currently takes content: &str which it uses
for span anchoring. For in-memory AST rewriting it needs to work
without content. Either pass empty string or extend the pass to
operate purely on the AST tree without span anchoring.

---

### 3. Wire update_mod_declarations Into apply()

After any rename or move that changes a module's name or file path,
the parent module's mod declaration must be updated.

Currently update_mod_declarations reads/writes files on disk.
It needs to be adapted to operate on registry.asts instead.

Or: after commit() writes ASTs to disk, run update_mod_declarations
as a post-commit fixup pass. This is simpler but breaks atomicity.

Recommended: post-commit pass for now. Mark as known limitation.

---

### 4. preview() Returns Real Diff

Currently returns "N files touched".

Should return unified diff per file:
  before: original source (read from disk at load time)
  after: render_file(registry.asts[file])

Implementation:
  - At load time, store original_sources: HashMap<PathBuf, String>
    alongside registry.asts
  - preview() iterates touched files, diffs original vs rendered
  - Format as unified diff (use similar crate or manual diff)

Add to NodeRegistry or ProjectEditor:
  original_sources: HashMap<PathBuf, String>

---

### 5. Spurious Write Check

commit() currently writes all files in touched_files.
touched_files may include files that were iterated but not mutated.

Fix: track actual mutation at apply_node_op and apply_rewrites level.
Only add file to touched_files if the AST actually changed.

In apply_node_op: return bool indicating whether mutation occurred.
In apply_rewrites: apply_symbol_edits_to_ast already returns bool.
In apply(): only insert into touched_files if either returned true.

---

## Build Order

1. Wire update_use_paths into apply() — completes rename propagation
2. Spurious write check — correctness fix, low risk
3. MoveSymbol op + propagate_move — new capability
4. Wire update_mod_declarations — completes module move
5. preview() real diff — usability

---

## Files To Modify

update_use_paths wiring:
  rename/core/project_editor/mod.rs — add run_use_path_rewrite call in apply()
  rename/structured/use_tree.rs — extend UsePathRewritePass to work on live AST

MoveSymbol:
  rename/structured/ops.rs — add MoveSymbol variant
  rename/core/project_editor/propagate.rs — add propagate_move
  rename/core/project_editor/mod.rs — handle MoveSymbol in apply()
  rename/core/types.rs — ChangeReport gains file_moves field
  rename/core/project_editor/mod.rs — commit() executes file_moves

preview() diff:
  rename/core/project_editor/mod.rs — add original_sources to load
  rename/core/project_editor/mod.rs — implement real diff in preview()

---

## Symbol ID Contract (Unchanged)

All IDs crossing oracle/registry boundary pass through:
  normalize_symbol_id_with_crate(raw, Some(crate_name))

This invariant must hold in all new code.

---

## What Is NOT Changing

- NodeOp / FieldMutation existing variants — unchanged
- OracleData / CargoProject — unchanged
- apply_symbol_edits_to_ast — unchanged
- All existing rename CLI — unchanged
- NullOracle — unchanged
- propagate.rs existing functions — unchanged
```
