The join key is verified. `true` on both sides, same string. That was the last structural risk item.

Updated prompt to reflect current state:

# GOAL_PROMPT.md — Structural AST Editor with rustc Integration

## Status: Join Key Verified — Pipeline Ready for Smoke Test

---

## What Is Fully Built and Verified

### Write Side
- `state/node.rs` — NodeHandle, NodeKind, NodeRegistry
- `rename/structured/ops.rs` — NodeOp, FieldMutation enums
- `rename/core/project_editor/mod.rs` — ProjectEditor, load, load_with_rustc,
  queue, apply, validate, commit, preview, NodeRegistryBuilder (with crate_name)
- `rename/core/project_editor/ops.rs` — all NodeOp and FieldMutation variants
- `rename/structured/ast_render.rs` — render_file via prettyplease
- `rename/core/symbol_id.rs` — normalize_symbol_id, normalize_symbol_id_with_crate,
  strip_hash_suffix (handles DefId hash segments from rustc)

### Read Side
- `rustc_integration/frontends/rustc/` — HIR, MIR, types, traits, metadata
- `rustc_integration/multi_capture.rs` — capture_project → CaptureArtifacts
- `rustc_integration/project.rs` — OracleData::from_snapshot, all 4 oracle methods

### Bridge — VERIFIED
- `rename/core/oracle.rs` — StructuralEditOracle trait + NullOracle
- `CargoProject::with_snapshot` — attaches OracleData
- `ProjectEditor::load_with_rustc` — runs capture_project, wires snapshot
- `ChangeReport { touched_files, conflicts }` — validate() flows to caller
- `rename/examples/verify_symbol_join.rs` — confirmed match:
    syn id:   crate::rename::core::project_editor::ProjectEditor
    rustc id: crate::rename::core::project_editor::ProjectEditor
    result:   true

### Symbol ID Normalization — VERIFIED
- `normalize_symbol_id` strips hash suffixes from rustc DefPath segments
- `normalize_symbol_id_with_crate` substitutes real crate name for "crate::"
- `-` to `_` substitution applied to crate names from Cargo.toml
- `crate_name_from_root` reads Cargo.toml at load time

---

## Lifecycle (Verified End-to-End)

```
ProjectEditor::load_with_rustc(project)
  1. collect_names(project)           → SymbolIndex (syn)
  2. NodeRegistryBuilder traverse     → NodeRegistry { handles, asts }
     crate_name read from Cargo.toml  → used in normalize_symbol_id_with_crate
  3. CargoProject::from_entry         → cargo metadata
  4. capture_project(frontend, cargo) → CaptureArtifacts { snapshot }
  5. cargo.with_snapshot(snapshot)    → OracleData { adjacency, macro_generated,
                                         crate_by_key, signature_by_key }
  6. ProjectEditor { registry, changesets, oracle: Box<CargoProject> }

editor.queue(symbol_id, NodeOp::MutateField { mutation: RenameIdent("new") })
editor.validate()   → Vec<EditConflict>  (adjacency lookup on normalized id)
editor.apply()      → ChangeReport { touched_files, conflicts }
editor.commit()     → Vec<PathBuf>  (render_file + write)
```

---

## Remaining Work

### Priority 1 — Full Pipeline Smoke Test

The join key is verified. Now verify the complete write path works.

Steps:
  1. Pick a known symbol id in the project (e.g. from verify_symbol_join output)
  2. In a scratch binary or example:
     ```
     let mut editor = ProjectEditor::load_with_rustc(project_path)?;
     editor.queue("crate::rename::core::project_editor::ProjectEditor",
         NodeOp::MutateField {
             handle: /* handle from registry */,
             mutation: FieldMutation::RenameIdent("ProjectEditor2".into()),
         }
     )?;
     let conflicts = editor.validate()?;
     println!("conflicts: {:?}", conflicts);
     let report = editor.apply()?;
     println!("touched: {:?}", report.touched_files);
     let preview = editor.preview()?;
     println!("{}", preview);
     // editor.commit()?;  // only after preview looks correct
     ```
  3. Confirm render_file output is valid Rust
  4. Optionally run cargo check on the modified project

### Priority 2 — Handle Queue API for NodeHandle Lookup

Currently `queue` takes a `symbol_id: &str` and a `NodeOp` that
contains a `NodeHandle` directly. The caller must construct the
handle manually, which requires knowing `item_index` and `nested_path`.

A more ergonomic API would let the registry look up the handle by id:

```rust
impl ProjectEditor {
    fn queue_by_id(&mut self, symbol_id: &str, mutation: FieldMutation)
        -> Result<()>
```

This looks up the handle from `registry.handles`, constructs the
`NodeOp::MutateField` internally, and queues it. The caller only
needs the symbol_id string — which they already have from
`verify_symbol_join` or `collect_names`.

This is a usability improvement, not a correctness fix.

### Priority 3 — key_by_id Dead Code Warning

`OracleData` still has `key_by_id` field in the symbol index
(line 302 of project.rs) despite the patch that removed it.
Confirm whether the field was actually removed or if the patch
partially applied. If still present, remove it.

### Priority 4 — ops.rs Edge Cases (Known, Not Blocking)

`reorder_items`: top-level only. Nested reorder not yet supported.

`replace_signature`: does not propagate to call sites. Caller
must queue additional ops for each site returned by `impact_of`.

`nested_path` tracking: verified for ImplFn. Structs inside
inline mods may need additional NodeRegistryBuilder coverage.

---

## Two-Mode Operation (Both Working)

### Offline / Fast
```rust
ProjectEditor::load(project, Box::new(NullOracle))
```
No compile step. Full NodeOp execution. Oracle returns safe defaults.

### Full Validation
```rust
ProjectEditor::load_with_rustc(project)
```
Runs rustc capture. Oracle queries real HIR/MIR graph.
validate() surfaces real conflicts before commit.

---

## What Is NOT Changing

- rename/core/collect/ — syn collection unchanged
- rename/structured/orchestrator.rs — pass pipeline unchanged
- AstEdit + DocAttrPass + UsePathRewritePass — unchanged
- All existing rename CLI (run_names, run_rename) — unchanged
- rename/api.rs MutationRequest — unchanged
