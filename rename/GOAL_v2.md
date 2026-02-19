## What This Commit Completed
**`NullOracle`** is in `oracle.rs` — `load` path is now usable without a rustc toolchain.
**`ProjectEditor::load_with_rustc`** exists in `project_editor/mod.rs` — the wire from `capture_project` → `with_snapshot` → oracle is closed.
**`ChangeReport`** now carries `conflicts` — `apply()` populates them, validate results flow back to the caller.
**`verify_symbol_join` example** — tooling to confirm the join key matches between both sides.

The architecture is complete. What remains is operational verification and the one known risk item.

---

## Updated GOAL_PROMPT.md

```markdown
# GOAL_PROMPT.md — Structural AST Editor with rustc Integration

## Status: Architecture Complete — Verification and Hardening Remaining

---

## What Is Fully Built

### Write Side
- `state/node.rs` — NodeHandle, NodeKind, NodeRegistry
- `rename/structured/ops.rs` — NodeOp, FieldMutation enums
- `rename/core/project_editor/mod.rs` — ProjectEditor, load, load_with_rustc, queue,
  apply, validate, commit, preview, NodeRegistryBuilder
- `rename/core/project_editor/ops.rs` — all NodeOp and FieldMutation variants (546 LOC)
- `rename/structured/ast_render.rs` — render_file via prettyplease
- `rename/core/symbol_id.rs` — normalize_symbol_id, normalize_symbol_id_with_crate
- `rename/api.rs` — UpsertRequest with node_ops, MutationRequest unchanged

### Read Side
- `rustc_integration/frontends/rustc/` — HIR, MIR, types, traits, metadata
- `rustc_integration/multi_capture.rs` — capture_project → CaptureArtifacts
- `rustc_integration/transform/normalizer.rs` — GraphNormalizer → GraphSnapshot
- `rustc_integration/project.rs` — OracleData::from_snapshot, all 4 oracle methods real

### Bridge
- `rename/core/oracle.rs` — StructuralEditOracle trait + NullOracle
- `CargoProject::with_snapshot` — attaches OracleData
- `ProjectEditor::load_with_rustc` — runs capture_project, wires snapshot into oracle
- `ChangeReport { touched_files, conflicts }` — validate() results flow to caller

### Verification Tooling
- `rename/examples/verify_symbol_join.rs` — compares syn vs rustc symbol ids

---

## Lifecycle (Complete)

```
ProjectEditor::load_with_rustc(project)
  1. collect_names(project)           → SymbolIndex (syn)
  2. NodeRegistryBuilder traverse     → NodeRegistry { handles, asts }
  3. CargoProject::from_entry         → cargo metadata
  4. capture_project(frontend, cargo) → CaptureArtifacts { snapshot }
  5. cargo.with_snapshot(snapshot)    → OracleData built from GraphSnapshot
  6. ProjectEditor { registry, changesets, oracle: Box<CargoProject> }

editor.queue(symbol_id, NodeOp::MutateField { mutation: RenameIdent("new") })
editor.validate()   → Vec<EditConflict>  (oracle fires here)
editor.apply()      → ChangeReport { touched_files, conflicts }
editor.commit()     → Vec<PathBuf>  (render_file + write)
```

---

## Remaining Work

### Priority 1 — Verify Symbol ID Join Key

This is the highest-risk item. Silent failure mode: oracle returns
empty results for everything because adjacency map lookups miss.

Run:
  cargo run --example verify_symbol_join

Expected output: syn and rustc produce identical normalized strings
for at least one struct/fn in the project.

If they do NOT match, fix normalize_symbol_id or the rustc
metadata::module_path() output until they do.

Key functions to compare:
  syn side:   rename/core/symbol_id.rs → normalize_symbol_id
  rustc side: rustc_integration/frontends/rustc/metadata.rs → module_path + format_display

Common mismatch patterns to check:
  - "crate" vs actual crate name prefix
  - hash suffixes on rustc DefPath segments (strip_hash_suffix handles this)
  - trailing "::" differences
  - impl block naming conventions

### Priority 2 — Smoke Test the Full Pipeline

Manual steps (no automated test):
  1. cargo run --example verify_symbol_join  (join key check)
  2. In a scratch binary:
     - ProjectEditor::load_with_rustc(project_path)
     - queue one RenameIdent on a known symbol id
     - validate() — confirm no false conflicts
     - apply() — confirm ChangeReport has the right file
     - preview() — read the diff before committing
     - commit() — confirm file changed on disk
     - cargo check on the modified project — confirm it still compiles

### Priority 3 — ops.rs Edge Cases

The 546 LOC in ops.rs covers all variants but some have known gaps:

`reorder_items`: only reorders top-level items. Items inside impl
blocks or inline mods are not reordered. If needed, extend
resolve_target_mut to handle nested contexts.

`replace_signature`: replaces the syn::Signature but does not
propagate the change to call sites. This is expected — the oracle
via impact_of gives the caller the list of affected sites.
The caller must queue additional ops for call sites manually.

`nested_path` in NodeHandle: NodeRegistryBuilder currently sets
nested_path for ImplFn contexts. Deeper nesting (mod inside mod,
impl inside mod) may not be fully tracked. Verify with a struct
defined inside an inline mod.

### Priority 4 — Transform Stubs

These exist but are empty:
  rustc_integration/transform/resolver.rs — Resolver::resolve
  rustc_integration/transform/linker.rs  — Linker::link

These are not needed for the current oracle implementation since
OracleData::from_snapshot works directly from GraphSnapshot.
Leave them until a use case requires them.

---

## Two-Mode Operation

### Without rustc toolchain (fast, offline)
  ProjectEditor::load(project, Box::new(NullOracle))
  - No compile step
  - Oracle returns safe defaults (impact_of empty, satisfies_bounds true)
  - Full NodeOp execution still works
  - Use for simple structural edits where impact analysis is not needed

### With rustc toolchain (full validation)
  ProjectEditor::load_with_rustc(project)
  - Runs rustc capture (compile-time cost)
  - Oracle answers from real HIR/MIR data
  - validate() surfaces real conflicts before commit
  - Use for refactors that touch public API or cross-crate symbols

---

## What Is NOT Changing

- rename/core/collect/ — syn collection unchanged
- rename/structured/orchestrator.rs — pass pipeline unchanged
- AstEdit + text-level passes (DocAttrPass, UsePathRewritePass) — unchanged
- All existing rename CLI functionality — unchanged
- rename/api.rs MutationRequest — unchanged
