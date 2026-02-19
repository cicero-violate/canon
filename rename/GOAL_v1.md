# GOAL_PROMPT.md — Structural AST Editor with rustc Integration

## Current State: Structurally Complete — One Wire Missing

---

## What Is Built and Working

### Write Side (syn)
- `state/node.rs` — NodeHandle, NodeKind, NodeRegistry
- `rename/structured/ops.rs` — NodeOp, FieldMutation enums
- `rename/core/project_editor/mod.rs` — ProjectEditor, NodeRegistryBuilder
- `rename/core/project_editor/ops.rs` — all NodeOp variants implemented (546 LOC)
- `rename/structured/ast_render.rs` — render_file via prettyplease
- `rename/core/symbol_id.rs` — normalize_symbol_id (join key)
- `rename/api.rs` — UpsertRequest with node_ops field

### Read Side (rustc)
- `rustc_integration/frontends/rustc/` — HIR, MIR, types, traits, metadata capture
- `rustc_integration/multi_capture.rs` — capture_project → CaptureArtifacts { snapshot }
- `rustc_integration/transform/normalizer.rs` — GraphNormalizer → GraphSnapshot
- `rustc_integration/project.rs` — OracleData::from_snapshot, all 4 oracle methods real
- `rename/core/oracle.rs` — StructuralEditOracle trait

### Bridge
- `CargoProject::with_snapshot(GraphSnapshot)` — exists, attaches OracleData
- `impl StructuralEditOracle for CargoProject` — queries OracleData

---

## The One Missing Wire

File: `rename/core/project_editor/mod.rs`
Function: `ProjectEditor::load`

Currently loads syn side only. Needs to also run rustc capture and
attach the snapshot to CargoProject before storing the oracle.

The call chain that needs to be added:
```
ProjectEditor::load(project, oracle):
  1. collect_names(project)          // syn side — already done
  2. NodeRegistryBuilder traversal   // already done
  3. CargoProject::from_entry(project)        // NEW
  4. capture_project(&frontend, &cargo, &[])  // NEW — rustc_integration/multi_capture.rs
  5. cargo.with_snapshot(artifacts.snapshot)  // NEW
  6. store cargo as oracle                     // NEW (replaces passed-in oracle)
```

OR: keep oracle as Box<dyn StructuralEditOracle> and require caller to
pass a pre-loaded CargoProject. This keeps ProjectEditor::load fast
for callers that don't need rustc validation.

Recommended: make rustc capture opt-in via a separate method:
```
impl ProjectEditor {
    fn load(project: &Path, oracle: Box<dyn StructuralEditOracle>) -> Result<Self>
    fn load_with_rustc(project: &Path) -> Result<Self>  // NEW: runs capture internally
}
```

`load_with_rustc` is the full pipeline. `load` stays fast for callers
that provide their own oracle (including NullOracle for offline use).

---

## NullOracle (add this)

For callers that want structural editing without rustc validation:

File: `rename/core/oracle.rs`
```rust
struct NullOracle;
impl StructuralEditOracle for NullOracle {
    fn impact_of(&self, _: &str) -> Vec<String> { vec![] }
    fn satisfies_bounds(&self, _: &str, _: &syn::Signature) -> bool { true }
    fn is_macro_generated(&self, _: &str) -> bool { false }
    fn cross_crate_users(&self, _: &str) -> Vec<String> { vec![] }
}
```

This makes ProjectEditor usable without a rustc toolchain present
(stub.rs path in rustc_integration).

---

## Symbol ID Join Key — Still Needs Verification

`rename/core/symbol_id.rs` → normalize_symbol_id
`rustc_integration/frontends/rustc/metadata.rs` → module_path() + format_display()

These must produce identical strings for the same symbol or OracleData
lookups silently return empty results. This has not been verified yet.

To verify: pick one struct in the project, print its id from both
sides after normalization, confirm they match.

The adjacency map in OracleData is keyed by normalized id. If the
syn side produces "rename::core::types::SymbolIndex" and the rustc
side produces "crate::core::types::SymbolIndex", all oracle queries
return empty — silently, with no error.

This is the highest risk item remaining.

---

## ops.rs Size

ops.rs is 546 LOC. If splitting is desired, natural boundaries are:

- `ops/ident.rs` — rename_ident, rename_ident_in_item
- `ops/structure.rs` — add_struct_field, remove_struct_field, add_variant, remove_variant
- `ops/node.rs` — replace_node, insert_node, delete_node, reorder_items
- `ops/attr.rs` — add_attribute, remove_attribute, change_visibility, replace_signature
- `ops/resolve.rs` — resolve_target_mut, TargetItemMut

Not required — 546 LOC is fine for a single file at this stage.

---

## Remaining Work (in order)

1. Verify symbol ID join key (highest risk, do first)
2. Add NullOracle to oracle.rs
3. Add ProjectEditor::load_with_rustc wiring capture_project
4. Integration smoke test: load project, queue RenameIdent on one symbol,
   validate (oracle fires), commit, verify file changed correctly
5. Wire validate() conflicts back into ChangeReport

---

## What Is NOT Changing

- rename/core/collect/ — syn collection unchanged
- rename/structured/orchestrator.rs — pass pipeline unchanged
- AstEdit + text-level passes — unchanged
- All existing rename functionality — unchanged
- NodeHandle, NodeOp, ProjectEditor structure — unchanged
