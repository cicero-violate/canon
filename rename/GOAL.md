Looking at the symbol index, significant progress has already been made. Let me map what exists against the goal.

---

## What Is Already Built

**`state/node.rs`** — `NodeHandle`, `NodeKind`, `NodeRegistry` all exist. Layer 1 is done structurally.

**`rename/structured/ops.rs`** — `NodeOp`, `FieldMutation`, `node_handle()` exist. Layer 2 is done.

**`rename/core/oracle.rs`** — `StructuralEditOracle` trait exists (stub).

**`rename/core/project_editor.rs`** — `ProjectEditor`, `NodeRegistryBuilder`, `apply_node_op`, `EditConflict`, `ChangeReport` all exist. Layer 3 is substantially built.

**`rename/core/symbol_id.rs`** — `normalize_symbol_id` exists. The join key normalization is done.

**`rename/structured/ast_render.rs`** — `render_file` exists alongside the node renderers.

**`rustc_integration/project.rs`** — `CargoProject` already `impl StructuralEditOracle`. The bridge is stubbed.

**`rename/api.rs`** — `UpsertRequest` already has `node_ops` field and `node_op()` builder. API surface is extended.

---

## What Is Actually Missing

The oracle implementation in `rustc_integration/project.rs` has all four methods returning empty/false stubs:

```
fn impact_of        -> Vec<String>   // returns []
fn satisfies_bounds -> bool          // returns false
fn is_macro_generated -> bool        // returns false  
fn cross_crate_users -> Vec<String>  // returns []
```

This is the only real gap. Everything else is scaffolded. The `RustcFrontend` in `frontends/rustc/collector.rs` captures HIR, MIR, traits, types — but that captured data is not yet flowing into the oracle answers.

---

## Updated GOAL_PROMPT.md

```markdown
# GOAL_PROMPT.md — Structural AST Editor with rustc Integration

## Status: Scaffolding Complete — Oracle Implementation Remaining

---

## What Is Built

- `state/node.rs` — NodeHandle, NodeKind, NodeRegistry
- `rename/structured/ops.rs` — NodeOp, FieldMutation
- `rename/core/oracle.rs` — StructuralEditOracle trait
- `rename/core/project_editor.rs` — ProjectEditor, NodeRegistryBuilder, apply_node_op
- `rename/core/symbol_id.rs` — normalize_symbol_id (join key)
- `rename/structured/ast_render.rs` — render_file via prettyplease
- `rename/api.rs` — UpsertRequest with node_ops field
- `rustc_integration/project.rs` — CargoProject impl StructuralEditOracle (STUB)

---

## Architecture

```
S_full = S_syn ⋈ S_rustc   (joined on normalize_symbol_id)

S_syn   → NodeRegistry (write authority, live syn::File ASTs)
S_rustc → CargoProject (read oracle, type truth from HIR/MIR)
```

State = NodeRegistry { handles, asts, changesets }

Lifecycle:
  load() → queue() → validate() → commit()
  parse all    accumulate    oracle check   render+write

---

## The Remaining Work: Oracle Implementation

File: `rustc_integration/project.rs`
Struct: `CargoProject impl StructuralEditOracle`

All four methods are stubs. They need to be wired to the
rustc frontend capture pipeline.

### 1. `impact_of(symbol_id) -> Vec<String>`

Answer: which other symbols reference this one?

Source: `frontends/rustc/mir.rs` — `collect_calls` builds the
call graph. Walk edges from the GraphSnapshot where
`to == symbol_id`, return all `from` node keys.

Also check `frontends/rustc/types.rs` — `collect_type_dependencies`
for type-level references (struct fields, function signatures).

### 2. `satisfies_bounds(id, new_sig) -> bool`

Answer: does a proposed new function signature satisfy all
trait bounds at call sites?

Source: `frontends/rustc/traits.rs` — `capture_trait` and
`capture_impl` have trait obligation data serialized via
`serialize_associated_items`. Compare the new_sig against
those predicates.

This is the hardest method. For now, a conservative
implementation that returns false when any trait impl
references this symbol is acceptable — it surfaces
conflicts without false negatives.

### 3. `is_macro_generated(symbol_id) -> bool`

Answer: does this symbol originate from macro expansion?

Source: `frontends/rustc/metadata.rs` — `capture_attributes`
captures derive and proc_macro attributes. If the symbol's
NodePayload has a `#[derive(...)]` or proc_macro attribute
that would generate it, return true.

Also: `frontends/rustc/items.rs` — check if the DefId's
span origin is a macro expansion site.

### 4. `cross_crate_users(symbol_id) -> Vec<String>`

Answer: which symbols in *other* crates use this one?

Source: `multi_capture.rs` — `capture_project` runs the
frontend across all crates. The resulting GraphSnapshot
contains cross-crate edges. Filter edges where:
  - `to` normalizes to symbol_id
  - `from` originates in a different crate prefix

---

## Wire-Up: CaptureArtifacts → Oracle

`multi_capture.rs` produces `CaptureArtifacts { snapshot, workspace, graph_deltas }`.

`CargoProject` needs to hold a `GraphSnapshot` populated at
`ProjectEditor::load` time:

```
struct CargoProject {
    root, target_dir,
    snapshot: Option<GraphSnapshot>,  // ADD THIS
}
```

`ProjectEditor::load` calls:
  1. `collect_names` (syn side) → NodeRegistry
  2. `capture_project` (rustc side) → GraphSnapshot
  3. Stores snapshot in CargoProject
  4. Oracle methods query the snapshot

The snapshot is the materialized read state.
The NodeRegistry is the materialized write state.
They are joined by normalize_symbol_id.

---

## Symbol ID Join Key — Verify This First

`rename/core/symbol_id.rs` → `normalize_symbol_id`
`rustc_integration/frontends/rustc/metadata.rs` → `module_path` + `format_display`

Before implementing oracle methods, verify both sides
produce identical strings for the same symbol.

Test case: pick one struct in the project, print its
id from both sides, confirm they match after normalization.

If they do not match, fix normalize_symbol_id first.
Everything else depends on this.

---

## Files To Modify (in order)

1. `rustc_integration/project.rs`
   - Add `snapshot: Option<GraphSnapshot>` field to CargoProject
   - Add `with_snapshot(mut self, s: GraphSnapshot) -> Self`
   - Implement all four oracle methods against snapshot

2. `rename/core/project_editor.rs`
   - In `ProjectEditor::load`: call `capture_project` after `collect_names`
   - Pass resulting snapshot into CargoProject

3. `rustc_integration/frontends/rustc/metadata.rs`
   - Verify `module_path()` output matches `normalize_symbol_id` format
   - Add normalization call if needed

---

## What Is NOT Changing

- rename/core/collect/ — syn collection unchanged
- rename/structured/orchestrator.rs — pass pipeline unchanged
- AstEdit + text-level passes — unchanged
- All existing rename functionality — unchanged
- NodeHandle, NodeOp, ProjectEditor structure — unchanged

---

## Build Order

1. Verify symbol ID join key (print both sides for one symbol)
2. Add snapshot field to CargoProject
3. Wire capture_project into ProjectEditor::load
4. Implement impact_of via GraphSnapshot edge query
5. Implement is_macro_generated via attribute metadata
6. Implement cross_crate_users via multi-crate snapshot edges
7. Implement satisfies_bounds (conservative: false on any trait ref)
8. Integration test: queue a NodeOp, validate, confirm oracle fires
