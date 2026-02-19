Right. Rename is just one mutation. The full picture is:

$$\Delta_\text{structural} \xrightarrow{\text{propagate}} \Delta_\text{references}$$

for every op that changes the public contract of a node.

---

## Which Ops Need Propagation

Not all ops need it. The ones that change what other code sees:

| Op                  | What propagates                                                                    |
|---------------------+------------------------------------------------------------------------------------|
| `RenameIdent`       | All reference sites rewrite the identifier                                         |
| `ChangeVisibility`  | Visibility leak analysis — callers in restricted modules get conflict warnings     |
| `RemoveStructField` | All field access sites (`expr.field`, patterns) need conflict or rewrite           |
| `RemoveVariant`     | All match arms on that variant need conflict or rewrite                            |
| `ReplaceSignature`  | All call sites need signature compatibility check, param rewrites if names changed |
| `DeleteNode`        | All import sites (`use`), all call sites → conflicts                               |
| `AddStructField`    | If non-optional (no default), all construction sites need update                   |
| `AddVariant`        | Non-exhaustive match arms → conflicts                                              |
| `ReplaceNode`       | Depends on what changed — treat as DeleteNode + new declaration                    |

Ops that do **not** propagate: `AddAttribute`, `RemoveAttribute`, `InsertBefore`, `InsertAfter`, `ReorderItems`. These don't affect external contract.

---

## The Propagation Model

Each op produces two classes of output:

$$\text{PropagationResult} = (\text{rewrites}: [\text{SymbolEdit}],\ \text{conflicts}: [\text{EditConflict}])$$

**Rewrites** — edits that can be applied automatically. Identifier renames at call sites, import path updates.

**Conflicts** — edits that require human decision. Removed field still accessed, removed variant still matched, signature incompatible at call site, visibility reduced but symbol is pub-used elsewhere.

The distinction matters: rewrites go into `ChangeReport.touched_files`, conflicts go into `ChangeReport.conflicts`. The caller decides whether to abort on conflicts or proceed anyway.

---

## The Architecture

One new file: `rename/core/project_editor/propagate.rs`

```rust
pub struct PropagationResult {
    pub rewrites: Vec<SymbolEdit>,
    pub conflicts: Vec<EditConflict>,
}

pub fn propagate(
    op: &NodeOp,
    symbol_id: &str,
    registry: &NodeRegistry,
    oracle: &dyn StructuralEditOracle,
) -> Result<PropagationResult>
```

Inside, a match on op variant dispatches to:

```rust
fn propagate_rename(symbol_id, new_name, registry) -> PropagationResult
fn propagate_delete(symbol_id, oracle) -> PropagationResult
fn propagate_visibility(symbol_id, new_vis, oracle) -> PropagationResult
fn propagate_signature(symbol_id, new_sig, oracle) -> PropagationResult
fn propagate_remove_field(symbol_id, field_name, registry, oracle) -> PropagationResult
fn propagate_remove_variant(symbol_id, variant_name, registry, oracle) -> PropagationResult
fn propagate_add_field(symbol_id, field, oracle) -> PropagationResult
fn propagate_add_variant(symbol_id, variant, oracle) -> PropagationResult
```

`propagate_rename` uses `EnhancedOccurrenceVisitor` across all files in `registry.asts`.

`propagate_delete` uses `oracle.impact_of` and `oracle.cross_crate_users` — all sites become conflicts, none become rewrites (deletion can't be auto-applied at call sites).

`propagate_visibility` uses `oracle.cross_crate_users` + `AliasGraph.analyze_visibility_leaks` — restricted visibility that leaks becomes a conflict.

`propagate_signature` uses `oracle.satisfies_bounds` per call site — incompatible becomes conflict, compatible becomes rewrite if param names changed.

`propagate_remove_field` / `propagate_remove_variant` walk `registry.asts` looking for field access expressions and match arms — found usages become conflicts.

`propagate_add_field` (non-optional) walks struct literal construction sites — incomplete constructors become conflicts.

`propagate_add_variant` walks match expressions on the enum — non-exhaustive matches become conflicts.

---

## Integration Into `apply()`

```rust
fn apply(&mut self) -> Result<ChangeReport> {
    let mut all_rewrites: Vec<SymbolEdit> = vec![];
    let mut all_conflicts: Vec<EditConflict> = vec![];

    for queued in &self.changesets {
        // 1. structural edit at definition site
        let ast = self.registry.asts.get_mut(&file)?;
        apply_node_op(ast, &self.registry.handles, &queued.symbol_id, &queued.op)?;

        // 2. propagate to reference sites
        let prop = propagate(&queued.op, &queued.symbol_id, &self.registry, &*self.oracle)?;
        all_rewrites.extend(prop.rewrites);
        all_conflicts.extend(prop.conflicts);
    }

    // 3. apply reference site rewrites
    for edit in &all_rewrites {
        let ast = self.registry.asts.get_mut(&edit.file)?;
        apply_symbol_edits_to_ast(ast, &[edit.clone()])?;
    }

    // 4. collect touched files
    let touched = /* union of definition files + rewrite files */;

    Ok(ChangeReport { touched_files: touched, conflicts: all_conflicts })
}
```

---

## Updated GOAL_PROMPT.md

```markdown
# GOAL_PROMPT.md — Structural AST Editor with rustc Integration

## Status: Pipeline Verified — Propagation Layer Next

---

## What Is Fully Built and Verified

- NodeRegistry, NodeHandle, NodeOp, FieldMutation — complete
- ProjectEditor: load, load_with_rustc, queue, queue_by_id, apply, validate,
  commit, preview — complete
- OracleData from GraphSnapshot, all 4 oracle methods — complete
- Symbol ID join key — verified (syn == rustc, true)
- Smoke test passed: 5 files touched, written, no false conflicts

---

## What Is Being Built: Propagation Layer

File to create: `rename/core/project_editor/propagate.rs`

### Core Type

```rust
pub struct PropagationResult {
    pub rewrites: Vec<SymbolEdit>,   // auto-applicable
    pub conflicts: Vec<EditConflict>, // require human decision
}

pub fn propagate(
    op: &NodeOp,
    symbol_id: &str,
    registry: &NodeRegistry,
    oracle: &dyn StructuralEditOracle,
) -> Result<PropagationResult>
```

### Dispatch Table

| Op | Propagation function | Output |
|---|---|---|
| RenameIdent | propagate_rename | rewrites at all reference sites |
| DeleteNode | propagate_delete | conflicts at all use sites |
| ChangeVisibility | propagate_visibility | conflicts where visibility leaks |
| ReplaceSignature | propagate_signature | conflicts/rewrites at call sites |
| RemoveStructField | propagate_remove_field | conflicts at field access sites |
| RemoveVariant | propagate_remove_variant | conflicts at match arm sites |
| AddStructField | propagate_add_field | conflicts at construction sites |
| AddVariant | propagate_add_variant | conflicts at non-exhaustive matches |
| AddAttribute, RemoveAttribute | none | no propagation needed |
| InsertBefore, InsertAfter | none | no propagation needed |
| ReorderItems | none | no propagation needed |
| ReplaceNode | propagate_delete + rename | conflicts + rewrites |

### Data Sources Per Function

propagate_rename:
  source: EnhancedOccurrenceVisitor across registry.asts
  output: SymbolEdit per occurrence (text rewrite)

propagate_delete:
  source: oracle.impact_of + oracle.cross_crate_users
  output: EditConflict per use site (cannot auto-resolve)

propagate_visibility:
  source: oracle.cross_crate_users + AliasGraph.analyze_visibility_leaks
  output: EditConflict where reduced visibility leaks

propagate_signature:
  source: oracle.satisfies_bounds per call site
         oracle.impact_of for call site list
  output: EditConflict where bounds fail
          SymbolEdit where param names changed but types compatible

propagate_remove_field:
  source: walk registry.asts for ExprField + PatStruct
  output: EditConflict per access site

propagate_remove_variant:
  source: walk registry.asts for match arms on this enum
  output: EditConflict per arm site

propagate_add_field (non-optional only):
  source: walk registry.asts for struct literal construction
  output: EditConflict per incomplete constructor

propagate_add_variant:
  source: walk registry.asts for non-exhaustive match on this enum
  output: EditConflict per match site

### Integration Into apply()

After apply_node_op (definition site):
  1. call propagate(op, symbol_id, registry, oracle)
  2. collect rewrites + conflicts from result
  3. apply rewrites via apply_symbol_edits_to_ast
  4. merge conflicts into ChangeReport

ChangeReport.conflicts is the human decision surface.
ChangeReport.touched_files includes definition + rewrite sites.

---

## Propagation Priority Order

Build in this order (each is independently useful):

1. propagate_rename — highest value, uses existing occurrence visitor
2. propagate_delete — safety net, blocks destructive ops at use sites
3. propagate_remove_field — catches field removal breakage
4. propagate_remove_variant — catches match exhaustiveness breakage
5. propagate_visibility — uses oracle + alias graph
6. propagate_signature — most complex, uses oracle.satisfies_bounds
7. propagate_add_field — constructor completeness check
8. propagate_add_variant — match exhaustiveness check

---

## Files To Create

- `rename/core/project_editor/propagate.rs` — PropagationResult + all functions

## Files To Modify

- `rename/core/project_editor/mod.rs` — wire propagate() into apply()
- `rename/core/project_editor/ops.rs` — no changes needed

## Files Already Providing Infrastructure

- `rename/occurrence.rs` — EnhancedOccurrenceVisitor (used by propagate_rename)
- `rename/alias/visibility.rs` — analyze_visibility_leaks (used by propagate_visibility)
- `rename/core/rename.rs` — apply_symbol_edits_to_ast (used by apply)
- `rename/core/oracle.rs` — StructuralEditOracle (used by all propagate_* fns)

---

## What Is NOT Changing

- All existing rename CLI — unchanged
- AstEdit + text-level passes — unchanged
- NodeOp / FieldMutation enums — unchanged
- OracleData / CargoProject — unchanged
- NullOracle — unchanged (propagation with NullOracle returns
  empty conflicts, rewrites still work for rename via occurrence visitor)
```
