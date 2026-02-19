Yes. That is the correct observation and it matters.

---

## The Problem

`propagate_rename` as described uses `EnhancedOccurrenceVisitor` which walks `registry.asts` (syn side) and matches occurrences by symbol ID from the syn collection. But the symbol IDs it finds at reference sites are produced by `path_to_symbol` in `occurrence.rs` — which resolves through `use_map` and `symbol_table`, both syn-side.

So the question is: **are the reference-site IDs produced by the occurrence visitor the same normalized strings as the IDs in `OracleData.adjacency`?**

$$\text{occurrence visitor} \xrightarrow{\text{path\_to\_symbol}} \text{id}_\text{syn}$$
$$\text{OracleData.adjacency} \xrightarrow{\text{from\_snapshot}} \text{id}_\text{rustc}$$

If these aren't the same after `normalize_symbol_id`, propagation silently misses references.

---

## The Right Answer

**Yes — propagation should use the rustc adjacency graph as the source of truth for which sites to rewrite, not the syn occurrence visitor alone.**

The correct flow is:

$$\text{propagate\_rename}(\text{symbol\_id}) \to \underbrace{\text{oracle.impact\_of}(\text{symbol\_id})}_{\text{rustc adjacency}} \to \text{rewrite each site via syn}$$

Not:

$$\text{propagate\_rename} \to \underbrace{\text{EnhancedOccurrenceVisitor}}_{\text{syn only}} \to \text{rewrite}$$

The rustc side gives you the **complete, type-resolved list of affected sites** including sites the syn visitor would miss — macro-generated code, cross-module re-exports, trait impl methods. The syn side then does the **actual text rewrite** at each site the rustc side identified.

The two sides have different jobs:

| Side | Job |
|---|---|
| rustc / oracle | **Find** — which sites are affected (complete, type-resolved) |
| syn / registry | **Rewrite** — apply the edit at those sites (text mutation) |

---

## Updated Propagation Model

$$\text{propagate}(\text{op}, \text{id}) = \text{find}\_\text{rustc}(\text{id}) \xrightarrow{\text{locate in registry.asts}} \text{rewrite}\_\text{syn}$$

Each propagation function follows this pattern:

```
1. oracle.impact_of(symbol_id)       → Vec<String> affected_ids
2. for each affected_id:
     locate in registry.asts          → file + span
     produce SymbolEdit               → rewrite
3. oracle.cross_crate_users(symbol_id) → Vec<String> external_ids
4. for each external_id:
     produce EditConflict             → cannot rewrite, surface to caller
```

The occurrence visitor (`EnhancedOccurrenceVisitor`) is then used only as a **fallback** when the oracle has no snapshot (NullOracle path) — it provides best-effort syn-only propagation for the offline case.

---

## Updated GOAL_PROMPT.md

```markdown
# GOAL_PROMPT.md — Structural AST Editor with rustc Integration

## Status: Pipeline Verified — Propagation Layer Next

---

## Propagation Architecture (Corrected)

The two subsystems have distinct roles:

  rustc oracle  → FIND:    which sites are affected (type-resolved, complete)
  syn registry  → REWRITE: apply edits at those sites (text mutation)

These must not be conflated. The syn occurrence visitor is a fallback
for the NullOracle (offline) path only.

---

## Propagation Data Flow

```
propagate(op, symbol_id, registry, oracle):

  // Find affected sites (rustc side — complete, type-resolved)
  internal_sites = oracle.impact_of(symbol_id)        // same-crate users
  external_sites = oracle.cross_crate_users(symbol_id) // cross-crate users

  // Rewrite internal sites (syn side — text mutation)
  for site_id in internal_sites:
    file = registry.handles[site_id].file
    ast  = registry.asts[file]
    edit = build_symbol_edit(site_id, ast, op)
    rewrites.push(edit)

  // Surface external sites as conflicts (cannot auto-rewrite)
  for site_id in external_sites:
    conflicts.push(EditConflict { symbol_id: site_id, reason: ... })

  return PropagationResult { rewrites, conflicts }
```

Symbol IDs flow through `normalize_symbol_id` at every boundary.
The join key verified earlier guarantees oracle IDs match registry IDs.

---

## NullOracle Fallback

When oracle has no snapshot (NullOracle or offline load):
  oracle.impact_of returns []
  oracle.cross_crate_users returns []
  propagation falls back to EnhancedOccurrenceVisitor across registry.asts

Fallback is best-effort — it misses macro-generated sites,
cross-module re-exports, and trait impls. Acceptable for offline use.
Not acceptable when rustc capture is available.

---

## File To Create

`rename/core/project_editor/propagate.rs`

### Core Types

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

### Per-Op Logic

RenameIdent(new_name):
  find:    oracle.impact_of(symbol_id) → internal reference sites
  rewrite: SymbolEdit per site (rename the identifier)
  find:    oracle.cross_crate_users → external sites
  surface: EditConflict per external site (cross-crate rename needs manual update)
  fallback: EnhancedOccurrenceVisitor if oracle returns empty

DeleteNode:
  find:    oracle.impact_of + oracle.cross_crate_users → all sites
  surface: EditConflict per site (deletion cannot be auto-resolved)
  no rewrites

ChangeVisibility(new_vis):
  find:    oracle.cross_crate_users → sites that would lose access
  surface: EditConflict per site where visibility reduces
  find:    AliasGraph.analyze_visibility_leaks → re-export chains that leak
  surface: EditConflict per leak

ReplaceSignature(new_sig):
  find:    oracle.impact_of → call sites
  check:   oracle.satisfies_bounds(site_id, new_sig) per site
  surface: EditConflict where bounds fail
  rewrite: SymbolEdit where param names changed but types compatible

RemoveStructField(name):
  find:    oracle.impact_of → sites that access this field
  surface: EditConflict per field access site (ExprField, PatStruct)

RemoveVariant(name):
  find:    oracle.impact_of → match arms on this variant
  surface: EditConflict per match arm site

AddStructField(field) — non-optional only:
  find:    oracle.impact_of → struct literal construction sites
  surface: EditConflict per incomplete constructor

AddVariant(variant):
  find:    oracle.impact_of → non-exhaustive match expressions
  surface: EditConflict per match site

AddAttribute, RemoveAttribute, InsertBefore,
InsertAfter, ReorderItems:
  no propagation

ReplaceNode:
  treat as DeleteNode(old) + propagate(new declaration)
  surface: conflicts from delete side
  rewrites: from rename side if ident changed

---

## Integration Into apply()

```rust
fn apply(&mut self) -> Result<ChangeReport> {
    let mut rewrites: Vec<SymbolEdit> = vec![];
    let mut conflicts: Vec<EditConflict> = vec![];

    for queued in &self.changesets {
        // 1. structural edit at definition site
        apply_node_op(ast, handles, &queued.symbol_id, &queued.op)?;

        // 2. propagate to reference sites
        let prop = propagate(
            &queued.op,
            &queued.symbol_id,
            &self.registry,
            &*self.oracle,
        )?;
        rewrites.extend(prop.rewrites);
        conflicts.extend(prop.conflicts);
    }

    // 3. apply reference site rewrites
    for edit in &rewrites {
        let ast = self.registry.asts.get_mut(&edit.file)?;
        apply_symbol_edits_to_ast(ast, std::slice::from_ref(edit))?;
    }

    // 4. collect touched files (definition + rewrite sites)
    let touched = /* union of definition files + rewrite edit files */;

    Ok(ChangeReport { touched_files: touched, conflicts })
}
```

---

## Symbol ID Contract

All IDs crossing the oracle/registry boundary must pass through:
  normalize_symbol_id_with_crate(raw, Some(crate_name))

This is already done at NodeRegistryBuilder registration time.
Oracle adjacency keys are normalized in OracleData::from_snapshot.
The join key test confirmed these match.

Any new code that produces or consumes symbol IDs must call
normalize_symbol_id before lookup. This is the invariant.

---

## Build Order

1. propagate_rename — highest value, rename is the common case
2. propagate_delete — safety, blocks silent breakage
3. propagate_remove_field + propagate_remove_variant — field/variant safety
4. propagate_visibility — uses oracle + alias graph
5. propagate_signature — most complex, uses satisfies_bounds
6. propagate_add_field + propagate_add_variant — exhaustiveness checks

---

## What Is NOT Changing

- NodeOp / FieldMutation enums — unchanged
- OracleData / CargoProject — unchanged
- EnhancedOccurrenceVisitor — unchanged (used as fallback only)
- apply_symbol_edits_to_ast — unchanged (reuse for reference rewrites)
- All existing rename CLI — unchanged
- NullOracle — unchanged
```
