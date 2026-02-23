# Goal: Project-Wide Use-Path Updates for MoveSymbol

## Problem

`NodeOp::MoveSymbol` only rewrites `use` paths in files whose module path
starts with the old module prefix (`UsePathUpdate::applies_to`). Cross-module
consumers of the moved symbol are silently skipped.

## Goal

After a `MoveSymbol` operation, every `use` statement in the project that
resolves to the moved symbol must be rewritten to the new path — regardless
of which module or file it lives in.

---

## Required Changes

### 1. Switch from module-prefix diffusion → symbol-occurrence targeting

**File:** `rename/core/project_editor/propagate.rs`  
**Function:** `propagate_move`

Current behaviour emits `UsePathUpdate { from: old_module, to: new_module }`
and gates application on `applies_to(file_module_path)`.

New behaviour:
- Call `build_symbol_index_and_occurrences(registry)` to get the full
  occurrence list.
- Filter occurrences to those whose `id == symbol_id` and `kind == "use"`.
- Emit one `SymbolEdit` per matching occurrence (span-targeted rewrite).
- Return these edits in `PropagationResult::rewrites`.

### 2. Remove (or bypass) the prefix gate

**File:** `rename/core/project_editor/mod.rs`  
**Struct/fn:** `UsePathUpdate::applies_to` and `run_use_path_rewrite`

- `applies_to` currently returns `false` for files outside `M_old`.
- Either remove the gate entirely, or introduce a second code-path that
  applies span-targeted `SymbolEdit` rewrites without any prefix check.
- `run_use_path_rewrite` should accept the `SymbolEdit` list from
  `propagate_move` and apply them via `apply_rewrites`.

### 3. Use AliasGraph to discover re-export consumers

**File:** `rename/core/project_editor/propagate.rs`

After collecting direct occurrences, also call:
- `alias_graph.get_importers(old_symbol_path)` — files that import via
  simple `use`.
- `alias_graph.find_reexport_chains(symbol_id)` — files that re-export the
  symbol transitively.

Emit `SymbolEdit` rewrites for each importer/re-exporter node's span.

### 4. Wire edits back into ProjectEditor::apply

**File:** `rename/core/project_editor/mod.rs`  
**Function:** `ProjectEditor::apply`

`PropagationResult::rewrites` already flows into `apply_rewrites`. Confirm
that the new `SymbolEdit` entries from step 1 and 3 are included in that
vec and that `apply_rewrites` touches every file unconditionally.

---

## Invariants to Preserve

- A move within a prefix must not rewrite unrelated types that share a
  leaf name (the bug already fixed by scoping). Span-targeted edits are
  inherently safe — they rewrite a concrete span, not a name string glob.
- `dry_run` and `preview` paths must still work (no direct file writes
  during propagation).
- `validate()` must still detect conflicts before `apply()` mutates state.

---

## Definition of Done

Running `nodeop_movesymbol.rs` (without `--dry-run`) on a crate where a
moved symbol is imported from a different subtree must:
1. Update every `use` statement referencing the old path to the new path.
2. Leave unrelated `use` statements untouched.
3. Produce no compile errors after `cargo check`.
