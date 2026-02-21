# Session Report: `rename` fix_point Emission Pipeline

## Project Context

`rename` is a Rust refactoring tool. It captures the codebase via a rustc frontend into a graph snapshot (`WireNode` metadata), then emits a materialized copy called `fix_point` via `src/core/project_editor/graph_pipeline.rs`. The fix_point crate must compile cleanly as proof the rename pipeline is correct.

---

## Work Done This Session

### Problem
fix_point had **1163 compile errors** (85% E0425 + E0433 — unresolved names/paths).

### Root Cause Chain Discovered

**Gap 1: `use` statements never emitted** *(FIXED)*

`project_plan` in `graph_pipeline.rs` called `is_top_level_item` which only allowed `struct | enum | fn | ...` — `"use"` was explicitly excluded. All import statements were silently dropped from every emitted file.

Fix: after computing `orig_path = module_file_path(module_path, has_children, project_root)`, parse the original source file and extract all `syn::Item::Use` items, render them with `render_node`, and prepend to `file_items`.

**Gap 2: Wrong source file for `use` extraction** *(FIXED)*

Initial implementation used `source_file` metadata from nodes to find the original file. But `source_file` is only set by the rustc frontend on item nodes, not module nodes — causing cross-contamination (e.g. `core/mod.rs` use statements leaking into every module's emitted file).

Fix: derive orig_path deterministically using `module_file_path(module_path, has_children, project_root)` — the same function used to compute the output path, pointed at the original `src/` tree. Added fallback: if computed path is `mod.rs` but doesn't exist, try sibling `.rs` file (handles `src/fs.rs` → `src/fs/mod.rs` layout difference).

**Result after both fixes: 1163 → 153 errors**

---

## Remaining Errors (153 total, fresh snapshot confirmed)

### E0432 (4) — Unresolved imports after symbol move
```
fix_point/src/core/project_editor/cross_file.rs:7  → super::utils::find_project_root_sync
fix_point/src/core/project_editor/editor.rs:34     → super::utils::{build_symbol_index, find_project_root}
fix_point/src/core/project_editor/use_path.rs:1    → super::utils::find_project_root
fix_point/src/scope/mod.rs:3                       → crate::scope::frame::ScopeFrame
```
The example `nodeop_movesymbol.rs` moves `utils::{build_symbol_index, find_project_root, find_project_root_sync}` into `editor`. After the move, `utils.rs` is emitted empty (only use statements, no function bodies). Files that imported from `super::utils` now have broken imports. The pipeline needs to either: (a) emit re-exports in `utils.rs` pointing to the new location, or (b) rewrite callers' import paths to point to `super::editor`.

### E0603 (16) — Private items used across modules
```
fix_point/src/core/collect/mod.rs:35       → struct ItemCollector is private
fix_point/src/core/project_editor/editor.rs → apply_cross_file_moves, collect_new_files (pub(super))
fix_point/src/core/project_editor/editor.rs → GraphSnapshotOracle, NodeRegistryBuilder (pub(crate)/pub(super))
```
Items with `pub(crate)` or `pub(super)` in the original are being emitted with those restricted visibilities in fix_point. Since fix_point is a separate crate, these are effectively private. The `normalize_visibility` function in `graph_pipeline.rs` now maps `"crate"` → `"pub "` but `"super"` visibility (`pub(super)`) is mapped as `restricted:` and falls through to `format!("pub({path}) ")` — which produces `pub(super)` in the output, still inaccessible.

### E0425 / E0433 (69 + 38) — Symbols not in scope
Many of these are **cascading** from E0432 and E0603 above — once those are fixed the cascade should collapse. Key independent ones:
```
fix_point/src/core/project_editor/editor.rs → SymbolIndex, add_file_module_symbol, collect_symbols
fix_point/src/occurrence/mod.rs             → ScopeBinder, PatternBindingCollector, SymbolOccurrence
fix_point/src/fs/mod.rs                     → PathBuf, WalkDir, IncludeVisitor
```
These indicate either missing `pub use` re-exports in module `mod.rs` files, or items that are `pub(crate)` in their defining module and therefore not visible.

### E0405 / E0422 (6 + 7) — Trait/type not found in lib.rs
```
fix_point/src/lib.rs → StructuralEditOracle, VisitMut (traits)
fix_point/src/lib.rs → VisibilityLeakAnalysis, ExposurePath, LeakedSymbol (structs)
```
`lib.rs` has impl blocks emitted into it that belong in submodules. These impls reference types from those submodules without qualifying paths. Root cause: impl nodes whose `module_path` resolves to `crate` (the root) get emitted into `lib.rs`, but their body references types that are only in scope inside the original submodule.

### unknown / serde (11) — Missing derive attributes
```
fix_point/src/api.rs         → #[serde(default)] with no #[derive(Serialize, Deserialize)]
fix_point/src/model/types.rs → same
```
`source_snippet` is captured by the rustc frontend via `extract_item_snippet` using byte-range slicing. The span starts at `span.lo()` which in rustc **excludes outer attributes** like `#[derive(...)]`. So the snippet has the struct body but not the derive. The `render_item_from_ast` fallback via syn would include them, but `extract_item_snippet` in `metadata_capture.rs` takes priority when the snippet parses.

---

## Key Files

| File                                                       | Role                                                 |
|------------------------------------------------------------+------------------------------------------------------|
| `src/core/project_editor/graph_pipeline.rs`                | Emission pipeline — all fixes applied here           |
| `compiler_capture/src/frontends/rustc/metadata_capture.rs` | Captures `source_snippet`, `visibility`, `node_kind` |
| `compiler_capture/src/frontends/rustc/node_builder.rs`     | Builds `WireNode` payloads                           |

---

## Recommended Next Steps (Priority Order)

**1. Fix `source_snippet` to include outer attributes** in `metadata_capture.rs`:
`extract_item_snippet` slices from `span.lo()` — but rustc spans exclude outer attributes. Walk backwards from `start` to include `#[...]` lines before the item. This fixes the serde derive problem at source.

**2. Fix `pub(super)` → `pub` promotion** in `graph_pipeline.rs` `normalize_visibility`: the `restricted:` arm currently emits `pub(super)` unchanged. For fix_point emission, all restricted visibilities should be promoted to `pub`.

**3. Fix import rewriting after symbol moves**: when a symbol moves from module A to module B, the pipeline must rewrite `use super::A::fn_name` → `use super::B::fn_name` in all files that imported from A. This is the `utils` → `editor` move case.

**4. Fix impl blocks landing in lib.rs**: impl nodes with `module_path = "crate"` or container items that resolve to the root should carry their defining module path. Check whether `module_path` metadata is set correctly for impl nodes in `metadata_capture.rs`.
