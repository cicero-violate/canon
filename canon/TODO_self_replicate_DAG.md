# Self-Replication Work DAG

**Goal:** Refactor Canon so `CanonicalIr` is purely semantic while all filesystem/module/file routing moves into a standalone `LayoutMap`, enabling arbitrary isomorphic reshaping of projects.

This document restates the blockers from `TODO_self_replicate.md` as a dependency graph so we can schedule incremental agents against the work.

## Legend
- `[ ]` unchecked tasks are still open
- `→` indicates prerequisite relationship

## DAG
1. `[ ]` **ING-001: Rust source ingestor**
   - Parse `src/**/*.rs` (via `syn`) → Canonical IR.
   - Emit modules, structs, enums, traits, impls, functions, const/static/type aliases, mod attributes.
   - Reconstruct call + module edges and populate `file_id`/`attributes`/`statics`.
   - _Status_: filesystem discovery + `syn` parsing stubs landed; builder now emits modules, structs, enums, traits, free functions, impl blocks, module edges, call edges, and per-module const/static/use metadata. AST coverage + round-trip validation still pending.

2. `[x]` **LAY-001: Semantic/Layout inventory**
   - Find every IR struct carrying `file_id`, `module_path`, or filesystem metadata; capture file + line numbers.
   - Output table drives later cleanup work.
   - *Depends on ING-001* (need real data to inspect).
   - _Status_: Added `cargo run --bin layout_inventory` which scans `src/ir/**/*.rs` and writes `docs/layout_inventory.md` with a live table of every layout-bound field + source location.

3. `[x]` **LAY-002: `LayoutMap` and graph split design**
   - Define `LayoutMap`, `SemanticGraph`, `LayoutGraph`, and `LayoutStrategy` trait APIs.
   - `SemanticGraph` holds only nodes + semantic edges; `LayoutGraph` holds file routing + `use` statements.
   - *Depends on LAY-001 for inventory*.
   - _Status_: Added `src/layout/mod.rs` plus re-exports so downstream code can build strategies atop the new data structures.

4. `[ ]` **LAY-003: Remove layout metadata from semantic nodes**
   - Drop `file_id` (and similar) from `Function`, `Struct`, `Trait`, `Enum`, etc.; ensure identity is NodeId-only.
   - Update constructors/evolution/renderers accordingly.
   - *Depends on LAY-002 definitions*.
   - _Exit criteria_: no type in `crate::ir` exposes filesystem-derived fields.
   - `[x]` **LAY-003a**: delete legacy layout fields/structs from `crate::ir` + evolution/validator plumbing.
   - `[x]` **LAY-003b**: migrate DOT import/export + auto-dot to emit/consume `LayoutGraph` updates directly (no sidecar topology patching). *(Auto DOT now patches topology + routing hints before returning; CLI no longer applies manual file edits.)*
   - `[x]` **LAY-003c**: wire CLI + materializer invocations to surface layout JSON alongside semantic IR (interim manual flag). `canon ingest` emits both halves; CLI commands accept `--layout`.
   - `[x]` **LAY-003d**: integrate layout routing deltas into proposal acceptance (`accept_proposal` / auto-accept helpers persist updated `LayoutGraph`).
   - `[x]` **LAY-003e**: finish DOT round-trip validators so they only read/write layout data (`LayoutGraph`) and never reference legacy `Module.files`. *(All DOT verify/export paths consume layout modules/files exclusively; legacy `Module.files` references removed.)*

5. `[x]` **LAY-004: ING emits `(SemanticGraph, LayoutGraph)`**
   - Refactor ingestion to return a `LayoutMap { semantic, layout }` instead of `CanonicalIr`.
   - Serialize layout metadata separately (`layout_map.json`) so CLI/tests can diff each half independently.
   - Preserve original `use` statements only in `LayoutGraph`.
   - *Depends on LAY-003*.
   - _Status_: `ingest_workspace` now returns `LayoutMap` and builder populates per-node routing assignments.

6. `[x]` **SEM-001: CanonicalIr builder consumes SemanticGraph**
   - Introduce `SemanticIrBuilder` that converts a `SemanticGraph` into `CanonicalIr` for validators/evolution.
   - Ensure `validate_ir` and evolution paths read only semantic data while layout consumers read `LayoutGraph`.
   - *Depends on LAY-004*.
   - _Status_: `SemanticIrBuilder` converts a semantic graph into a legacy `CanonicalIr` shell so existing validation/evolution paths remain functional while we migrate layout consumers.

7. `[ ]` **LAY-005: Materializer consumes both graphs**
   - `materialize(&SemanticGraph, &LayoutGraph)` plus `LayoutStrategy` implementations.
   - Implement `Original`, `SingleFile`, `PerTypeFile` strategies.
   - *Depends on SEM-001*.
   - `[x]` _Stage 1_: materializer takes explicit `LayoutGraph`; renderers consult layout routing.
   - `[x]` _Stage 2_: finish migrating render/diff tooling to layout data (DOT + CLI auto-materialize paths, layout deltas applied atomically). *DOT import/export, CLI materialize/decide/import, and layout validators now consume `LayoutGraph` exclusively; layout deltas from DOT are applied atomically.*
   - `[x]` **LAY-005a**: expose `LayoutStrategy` planning hooks + `SemanticIrBuilder` shims so strategies can materialize alternate layouts. *New `canon layout --strategy {original|single-file|per-type}` command builds alternate layouts using the built-in strategy implementations.*
   - `[x]` **LAY-005b**: teach proposal acceptance/import flows to persist layout edits atomically (no manual file topology patching). *`accept_proposal` now carries layout deltas end-to-end and auto-accept DOT proposals inject routing/topology updates immediately.*

8. `[ ]` **LAY-006: Layout invariants + tests**
   - Ensure `ingest(materialize(G, L)).semantic == G.semantic` for each strategy.
   - Add `canon layout --strategy ...` harness and debug diff.
   - *Depends on LAY-005*.
   - `[x]` **LAY-006a**: validator enforces `LayoutGraph` soundness (nodes routed exactly once to existing files). *Validator now enforces routing completeness + `use_block` hygiene.*
   - `[x]` **LAY-006b**: regression tests for ingest→materialize round-trip invariants. *New `tests/layout_roundtrip.rs` materializes a demo IR + layout, ingests the output, and asserts semantic equality.*
   - `[x]` **LAY-006c**: CLI harness (`canon layout --strategy ...`) for experimenting with layout planners. *Command now emits planned layouts, validates them, and can diff against a base layout via `--diff`.*

9. `[ ]` **LAY-007: Layout serialization + CLI plumbing**
   - Extend CLI/serde surface so `LayoutGraph` persists alongside the semantic IR.
   - `canon ingest` exports both halves; `canon materialize` accepts `--layout foo.layout.json` overrides.
   - *Depends on LAY-006*.
   - `[x]` _Stage 1_: CLI commands accept layout JSON inputs (materialize/decide/import/submit DSL/Fn AST + new `canon ingest` output now updated).
   - `[x]` _Stage 2_: persist layout graphs during proposal acceptance/evolution so CLI no longer needs manual flags. *Layout paths default to `<ir>.layout.json`, write-backs pair with `output_ir`, and `canon ingest` auto-writes the layout artifact when no explicit path is supplied.*

2. `[x]` **TYP-001: Lifetime parameters on functions**
   - Add `lifetime_params: Vec<String>` to `Function` / `FunctionSignature`.
   - Render lifetimes ahead of type generics.
   - Extend validator to enforce unique lifetime params.
   - _Status_: Completed — IR fields, renderer, and validator landed (see PR with lifetime params).

3. `[x]` **TYP-002: `Self` / `impl Trait` / `dyn Trait` support**
   - Add `TypeKind::SelfType`, `TypeKind::ImplTrait`, `TypeKind::DynTrait`.
   - Render them and teach validator to allow them.
   - _Status_: Renderer now formats `Self`, `impl Foo`, `dyn Foo` and TypeKind variants exist in IR.

4. `[ ]` **AST-001: Macro invocations & structured patterns**
   - Represent `macro_invocation`, typed closure params, `let-else`, `if let`, tuple/struct patterns, match guards.
   - Update renderer + validator’s kind whitelist.
   - *Depends on ING-001* (ingestor must populate these nodes).

5. `[x]` **MAT-001: File placement routing**
   - Use `LayoutGraph` assignments to route structs/traits/functions into sibling files.
   - Required so materialized tree matches ingested tree.
   - *Depends on ING-001* (needs populated layout assignments).
   - _Status_: ING now emits per-node `LayoutAssignment`s and the materializer consults them to render multi-file modules; `tests/layout_roundtrip.rs` verifies the materialize→ingest cycle preserves semantic structure via these assignments.

6. `[ ]` **USE-001: Exact `use` preservation**
   - Store per-file `use` statements in each `LayoutFile` with absolute fidelity (order/commented segments).
   - Emit them verbatim before Canon’s synthesized `use super` / `use crate`.
   - *Depends on ING-001*.

7. `[x]` **CLI-ING: `canon ingest` command**
   - Wire `ingest/` module into CLI (`canon ingest --src ./canon --semantic-out foo.semantic.json --layout-out foo.layout.json`).
   - Validate + materialize round-trip harness.
   - *Depends on ING-001` + `TYP-001` + `TYP-002`.
   - _Status_: `canon ingest` now emits both semantic and layout JSON artifacts.

8. `[ ]` **SR-READY: Self-replication harness**
   - Script: `ingest → validate → materialize → cargo build` and compare output.
   - Requires `ING-001`, `TYP-001`, `TYP-002`, `AST-001`, `MAT-001`, `USE-001`.

## Execution Order
1. TYP-001 → TYP-002 (small prerequisites)
2. ING-001 (large, parallelizable once type groundwork lands)
3. MAT-001, USE-001 (post-ingestor)
4. AST-001 (shared dependency; can be staggered alongside MAT/USE)
5. CLI-ING
6. SR-READY validation script
