# Self-Replication Work DAG

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

5. `[ ]` **LAY-004: ING emits `(SemanticGraph, LayoutGraph)`**
   - Refactor ingestion to separate layout data (`LayoutGraph`) from semantic data (`SemanticGraph`).
   - Preserve original `use` statements only in `LayoutGraph`.
   - *Depends on LAY-003*.

6. `[ ]` **LAY-005: Materializer consumes both graphs**
   - `materialize(&SemanticGraph, &LayoutGraph)` plus `LayoutStrategy` implementations.
   - Implement `Original`, `SingleFile`, `PerTypeFile` strategies.
   - *Depends on LAY-004*.

7. `[ ]` **LAY-006: Layout invariants + tests**
   - Ensure `ingest(materialize(G, L)).semantic == G.semantic` for each strategy.
   - Add `canon layout --strategy ...` harness and debug diff.
   - *Depends on LAY-005*.

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

5. `[ ]` **MAT-001: File placement routing**
   - Use `Function.file_id` + `Module.files` to route structs/traits/functions into sibling files.
   - Required so materialized tree matches ingested tree.
   - *Depends on ING-001* (needs populated file IDs).

6. `[ ]` **USE-001: Exact `use` preservation**
   - Store per-file `use` statements with absolute fidelity (order/commented segments).
   - Emit them verbatim before Canon’s synthesized `use super` / `use crate`.
   - *Depends on ING-001*.

7. `[ ]` **CLI-ING: `canon ingest` command**
   - Wire `ingest/` module into CLI (`canon ingest --src ./canon --out canon_self.json`).
   - Validate + materialize round-trip harness.
   - *Depends on ING-001` + `TYP-001` + `TYP-002`.

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
