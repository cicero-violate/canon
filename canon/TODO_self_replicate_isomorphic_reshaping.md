**GOAL:**
Refactor Canon so that `CanonicalIR` is purely semantic (layout-agnostic), and all filesystem/module/file routing is moved into a separate `LayoutMap`, enabling arbitrary isomorphic project reshaping.

---

1. Identify every struct in `CanonicalIR` that contains `file_id`, `module_path`, or filesystem-related metadata; list exact file + line numbers.
2. Propose a new `LayoutMap` struct that maps `{NodeId → FileId}` and `{FileId → ModulePath}`; define precise Rust types.
3. Remove `file_id` from `Function`, `Struct`, `Trait`, `Enum`, and any semantic node; update constructors accordingly.
4. Ensure no semantic node depends on directory structure for identity (identity must be stable NodeId only).
5. Define `SemanticGraph` type that contains only nodes + semantic edges (calls, implements, contains, type refs).
6. Define `LayoutGraph` type that contains file routing, module grouping, ordering, and preserved `use` blocks.
7. Modify ING-001 so ingestion outputs `(SemanticGraph, LayoutGraph)` instead of a layout-mixed IR.
8. Ensure ingestion reconstructs semantic edges independent of file/module boundaries.
9. Store original `use` statements inside `LayoutGraph` only; do not allow them inside semantic nodes.
10. Refactor materializer to accept `(&SemanticGraph, &LayoutGraph)` instead of a single mixed IR.
11. Add `trait LayoutStrategy { fn layout(&SemanticGraph) -> LayoutGraph; }`.
12. Implement `LayoutStrategy::Original` that reproduces ingested layout exactly.
13. Implement `LayoutStrategy::SingleFile` that emits entire project into one `lib.rs`.
14. Implement `LayoutStrategy::PerTypeFile` that groups each type into its own file.
15. Ensure semantic hashes (NodeId) remain invariant across layout transformations.
16. Add test: `ingest → materialize(SingleFile) → ingest` must produce identical `SemanticGraph`.
17. Add test: `ingest → materialize(PerTypeFile) → ingest` must produce identical `SemanticGraph`.
18. Verify `cargo build` succeeds for all layout strategies on Canon itself.
19. Add debug mode to diff semantic graphs only (ignore layout differences).
20. Document invariant: `∀ L, ingest(materialize(G, L)).semantic == G.semantic`.
