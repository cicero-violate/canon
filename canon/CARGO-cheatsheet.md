# Canon Layout Refactor Cheatsheet

Canonical IR is being split into two parts: a semantic graph (`layout::SemanticGraph`) and a layout graph (`layout::LayoutGraph`).

## New Types
* `LayoutMap { semantic: SemanticGraph, layout: LayoutGraph }`
* `LayoutGraph` contains `LayoutModule` (module id, name, file list, imports), `LayoutFile` (id/path/use block), and `LayoutAssignment { node, file_id, rationale }` describing which file renders which node.
* `SemanticGraph` mirrors the old Canonical IR contents but without filesystem metadata. Use `SemanticIrBuilder` to produce a legacy `CanonicalIr` reference for validators/evolution.

## Ingestion Flow
```
canon::ingest::ingest_workspace(opts) -> LayoutMap
```
Persist the split artifacts separately (e.g., `canon.semantic.json` + `canon.layout.json`) and pass
both halves to downstream commands like `canon materialize` and `canon decide`.
