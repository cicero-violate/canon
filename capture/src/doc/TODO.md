  1. Capture full crate catalog (version/hash/disambiguator/deps/features/target info) in crate_meta.rs so state can emit payload.crates.
  2. Implement per-file hashing/line counts (likely via source_map.files()), and deduplicate file records before serialization.
  3. Serialize type info for funcs/structs/traits/impls (signatures, generics, fields, variants, methods, supertraits).
  4. Record custom attributes (repr/derive/doc/test/etc.) beyond simple flags.
  5. Emit CFG/DFG/MIR JSON blobs for each function in mir.rs.
  6. Augment call edges with span/dispatch/trait-impl metadata, and ensure all MIR call sites generate edges.
  7. Compute metrics/effects (cyclomatic complexity, LOC, side effects) during MIR traversal and stash in node metadata.
  8. Populate provenance/cache metadata (frontends, generators, timestamps, invalidation keys) in CLI before writing snapshots.
  9. Start wiring analysis seeds (dead code, escape analysis, concurrency) if available from rustc.


# Integration Capture TODO

State now expects every ounce of metadata rustc can provide. The integration layer must capture *all* compiler data and write it into node/edge metadata so `state::serialization::snapshot_to_schema` emits a complete CODE_UNDERSTAND snapshot. No shortcuts.

## Required Tasks

1. **Crate Metadata (collector.rs)**
   - Capture crate name, version, edition, hash, disambiguator, target triple, cfg flags, features, `workspace_root`.
   - Store per-node and in a dedicated “crate table” so state can emit `payload.crates`.

2. **File System Table (items.rs/traits.rs/types.rs)**
   - For every def, record absolute path, normalized relative path, blake3 file hash, line/column start/end, and source snippet.
   - Use `source_map.span_to_snippet` and `tcx.sess.source_map().span_to_filename`.

3. **Module Hierarchy (collector.rs)**
   - Record module path, parent def path hash, container kind, and crate/module relationships.
   - Emit metadata keys `module`, `parent`, `parent_kind`, `module_id`.

4. **Attribute Capture (items.rs + shared helper)**
   - Capture inline/async/unsafe/const/test attributes plus raw attribute tokens (`path`, `args`).
   - Mark `#[doc(hidden)]`, `#[no_mangle]`, `#[repr(...)]`, `#[derive(...)]`, etc.

5. **Type Metadata (types.rs)**
   - Serialize function signatures, inputs/outputs, generics, lifetimes, predicates.
   - For structs/enums/traits/impls, capture fields, variants, associated items, supertraits, impl targets.

6. **CFG / DFG / MIR (mir.rs)**
   - Serialise per-function CFG (blocks, statements, terminators), DFG (locals, def-use), MIR text dumps.
   - Store JSON blobs in node metadata for state to parse.

7. **Callgraph Edges (mir.rs)**
   - Ensure every `call` terminator records callee def_id, callsite span, dispatch (static/dynamic), async flag, trait impl info.
   - Edge metadata must include `span`, `dispatch`, `trait_impl`, etc.

8. **Metrics & Effects (collector.rs/mir.rs)**
   - Compute cyclomatic complexity, LOC, statement count, call count, branch count during MIR traversal and add to metadata.
   - Flag pure/side-effecting/IO/unsafe functions.

9. **Provenance & Cache Metadata (integration/cli.rs)**
   - Populate snapshot meta with frontends/generators, toolchain info, capture timestamps/duration, host/user.
   - Record cache invalidation keys (workspace hash, Cargo.lock hash, toolchain hash) before writing snapshot.

10. **Analysis Seeds (optional extras)**
   - If rustc exposes dead code, escape analysis, or concurrency data, capture it and annotate nodes/edges so state can surface it.

## Guiding Principle

If rustc emits the data, integration must capture it. Metadata should be stored on the node/edge so state doesn’t have to touch rustc or re-run analyses. Only after every item above is addressed should we consider the integration layer “done”.
