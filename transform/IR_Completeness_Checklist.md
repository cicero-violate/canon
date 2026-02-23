# IR Completeness Checklist

## 1. Identity
1. Stable, global symbol IDs (not path/spans)
2. ID survives rename, move, re-export, inline ↔ file module changes
3. Cross-crate ID mapping (external deps)

## 2. Syntax Fidelity
1. Lossless token storage (whitespace/comments)
2. Exact span mapping between IR and source
3. Round-trip guarantee (no-op preserves source byte-for-byte)

## 3. Module & File Mapping
1. Canonical module path resolution (inline, mod.rs, nested)
2. Reverse mapping from module path → file(s)
3. Support for `#[path]` and `include!`

## 4. Name Resolution
1. Fully resolved def-use chains
2. `use` alias + re-export chain resolution
3. Glob import resolution with exact provenance
4. Shadowing and scope tracking

## 5. Type & Trait Semantics
1. Type inference edges for expressions
2. Trait impl ↔ trait ↔ type resolution
3. Method call resolution (UFCS + autoderef)
4. Associated type/value resolution

## 6. Call & Data Flow
1. Call graph edges (direct + dynamic)
2. Control flow graph per function
3. Data-flow / def-use edges for locals

## 7. Macro & Generated Code
1. Macro expansion provenance (token origin)
2. Generated symbols mapped back to macro invocations
3. Proc-macro attribute effects captured

## 8. Visibility & Access
1. Visibility graph across modules/crates
2. Private leakage detection
3. Export surface (public API) snapshot

## 9. Conditional Compilation
1. `cfg` edge annotations for all items
2. Alternative item graphs per cfg set
3. Dependency gating for features

## 10. Build & Crate Context
1. Target-specific graphs (bin/lib/test)
2. Feature-flagged symbol sets
3. Workspace + dependency metadata linked to IR

## 11. Editability
1. Edit plan supports rename/move/insert/delete with correct propagation
2. Conflict detection before apply
3. Precise rewrite targeting (span-based, not text glob)
