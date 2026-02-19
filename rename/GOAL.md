# GOAL_PROMPT.md — Structural AST Editor with rustc Integration

## Project Location
`canon_workspace/rename/src/`

## What This Project Is

A **project-level structural Rust AST editor** built on two pillars:

1. `rename/` — syn-based symbol index, span tracking, structured pass pipeline
2. `integration/` — rustc frontend (HIR, MIR, types, traits, metadata, capture pipeline)

The rename side owns **write authority** (NodeHandle, NodeOp, render, commit).
The integration side owns **read oracle** (type resolution, impact analysis, macro expansion, cross-crate visibility).

---

## Architecture Equation

```
S_full = S_syn ⋈ S_rustc

S_syn   → NodeHandle (live syn::Item pointer, editable)
S_rustc → type truth (read-only, validation before commit)
```

---

## Layers To Build

### Layer 1 — Node Identity (`NodeHandle`, `NodeRegistry`)

**Location:** `rename/structured/` or new `rename/node/`

Every symbol in `SymbolIndex` gains a `NodeHandle`:

```rust
struct NodeHandle {
    file: PathBuf,
    item_index: usize,       // index in syn::File::items
    nested_path: Vec<usize>, // drill into impl blocks, mods
    kind: NodeKind,
}

enum NodeKind { Fn, Struct, Enum, Trait, Impl, ImplFn, Use, Mod, Type, Const }

struct NodeRegistry {
    handles: HashMap<String, NodeHandle>,  // symbol_id -> handle
    asts: HashMap<PathBuf, syn::File>,     // file -> live AST
}
```

**Key invariant:** symbol ID string (`crate::module::Name`) is the join key
between `NodeRegistry` and `integration/project.rs`.

---

### Layer 2 — Node Operations (`NodeOp`)

**Location:** `rename/structured/ops.rs` (new file)

Replace span-anchored `AstEdit` with typed node operations:

```rust
enum NodeOp {
    ReplaceNode   { handle: NodeHandle, new_node: syn::Item },
    InsertBefore  { handle: NodeHandle, new_node: syn::Item },
    InsertAfter   { handle: NodeHandle, new_node: syn::Item },
    DeleteNode    { handle: NodeHandle },
    MutateField   { handle: NodeHandle, mutation: FieldMutation },
    ReorderItems  { file: PathBuf, new_order: Vec<String> },
}

enum FieldMutation {
    RenameIdent(String),
    ChangeVisibility(syn::Visibility),
    AddAttribute(syn::Attribute),
    RemoveAttribute(String),
    ReplaceSignature(syn::Signature),
    AddStructField(syn::Field),
    RemoveStructField(String),
    AddVariant(syn::Variant),
    RemoveVariant(String),
}
```

`AstEdit` (text/span edits) stays alive for `DocAttrPass` and `UsePathRewritePass`.
`NodeOp` is additive — it does not replace the existing text edit path.

---

### Layer 3 — Project Editor (`ProjectEditor`)

**Location:** `rename/core/project_editor.rs` (new file)

```rust
struct ProjectEditor {
    registry: NodeRegistry,
    changesets: HashMap<PathBuf, Vec<NodeOp>>,
    oracle: Box<dyn StructuralEditOracle>,
}

impl ProjectEditor {
    fn load(project: &Path) -> Result<Self>
    fn queue(&mut self, symbol_id: &str, op: NodeOp) -> Result<()>
    fn apply(&mut self) -> Result<ChangeReport>
    fn validate(&self) -> Result<Vec<EditConflict>>  // calls oracle
    fn commit(&self) -> Result<Vec<PathBuf>>
    fn preview(&self) -> Result<String>
}
```

Operations execute against `Vec<syn::Item>` by index — never by byte offset.
Order of application within a file: declaration order, top to bottom.

---

### Layer 4 — Oracle Trait (Bridge to `integration/`)

**Location:** `rename/core/oracle.rs` (new file)

```rust
trait StructuralEditOracle {
    fn impact_of(&self, symbol_id: &str) -> Vec<String>;
    fn satisfies_bounds(&self, id: &str, new_sig: &syn::Signature) -> bool;
    fn is_macro_generated(&self, symbol_id: &str) -> bool;
    fn cross_crate_users(&self, symbol_id: &str) -> Vec<String>;
}
```

`integration/project.rs` implements this trait.
`ProjectEditor` holds `Box<dyn StructuralEditOracle>`.
The two subsystems stay decoupled — joined only by symbol ID strings.

---

### Layer 5 — Round-Trip Render

**Location:** `rename/structured/ast_render.rs` (extend existing)

Add `prettyplease` as a dependency.

```rust
fn render_file(ast: &syn::File) -> String {
    prettyplease::unparse(ast)
}
```

Fidelity guarantee: `parse(render(ast)) == ast` for all valid ASTs.

Comment preservation caveat: comments are lost through syn. If needed,
use a hybrid: keep original source, overwrite only changed node ranges
using existing span infrastructure as fallback.

---

## Symbol ID Normalization (Critical)

Both `rename/core/collect/mod.rs` and `integration/project.rs` independently
compute symbol IDs. Before any join works, verify they produce identical
strings for the same symbol.

Write a normalization function in a shared location:

```
rename/core/symbol_id.rs  (new file)
fn normalize_symbol_id(raw: &str) -> String
```

Both sides call this. The join key must be identical or the oracle bridge silently fails.

---

## Minimal Rebuild State Needed

The integration crate needs to be wired into the rename crate's module tree.
`src/lib.rs` needs `pub mod integration;`
`integration/mod.rs` needs to compile cleanly against this crate's dependencies.
`integration/project.rs` needs to implement `StructuralEditOracle`.

This is the minimal state. Do not refactor integration internals until the
oracle trait is implemented and the join key is verified.

---

## What Is NOT Changing

- `rename/core/collect/` — syn-based collection stays as-is
- `rename/structured/orchestrator.rs` — pass pipeline stays as-is
- `AstEdit` + text-level passes — stays as-is
- `rename/api.rs` — `MutationRequest` / `UpsertRequest` stays, gains `NodeOp` variant later
- All existing rename functionality continues to work unchanged

---

## Build Order

1. Verify symbol ID normalization between both sides
2. Build `NodeHandle` + `NodeRegistry`
3. Build `NodeOp` + `FieldMutation`
4. Build `StructuralEditOracle` trait
5. Implement trait in `integration/project.rs`
6. Build `ProjectEditor` with validate + commit
7. Add `prettyplease` render
8. Extend `api.rs` with `NodeOp` variant

---

## Dependencies To Add

```toml
prettyplease = "0.2"
```

All other dependencies (syn, proc-macro2, quote) already present.

---

## Files To Create

- `rename/structured/ops.rs`
- `rename/core/project_editor.rs`
- `rename/core/oracle.rs`
- `rename/core/symbol_id.rs`

## Files To Extend

- `rename/core/types.rs` — add `NodeHandle`, `NodeKind`, `NodeRegistry`
- `rename/structured/ast_render.rs` — add `render_file` via prettyplease
- `rename/api.rs` — add `NodeOp` variant to `UpsertRequest`
- `src/lib.rs` — add `pub mod integration`
