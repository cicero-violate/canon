# Canon — Full Rust Code Generation TODO

## Status

The pipeline currently generates:
- Project scaffold (Cargo.toml, src/lib.rs, module directories)
- File layout from DOT intra-module topology
- Struct definitions with fields
- Trait definitions with function signatures
- Impl blocks with stub function bodies
- Real function bodies from JSON AST nodes (block, let, if, match, while, return, call, lit)
- use statements from inter-module imported_types

---

## 1. IR gaps — things Canon cannot yet express

### 1.1 Expression nodes missing from AST renderer
- [ ] Binary operators: `a + b`, `a - b`, `a * b`, `a / b`, `a % b`
- [ ] Comparison operators: `a == b`, `a != b`, `a < b`, `a > b`
- [ ] Logical operators: `a && b`, `a || b`, `!a`
- [ ] Field access: `self.field`, `a.b.c`
- [ ] Index: `a[i]`
- [ ] Method call: `a.method(args)`  — distinct from free `call`
- [ ] Struct literal: `Foo { field: value }`
- [ ] Tuple: `(a, b, c)`
- [ ] Array literal: `[a, b, c]`
- [ ] Reference / dereference: `&x`, `*x`, `&mut x`
- [ ] Closure: `|args| body`
- [ ] Range: `0..n`, `0..=n`
- [ ] Cast: `x as u64`
- [ ] Question mark operator: `expr?`

### 1.2 Statement nodes missing
- [ ] `for` loop: `for x in iter { body }`
- [ ] `loop` with `break`/`continue`
- [ ] `break` with value
- [ ] `continue`
- [ ] Assignment: `x = value`
- [ ] Compound assignment: `x += value`, `x -= value` etc.
- [ ] `let mut` binding

### 1.3 Type system — TypeRef is too simple
- [ ] Generic types: `Vec<T>`, `Option<T>`, `Result<T, E>`
- [ ] Reference types: `&T`, `&mut T`
- [ ] Tuple types: `(A, B)`
- [ ] Slice types: `&[T]`
- [ ] Function pointer types: `fn(A) -> B`
- [ ] Lifetime annotations: `&'a T`
- [ ] `Self` type in trait contexts

### 1.4 Struct features
- [ ] Tuple structs: `struct Foo(A, B);`
- [ ] Unit structs: `struct Marker;`
- [ ] Derive macros: `#[derive(Debug, Clone, ...)]`
- [ ] `pub(crate)` and `pub(super)` visibility
- [ ] Doc comments on structs and fields

### 1.5 Trait features
- [ ] Default method bodies in traits
- [ ] Associated types: `type Output;`
- [ ] Associated constants: `const N: usize;`
- [ ] Trait bounds: `T: Trait + OtherTrait`
- [ ] Supertrait declarations: `trait Foo: Bar`
- [ ] Generic trait parameters: `trait Foo<T>`

### 1.6 Function features
- [ ] `self`, `&self`, `&mut self` receiver — currently no receiver concept in IR
- [ ] Default parameter values (via wrapper fns)
- [ ] `async fn` flag on Function
- [ ] `unsafe fn` flag on Function
- [ ] Generic parameters on functions: `fn foo<T: Trait>(x: T)`
- [ ] Where clauses: `where T: Debug`
- [ ] Variadic inputs (via slice type)
- [ ] Doc comments on functions

### 1.7 Enum support — entirely missing
- [ ] `DeltaPayload::AddEnum` variant in ir.rs
- [ ] `EnumNode { id, name, module, variants: Vec<EnumVariant> }` in IR
- [ ] `EnumVariant { name, fields: EnumVariantFields }` — unit / tuple / struct variants
- [ ] `apply_structural_delta` arm for `AddEnum`
- [ ] `render_enum` in materialize/
- [ ] Enum in `validate/check_artifacts.rs`

### 1.8 Module-level items missing
- [ ] `use` re-exports: `pub use crate::foo::Bar;`
- [ ] Module-level constants: `pub const MAX: usize = 100;`
- [ ] Module-level statics
- [ ] Type aliases: `type Result<T> = std::result::Result<T, Error>;`
- [ ] Attribute macros on modules: `#![allow(dead_code)]`

---

## 2. Delta pipeline gaps

### 2.1 Missing DeltaPayload variants
- [ ] `UpdateFunctionInputs { function_id, inputs }` — change a function's input ports
- [ ] `UpdateFunctionOutputs { function_id, outputs }` — change output ports
- [ ] `UpdateStructVisibility { struct_id, visibility }`
- [ ] `AddEnumVariant { enum_id, variant }`
- [ ] `RemoveField { struct_id, field_name }` — with proof requirement
- [ ] `RenameArtifact { kind, old_id, new_id }` — rename with full ref-update

### 2.2 auto_accept_fn_ast structural flaw (Gap 1)
- [ ] Replace fake Proposal+Trait scaffolding with a dedicated `FunctionBodyProposal`
      type that bypasses `enforce_proposal_ready` legitimately
- [ ] Or: relax `enforce_proposal_ready` to allow proposals with no edges
      when `proposal.kind == ProposalKind::FunctionBody`
- [ ] Add `ProposalKind` enum to IR: `Structural | FunctionBody | SchemaEvolution`

---

## 3. Materialize gaps

### 3.1 File placement
- [ ] When a function has a known file assignment (e.g. from DOT node),
      emit it into that specific file rather than always mod.rs
- [ ] `Function.file_id: Option<String>` field in IR pointing to a FileNode id

### 3.2 Imports
- [ ] Emit `use super::...` for intra-module cross-file references
- [ ] Emit `use crate::...` for intra-crate cross-module references
- [ ] Emit `use ::external_crate::...` for external dependencies
- [ ] Deduplicate use statements per file

### 3.3 Formatting
- [ ] Run `rustfmt` on emitted files if available on PATH
- [ ] Configurable indentation (spaces vs tabs)

### 3.4 Incremental materialization
- [ ] Only re-emit files whose content has changed (hash-based)
- [ ] Preserve hand-edited regions via `// canon:preserve` marker comments

---

## 4. DOT import/export gaps

### 4.1 Round-trip fidelity (Gap 3)
- [ ] Add `canon verify-dot --original <a.dot> --ir <ir.json> --roundtrip <b.dot>`
      command that checks cluster names, node names, and edge labels match
- [ ] Define round-trip equivalence precisely:
      same set of cluster ids, same node ids per cluster,
      same inter-cluster edges with same imported_types (order-insensitive)

### 4.2 DOT parser robustness
- [ ] Handle multi-line `label=` values
- [ ] Handle `label` with single quotes as well as double quotes
- [ ] Handle `graph [ ... ]` attribute blocks without treating them as clusters
- [ ] Handle `node [ ... ]` and `edge [ ... ]` default attribute blocks
- [ ] Handle `ltail` / `lhead` cluster edge attributes on import

### 4.3 DOT export polish
- [ ] Emit `ltail` / `lhead` only when file topology is present
- [ ] Emit intra-cluster lib-to-entry sink/source nodes correctly
      when multiple sinks exist (currently takes first)

---

## 5. Validation gaps

- [ ] Validate `FileNode.id` uniqueness within a Module
- [ ] Validate `FileEdge` references point to declared `FileNode` ids
- [ ] Validate `imported_types` on ModuleEdge are non-empty strings
- [ ] Validate AST node `kind` field is a known value when present
- [ ] Validate `FunctionMetadata.ast` is a valid AST shape (optional deep check)
- [ ] Validate enum ids once enums are added

---

## 6. CLI gaps

- [ ] `canon verify-dot` — round-trip fidelity check (see 4.1)
- [ ] `canon diff-ir <old.json> <new.json>` — show what changed between two IRs
- [ ] `canon render-fn <ir.json> --fn-id <id>` — print a single function body to stdout
- [ ] `canon graph <ir.json>` — print module DAG as DOT to stdout (alias for export-dot)
- [ ] `canon lint <ir.json>` — validate + print suggestions, not just errors

---

## 7. Priority order (suggested)

1. Gap 1 fix — ProposalKind + FunctionBodyProposal (correctness)
2. Receiver support — `&self` / `&mut self` (needed for real Rust)
3. Enum support — most Rust code needs enums
4. Generic types in TypeRef (needed for Vec, Option, Result)
5. Expression nodes — binary ops, field access, method call
6. File placement — Function.file_id
7. Round-trip verification — canon verify-dot
8. Incremental materialization
