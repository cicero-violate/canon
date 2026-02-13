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
- ProposalKind enum (Structural / FunctionBody / SchemaEvolution) — auto_accept_fn_ast no longer needs fake scaffold
- Receiver support on Function and FunctionSignature (&self, &mut self, self, none)
- Enum support (EnumNode, EnumVariant, EnumVariantFields, AddEnum/AddEnumVariant deltas, render_enum, check_enums)
- Generic types in TypeRef (params: Vec<TypeRef>, ref_kind: Ref/MutRef/None, recursive render_type)
- Full expression node coverage in AST renderer: bin, cmp, logical, unary, field, index, method,
  struct_lit, tuple, array, ref, range, cast, question
- Full statement node coverage: for, loop, break, continue, assign, compound_assign, let mut

---

## 1. IR gaps — things Canon cannot yet express

### 1.1 Expression nodes missing from AST renderer
- [x] Binary operators: `a + b`, `a - b`, `a * b`, `a / b`, `a % b`
- [x] Comparison operators: `a == b`, `a != b`, `a < b`, `a > b`
- [x] Logical operators: `a && b`, `a || b`, `!a`
- [x] Field access: `self.field`, `a.b.c`
- [x] Index: `a[i]`
- [x] Method call: `a.method(args)`  — distinct from free `call`
- [x] Struct literal: `Foo { field: value }`
- [x] Tuple: `(a, b, c)`
- [x] Array literal: `[a, b, c]`
- [x] Reference / dereference: `&x`, `&mut x`
- [ ] Closure: `|args| body`
- [x] Range: `0..n`, `0..=n`
- [x] Cast: `x as u64`
- [x] Question mark operator: `expr?`

### 1.2 Statement nodes missing
- [x] `for` loop: `for x in iter { body }`
- [x] `loop` with `break`/`continue`
- [x] `break` with value
- [x] `continue`
- [x] Assignment: `x = value`
- [x] Compound assignment: `x += value`, `x -= value` etc.
- [x] `let mut` binding

### 1.3 Type system — TypeRef is too simple
- [x] Generic types: `Vec<T>`, `Option<T>`, `Result<T, E>`
- [x] Reference types: `&T`, `&mut T`
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
- [x] `self`, `&self`, `&mut self` receiver — Receiver enum on Function and FunctionSignature
- [ ] Default parameter values (via wrapper fns)
- [ ] `async fn` flag on Function
- [ ] `unsafe fn` flag on Function
- [ ] Generic parameters on functions: `fn foo<T: Trait>(x: T)`
- [ ] Where clauses: `where T: Debug`
- [ ] Variadic inputs (via slice type)
- [ ] Doc comments on functions

### 1.7 Enum support — entirely missing
- [x] `DeltaPayload::AddEnum` / `AddEnumVariant` variants in ir.rs
- [x] `EnumNode { id, name, module, variants: Vec<EnumVariant> }` in IR
- [x] `EnumVariant { name, fields: EnumVariantFields }` — unit / tuple / struct variants
- [x] `apply_structural_delta` arm for `AddEnum` / `AddEnumVariant`
- [x] `render_enum` in materialize/render_struct.rs
- [x] `check_enums` in `validate/check_artifacts.rs`

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
- [x] `ProposalKind` enum added: `Structural | FunctionBody | SchemaEvolution`
- [x] `enforce_proposal_ready` skips node/api/edge check for `FunctionBody` proposals
- [x] `auto_accept_fn_ast` uses `ProposalKind::FunctionBody`, fake scaffold removed

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
- [x] `canon verify-dot --ir <ir.json> --original <a.dot>` implemented
- [x] Round-trip equivalence: cluster ids, per-cluster node ids,
      inter-cluster edges with sorted imported_types (order-insensitive)

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

- [x] 1. ProposalKind + FunctionBodyProposal (correctness)
- [x] 2. Receiver support — `&self` / `&mut self`
- [x] 3. Enum support
- [x] 4. Generic types in TypeRef
- [x] 5. Expression nodes — binary ops, field access, method call, statements
- [x] 6. File placement — Function.file_id
- [x] 7. Round-trip verification — canon verify-dot
- [ ] 8. Incremental materialization

Next session — natural starting points:
Priority 8: Incremental materialization — hash-based re-emit, // canon:preserve regions (section 3.4)
Section 1.4/1.5: Struct and trait features — derive macros, pub(crate), associated types, default method bodies, supertraits
Section 1.6: Function features — async fn, unsafe fn, generics on functions, where clauses
Section 1.8: Module-level items — pub use, pub const, type aliases
Section 2.1: Delta pipeline gaps — UpdateFunctionInputs, UpdateFunctionOutputs, UpdateStructVisibility, RemoveField
Section 3.2: Import deduplication — use super::, use crate::, use ::external_crate::
