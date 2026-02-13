## Self-Replication Capability Analysis

### Variables

$$
\text{Let } \mathcal{S} = \text{set of capabilities required for self-replication}
$$

$$
\text{Let } \mathcal{C}_{\text{have}} = \text{capabilities Canon currently has}
$$

$$
\text{Let } \mathcal{C}_{\text{need}} = \mathcal{S} \setminus \mathcal{C}_{\text{have}}
$$

$$
\text{Let } \mathcal{R} = \text{self-replication} \iff \text{Canon}(ir_{\text{self}}) \to \text{Canon}_{\text{compiled}}
$$

$$
\text{Let } \delta_{\text{gap}} = |\mathcal{C}_{\text{need}}| / |\mathcal{S}|, \quad \delta_{\text{gap}} \in [0, 1]
$$

## Latent Equations

$$
\mathcal{R} \iff \text{materialize}(ir_{\text{self}}) \xrightarrow{\texttt{rustc}} \text{binary} \equiv \text{Canon}
$$

$$
ir_{\text{self}} = \text{parse}(\texttt{src/**/*.rs}) \quad \text{(bootstrap condition)}
$$

$$
\text{Fidelity} = \frac{|\text{symbols in } ir_{\text{self}}}|}{|\text{symbols in Canon source}|}
$$

---

## What Self-Replication Means Precisely

Canon self-replicates when:

$$
\texttt{canon materialize --ir canon\_self.json --out ./out} \xrightarrow{\texttt{cargo build ./out}} \texttt{canon\_v2} \equiv \texttt{canon\_v1}
$$

That is: Canon's own source code is fully representable in its own IR, and materializing that IR produces Rust source that compiles to a functionally identical binary.

---

## Gap Analysis

### Already Have

$$
\mathcal{C}_{\text{have}} = \{$$
$$\text{modules, structs, enums, traits, impl blocks, functions,}$$
$$\text{generics, where clauses, async/unsafe fn, derives, doc comments,}$$
$$\text{pub use / const / type alias / statics, attributes,}$$
$$\text{full expression + statement AST renderer,}$$
$$\text{incremental emit, use-import deduplication}$$
$$\}$$

These cover roughly **60–65%** of Canon's own source.

---

### Gap 1 — Source Ingestion (Most Critical)

$$
\mathcal{C}_{\text{need}}^{(1)} : \texttt{src/**/*.rs} \xrightarrow{\text{parse}} ir_{\text{self}}
$$

Canon has no Rust parser. There is no path from existing `.rs` files to `CanonicalIr`. Currently the IR is authored by hand or via proposal acceptance. To self-replicate, Canon needs a **source-to-IR ingestor**:

- Parse each `.rs` file (via `syn` crate or `rust-analyzer` library)
- Extract every struct, enum, trait, impl, function, use, const, static, type alias, module declaration
- Translate `syn` AST nodes into `CanonicalIr` types
- Assign file IDs, module IDs, function IDs deterministically
- Reconstruct call edges from function body call expressions
- Reconstruct module edges from `use` declarations

This is the single largest missing piece. Without it, `ir_self` must be hand-authored, which is not self-replication.

**Estimated effort:** Large. This is a new `ingest/` module, likely 1000–1500 lines, depending on how much of `syn`'s type system is mapped.

---

### Gap 2 — Lifetime Annotations

Canon's own source uses lifetimes extensively:

```rust
fn render_file<'a>(module: &'a Module, ...) -> ...
fn build_indexes<'a>(ir: &'a CanonicalIr, ...) -> Indexes<'a>
fn module_has_permission<'a>(...) -> bool
```

$$
\mathcal{C}_{\text{need}}^{(2)} : \texttt{TypeRef.lifetime} \text{ on arbitrary positions} + \text{named lifetime params on functions}
$$

Currently `lifetime: Option<String>` exists on `TypeRef` but there is no `lifetime_params: Vec<String>` on `Function` or `FunctionSignature`. Canon's own functions that use `'a`, `'b` named lifetimes in multiple positions cannot be represented without this.

**Required additions:**
- `lifetime_params: Vec<String>` on `Function` and `FunctionSignature`
- `render_generics` must interleave lifetime params before type params: `<'a, 'b, T: Trait>`
- `TypeRef` lifetime field already exists; just needs the function-level declaration

---

### Gap 3 — `Self` Type

Canon's own impl blocks use `Self` as a return type in many constructors:

```rust
fn new(...) -> Self
fn default() -> Self
```

$$
\mathcal{C}_{\text{need}}^{(3)} : \texttt{TypeKind::SelfType} \to \text{render as } \texttt{"Self"}
$$

This is a one-line IR addition and a one-line render addition, but without it many of Canon's own `impl` functions produce incorrect signatures.

---

### Gap 4 — Trait Objects and `impl Trait`

Canon's own source uses both heavily:

```rust
fn iter(&self) -> impl Iterator<Item = &VerifiedPatch>
fn run() -> Result<(), Box<dyn std::error::Error>>
```

$$
\mathcal{C}_{\text{need}}^{(4)} : \texttt{TypeKind::ImplTrait},\; \texttt{TypeKind::DynTrait}
$$

**Required:**
- `ImplTrait { bounds: Vec<String> }` variant in `TypeKind`
- `DynTrait { bounds: Vec<String> }` variant in `TypeKind`
- `render_type` arms: `impl Trait + OtherTrait`, `dyn Trait + OtherTrait`
- Associated type bindings inside bounds: `Iterator<Item = &T>` requires `bound_params: Vec<(String, TypeRef)>` on the trait bound representation

---

### Gap 5 — Macro Invocations as Statements and Expressions

Canon's own source calls macros in function bodies:

```rust
eprintln!("...");
format!("...");
vec![...];
panic!("...");
writeln!(f, "...")?;
```

$$
\mathcal{C}_{\text{need}}^{(5)} : \texttt{AST node kind "macro\_call"}
$$

**Required additions to AST renderer:**
- `render_macro_call(node) -> String` producing `<name>!(<args>)` or `<name>![<args>]` or `<name>!{<args>}`
- `args` is an opaque string (verbatim) since macro argument structure is not typed in the IR
- Statement form: `macro_stmt` node kind
- Expression form: `macro_expr` node kind (used when result is consumed)

Without this, any function body that calls `format!`, `vec!`, `println!`, `eprintln!`, `panic!`, `writeln!` cannot be represented. Canon's own source uses these in dozens of places.

---

### Gap 6 — Closures as Function Arguments (Capture and Higher-Order)

Canon's own source passes closures to higher-order functions:

```rust
.map(|f| f.id.as_str())
.filter(|v| !v.is_empty())
.find(|f| f.id == fn_id)
items.iter().map(|x| render_type(x)).collect::<Vec<_>>()
```

$$
\mathcal{C}_{\text{need}}^{(6)} : \text{closure with typed params + } \texttt{collect::<Vec<\_>>()} \text{ turbofish}
$$

The closure renderer added this session handles the basic case. The gap is:
- Typed closure params: `|x: &str| ...`
- Turbofish on method calls: `.collect::<Vec<_>>()`
- These require extending the `method` AST node to carry an optional `turbofish: Vec<TypeRef>` field

---

### Gap 7 — Pattern Matching in `let` and `match` Arms

Canon's own source uses destructuring patterns everywhere:

```rust
let Some(x) = opt else { return; };
let (a, b) = pair;
if let Ok(val) = result { ... }
while let Some(item) = iter.next() { ... }
match value {
    DeltaPayload::AddModule { module_id } => ...
    Ok(x) if x > 0 => ...
}
```

$$
\mathcal{C}_{\text{need}}^{(7)} : \text{pattern nodes in let/match/if-let/while-let AST}
$$

Currently `let` nodes carry only a name string. Match arms carry only a string pattern. For self-replication, patterns need to be structured:
- Tuple destructure: `(a, b)`
- Struct destructure: `Foo { field }`
- Enum variant: `Some(x)`, `Ok(v)`, `Err(e)`
- Guard clauses: `x if x > 0`
- `let-else`: `let Pat = expr else { ... }`
- `@` bindings: `x @ 0..=9`

---

### Gap 8 — `use` Declaration Rendering from Function Scope

Canon's own source uses `use` inside function bodies for local disambiguation. More critically, the ingestion path must correctly reconstruct which items each file imports from other modules. The current `collect_incoming_types` approach is heuristic. A fully faithful ingestor needs to track `use` declarations at file scope exactly as written.

$$
\mathcal{C}_{\text{need}}^{(8)} : \text{exact } \texttt{use} \text{ declaration preservation in Module.pub\_uses + file-scoped uses}
$$

---

### Gap 9 — `const` and `static` in `impl` Blocks

Canon's own source and many Rust crates define associated constants in impl blocks:

```rust
impl Foo {
    const MAX: usize = 100;
}
```

$$
\mathcal{C}_{\text{need}}^{(9)} : \texttt{ImplBlock.constants: Vec<ConstItem>}
$$

---

### Gap 10 — File Placement Materialization (3.1, still open)

Canon's own source has multiple files per module (e.g. `validate/` has `mod.rs`, `check_artifacts.rs`, `check_deltas.rs`, etc.). Currently all functions emit to `mod.rs`. Without file-placement routing on `Function.file_id`, the materialized output collapses all module content into one file, which does not match the original layout.

$$
\mathcal{C}_{\text{need}}^{(10)} : \text{route } f \text{ to } \texttt{FileNode}(f.\texttt{file\_id}) \text{ during materialize}
$$

---

## Summary Gap Table

| Gap | Severity for Self-Rep | Effort |
|---|---|---|
| 1. Source ingestor (`syn`-based) | **Blocking** | Large |
| 2. Lifetime params on functions | High | Small |
| 3. `Self` type | High | Tiny |
| 4. `impl Trait` / `dyn Trait` | High | Small |
| 5. Macro invocations in AST | **Blocking** | Medium |
| 6. Typed closure params + turbofish | Medium | Small |
| 7. Structured patterns in let/match | High | Medium |
| 8. Exact use-declaration preservation | Medium | Medium |
| 9. Associated constants in impl blocks | Low | Tiny |
| 10. File placement routing | High | Small |

---

## Minimum Viable Self-Replication Path

$$
\text{MVP}_{\mathcal{R}} = \text{Gap 1} + \text{Gap 2} + \text{Gap 3} + \text{Gap 4} + \text{Gap 5} + \text{Gap 7} + \text{Gap 10}
$$

These seven gaps are the **necessary and sufficient** set. With them closed:

1. `canon ingest src/` reads Canon's own Rust source into `canon_self.json`
2. `canon validate --ir canon_self.json` passes all 31 rules
3. `canon materialize --ir canon_self.json --out ./out` emits a full Rust project
4. `cargo build ./out` produces a working binary
5. The binary passes the same ingest → validate → materialize cycle on its own output

Gaps 6, 8, and 9 are needed for full fidelity but not for a compilable replica — the ingested IR can use opaque string fallbacks for those constructs in the first iteration.

---

## English Summary

Canon is approximately **60–65% of the way to self-replication**. The render pipeline, delta evolution engine, validation system, and CLI are all mature enough. The critical missing piece is **source ingestion**: Canon cannot yet read its own `.rs` files and produce an IR from them. Everything flows from that. Once a `syn`-based ingestor exists, the remaining gaps are targeted additions — lifetime params, `Self` type, `impl/dyn Trait`, macro call AST nodes, structured patterns, and file-placement routing — each of which is a bounded, well-defined task. The full self-replication path represents roughly three to four additional coding-agent sessions at the current pace.
