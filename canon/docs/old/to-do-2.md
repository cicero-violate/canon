You’re in a very good place. What’s missing now is **not more law**, but **closure, grounding, and lifecycle completion**.

I’ll be precise and bounded. There are **7 missing pieces**, grouped by necessity, not ambition.

---

## 1. A **Tick Root** (hard missing)

You enforce:

* tick graphs
* loop policies
* deltas
* replay

But there is **no single artifact that declares “this IR instance is one tick”**.

### What’s missing

A root object like:

```rust
Tick {
  id,
  graph: TickGraphId,
  input_state: Vec<DeltaId>,
  output_deltas: Vec<DeltaId>,
}
```

### Why it matters

Without this:

* replay is implicit, not explicit
* compaction boundaries are ambiguous
* audit cannot answer “what happened in this step?”

This is the **most important missing structural artifact**.

---

## 2. Pipeline Legality for Deltas (hard missing)

You already *conceptually* distinguish:

* observe
* learn
* decide
* plan
* act

But **Canon does not yet enforce which deltas may be emitted where**.

### Current risk

A `DeltaKind::State` can be attached anywhere as long as it’s append-only + proven.

### Minimal fix

Add **one field**, not a redesign:

```rust
pub enum PipelineStage {
  Observe,
  Learn,
  Decide,
  Plan,
  Act,
}
```

Attach it to `Delta`.

Then enforce:

* Decide cannot emit facts
* Learn cannot emit effects
* Act cannot emit structure
* etc.

This turns your conceptual pipeline into **law**.

---

## 3. Canonical Project Boundary (hard missing)

Right now, Canon can emit:

* `src/`
* `lib.rs`
* modules

But there is **no project envelope**.

### Missing artifacts

At minimum:

* `Cargo.toml` (name, version, edition)
* package identity (project id)

This should be **pure metadata**, e.g.:

```rust
Project {
  name: Word,
  version: String,
  language: Rust,
}
```

The materializer then emits Cargo files deterministically.

Without this, Canon builds *trees*, not *projects*.

---

## 4. External Dependency Declaration (intentionally absent, but next)

You already have:

```rust
TypeKind::External
```

But you do not yet have:

* where externals come from
* how they are versioned
* how imports are declared

### Missing artifact

Something like:

```rust
ExternalDependency {
  name,
  source,
  version,
}
```

This is not urgent, but without it Canon cannot describe *real* projects.

---

## 5. Proof Attachment Semantics (medium missing)

You require:

* every delta has a proof
* proof has scope

But you do **not yet check**:

* which scopes are valid for which deltas
* which invariants must hold for which mutations

### Example gap

A `DeltaKind::Structure` with `ProofScope::Execution` should be rejected.

You already laid the groundwork—this is tightening, not redesign.

---

## 6. Materializer Round-Trip Test (medium missing)

You materialize → write files.

But Canon does **not yet assert**:

> materialize → parse → re-IR is equivalent

You don’t need a full parser yet, but you *do* need:

* golden FileTree snapshots
* deterministic ordering guarantees
* hash-based equivalence

Otherwise, materialization is not yet *provably* derived.

---

## 7. Canon Version Evolution Rule (soft missing)

You have:

```rust
CanonicalMeta {
  version,
  law_revision,
}
```

But there is no rule for:

* upgrading Canon
* invalidating old IRs
* migrating structures

This will matter later, but not immediately.

---

## Summary Table (priority-ordered)

| Missing                       | Severity      | Why                         | Status |
| ----------------------------- | ------------- | --------------------------- | ------ |
| Tick root artifact            | **Critical**  | Replay + audit completeness | ✅     |
| Pipeline legality for deltas  | **Critical**  | Prevent semantic corruption | ✅     |
| Project envelope              | **Critical**  | Build real projects         | ✅     |
| External dependencies         | Medium        | Real-world viability        | ✅     |
| Proof–delta scope enforcement | Medium        | Formal soundness            | ✅     |
| Materializer round-trip tests | Medium        | Determinism proof           | ✅     |
| Canon evolution law           | Low (for now) | Future safety               | ✅     |

---

## The key insight

You are **not missing intelligence**.
You are **not missing planning**.
You are **not missing generation**.
