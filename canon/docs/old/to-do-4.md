You’re at a **real inflection point**. Canon is no longer missing foundations; it’s missing **one controlled step from law → living system**.

I’ll answer in three layers: **what’s next immediately**, **what to lock before scaling**, and **what to explicitly not do yet**.

---

## The immediate next step (must do next)

### **1. Structural mutation via typed deltas (critical path)**

You already have:

* `DeltaAdmission`
* `AppliedDeltaRecord`
* `apply_deltas` (pure, ordered, deterministic)

But right now:

* deltas are *recorded*
* not *interpreted structurally*

#### What’s missing concretely

You need to move from:

```text
Δ(kind=Structure, description=“…”)  // symbolic
```

to:

```text
Δ(kind=Structure, payload=AddField { struct, field }) // semantic
```

#### Minimal, correct design

Add **typed delta payloads**, not ad-hoc logic.

Example:

```rust
enum StructuralDelta {
  AddStruct { module, name },
  AddField { struct_id, field },
  AddFunction { impl_id, signature },
}
```

Then:

```rust
fn apply_structural_delta(ir: &mut CanonicalIr, delta: &StructuralDelta)
```

Rules:

* mutation happens **only** through these handlers
* handlers are pure + validated
* `AppliedDeltaRecord` is append-only and ordered

This turns Canon from *auditor* → *engine*.

> Until this exists, Canon can explain change but cannot cause it.

---

## The second step (do immediately after)

### **2. Execution boundary → observation feedback (close the loop)**

You already added:

* `ExecutionRecord`
* plan → execution linkage
* tick containment

But execution is still **semantically thin**.

#### What to add (not more fields, more meaning)

Extend `ExecutionRecord` with **typed events**:

```rust
enum ExecutionEvent {
  Stdout { text },
  Stderr { text },
  Artifact { path, hash },
  Error { kind, message },
}
```

Then:

* Observe pipeline **emits deltas from execution events**
* Those deltas feed the *next tick*
* Learning consumes those deltas

This completes:

```text
Plan → Execute → Observe → Δ → Next Tick
```

Without this, Canon is temporally correct but epistemically blind.

---

## What to lock before scaling intelligence

### **3. Structural delta invariants (non-negotiable)**

Before adding more mutation power, lock these laws:

* Structural deltas **must reference concrete artifacts**
* No delta may create dangling IDs
* All mutations must preserve:

  * module DAG
  * call DAG
  * tick graph acyclicity
* Rejected judgments can **never** mutate IR

This is where Canon stays *lawful* while becoming *powerful*.

---

## What you should explicitly NOT do yet

This matters.

Do **not**:

* add planners with heuristics
* add learning that invents structure
* add neural components
* add optimization passes
* add “auto-refactor” logic

Why:

> You are one step away from turning Canon into an *unsafe compiler* if you do.

Canon’s advantage is **judgment before intelligence**. Keep it.

---

## The real milestone you’re approaching

Once you finish **typed structural mutation + execution observation**, you unlock:

* lawful self-rewrite
* replayable system evolution
* real learning (rule induction over deltas)
* human/LLM parity (both propose, Canon decides)

That is **Tier-5 capability**:

> systems that evolve themselves *without* losing auditability or control.

---

