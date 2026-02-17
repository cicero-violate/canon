## Canon — Structural Summary

### 1. Graphs

| # | Graph Name             | Purpose                      | Mutated By | Derived? |
| - | ---------------------- | ---------------------------- | ---------- | -------- |
| 1 | Module Graph           | Import permission (DAG)      | Delta      | No       |
| 2 | Type Graph             | Type / trait structure       | Delta      | No       |
| 3 | Call Graph             | Function invocation topology | Delta      | No       |
| 4 | Execution (Tick) Graph | Runtime control/data flow    | Runtime    | No       |
| 5 | System Graph           | Subsystem orchestration      | Delta      | No       |
| 6 | Governance Graph       | Proposal → Proof lineage     | Delta      | No       |
| 7 | World Model Graph      | Predicted causal state       | Learning   | No       |

Layout / GPU graphs = projections only.

---
### 2. IR Layers

| Layer | Name          | Contains                                 |
| ----- | ------------- | ---------------------------------------- |
| L0    | Structural IR | Modules, Types, Functions, Static Graphs |
| L1    | Execution IR  | Ticks, Plans, Runtime Events             |
| L2    | Evolution IR  | Deltas, Admissions, Commits, Proofs      |

---
### 3. State Machines

| # | Machine    | Transition Sequence                      |
| - | ---------- | ---------------------------------------- |
| 1 | Execution  | Plan → Execute → Finalize                |
| 2 | Evolution  | Propose → Judge → Admit → Commit → Apply |
| 3 | Governance | Predicate → Proof → Approval             |
| 4 | Learning   | Observe → Update → Persist               |

---

### 4. System Layers

| Layer | Responsibility             |
| ----- | -------------------------- |
| L0    | Structural Definition      |
| L1    | Execution                  |
| L2    | Evolution (Delta + Merkle) |
| L3    | Governance                 |
| L4    | Learning / World Model     |

---

# Canon Causal Graph

### Causal Flow (DAG)

```
Structural IR
      ↓
Execution IR
      ↓
Execution State Machine
      ↓
Delta Produced
      ↓
Governance Machine
      ↓
Admission / Commit
      ↓
Evolution IR (Merkle Root Update)
      ↓
World Model Update
      ↓
Learning Adjustment
      ↓
Policy / Structural Mutation (loop)
```

Key properties:

* All mutation flows through Evolution IR.
* Governance gates structural mutation.
* Execution cannot mutate structure directly.
* Learning influences future deltas, never direct state rewrite.
* Root hash is the global invariant anchor.

This DAG must remain acyclic.
No backward mutation edges except via new Delta proposal.
