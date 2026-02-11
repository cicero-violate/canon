## **Canonical Build Instructions (100 Lines)**

### **Build Progress Legend**
- `[check]` → done
- `(circle)` → in progress

### **Current Work Status**
- `[check]` Canonical IR schema + validator CLI implemented.
- `[check]` Sample canonical IR fixture + golden tests landed.
- `[check]` Canon materializer emits FileTree stubs.
- `[check]` Materializer CLI subcommand writes FileTree to disk.
- `[check]` Tick roots, pipeline legality, project envelopes, and version contracts are enforced.
- `(circle)` Pipeline integration + automation for the Canon validator.

1. Implement a law-governed structural intelligence engine.
2. Treat this document as binding Canon.
3. Do not infer intent beyond what is written.
4. Do not add features not explicitly allowed.
5. All artifacts must be explicit and enumerable.
6. Represent the system using a canonical IR.
7. The IR is the source of truth.
8. DOT or visuals are derived views only.
9. Never write execution logic in views.
10. Never infer structure from code text.
11. Every module is one word.
12. Every module is exactly one node.
13. Modules form a strict acyclic DAG.
14. No module may import itself.
15. No module may form a cycle.
16. Module edges represent import permission only.
17. No edge implies no permission.
18. Visibility is explicit or forbidden.
19. Public APIs must be declared.
20. Private symbols are inaccessible.
21. Structs are one-word nouns.
22. Structs contain state only.
23. Traits are one-word verbs.
24. Traits declare capability only.
25. Traits contain no execution.
26. impl blocks bind noun → verb.
27. impl blocks contain execution only.
28. No execution exists outside impl.
29. No control flow exists inside impl.
30. Only composition of calls is allowed.
31. All functions are total.
32. All inputs are explicit.
33. All outputs are explicit.
34. All effects are returned as deltas.
35. Hidden mutation is forbidden.
36. Implicit IO is forbidden.
37. All IO must be modeled as deltas.
38. All deltas are append-only.
39. History is immutable.
40. Replay must be deterministic.
41. Call graphs are explicit artifacts.
42. Call graph edges reference public APIs only.
43. Call graphs must respect module DAGs.
44. No dynamic dispatch beyond declared traits.
45. No reflection or runtime discovery.
46. The IR represents a single tick.
47. Each tick is finite.
48. Each tick graph is acyclic.
49. No loops exist inside a tick.
50. The only permitted cycle is time.
51. Loops are represented as re-application of the same DAG.
52. Loop continuation is a judgment decision.
53. Judgment is external to execution.
54. Judgment evaluates structure, not code.
55. Judgment may accept or reject proposals.
56. LLMs may propose structure only.
57. Proposals are top-down and declarative.
58. Proposals specify goals, nodes, APIs, edges.
59. Proposals never specify execution.
60. Execution occurs only after acceptance.
61. Learning may propose new rules or goals.
62. Learning has no authority to execute.
63. Proof systems validate invariants only.
64. Proof does not decide policy.
65. Judgment decides policy.
66. Self-rewrite is allowed only via IR deltas.
67. All self-rewrite requires proof obligations.
68. Proofs must be attached to deltas.
69. Unproven deltas are rejected.
70. Rollback must be possible.
71. Functions must be referentially transparent.
72. Side effects must be explicit outputs.
73. No branching on hidden state.
74. No early exit.
75. No recursion.
76. Data parallelism is first-class.
77. Any node without dependencies is parallelizable.
78. Batch execution must preserve semantics.
79. GPU execution must be optional and lawful.
80. CPU and GPU must share the same IR.
81. SIMD/GPU code may contain math only.
82. No allocation inside GPU kernels.
83. No IO inside GPU kernels.
84. No judgment inside GPU kernels.
85. No branching beyond arithmetic masks.
86. Kernel inputs are value arrays.
87. Kernel outputs are value arrays or deltas.
88. Kernels are pure functions.
89. Kernel correctness must match scalar execution.
90. Kernel fusion is permitted if semantics preserved.
91. Performance is subordinate to correctness.
92. Correctness is subordinate to law.
93. Law violations are hard errors.
94. Errors must be explicit artifacts.
95. Silence is forbidden.
96. The system must be inspectable.
97. The system must be replayable.
98. The system must be auditable.
99. The system must be provable.
100. The system must choose rightly, not merely act.

---

## **Example Canonical IR Schema (for the coding agent)**

### **Core Structural Tables**

```rust
// Module DAG
Module {
  id: ModuleId,
  name: Word,
}

// Import permission
ModuleEdge {
  from: ModuleId,
  to: ModuleId,
}

// State (noun)
Struct {
  id: StructId,
  name: Word,
  fields: Vec<Field>,
}

// Capability (verb)
Trait {
  id: TraitId,
  name: Word,
  signature: FnSig,
}

// Binding noun → verb
Impl {
  struct_id: StructId,
  trait_id: TraitId,
  functions: Vec<FunctionId>,
}
```

---

### **Execution + Call Graph**

```rust
Function {
  id: FunctionId,
  name: Word,
  inputs: Vec<Type>,
  outputs: Vec<Type>,   // includes Delta
}

CallEdge {
  caller: FunctionId,
  callee: FunctionId,   // must be public API
}

TickGraph {
  nodes: Vec<FunctionId>,
  edges: Vec<CallEdge>, // acyclic
}
```

---

### **Loop Representation (No Cycles in Graph)**

```rust
LoopPolicy {
  graph: TickGraphId,
  continuation: JudgmentPredicateId,
  max_ticks: Option<u64>,
}
```

---

## **GPU / SIMD-Only Math Functions (at the end)**

```rust
// Allowed ONLY for math kernels
GpuFunction {
  id: FunctionId,
  inputs: Vec<ScalarOrVector>,
  outputs: Vec<ScalarOrVector>,
  properties: {
    pure: true,
    no_io: true,
    no_alloc: true,
    no_branch: true,
  }
}

// Example
fn add_vec(a: f32xN, b: f32xN) -> f32xN;
fn mul_vec(a: f32xN, b: f32xN) -> f32xN;
fn dot_vec(a: f32xN, b: f32xN) -> f32;
fn affine(x: f32xN, w: f32xN, b: f32) -> f32xN;
```

These functions:

* contain **math only**
* are **lifted automatically** from scalar semantics
* are **swappable CPU ↔ GPU**

---

### **Final lock-in sentence**

> **Control lives in the graph.
> Truth lives in the IR.
> Math lives at the leaves.**

* Compress this into a **single JSON schema**
* Write the **validator that enforces Canon**
