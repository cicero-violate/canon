Define a Proposal DSL whose sole purpose is to express intent, not history.
The DSL MUST NOT contain ticks, epochs, executions, admissions, ordering, or applied deltas.
Each DSL file represents exactly one proposal.
A proposal consists of: id, goal, and an ordered list of changes.
changes is the only mutable surface of the DSL.
Supported change primitives are limited to: add_module, add_trait, add_function, add_edge, modify_struct, delete_node.
Each change must be self-contained and reference existing IDs or symbolic names only.
No change may reference a delta, admission, judgment, or proof.
The DSL must be order-independent; sequencing is inferred later.
The DSL must allow symbolic references without predeclared IDs.
IDs are assigned only during lowering, never in the DSL.
The DSL must be ≤200 lines for large proposals and ≤20 lines for small ones.
The DSL must support YAML or Lisp-like syntax, not JSON.
The DSL must be trivially diffable by humans.
A lowering pass converts DSL → Canon IR deltas.
The lowering pass is the only place that emits deltas.
Canon validates and judges lowered output, never the DSL.
Rejected proposals do not modify Canon IR.
Accepted proposals append deltas to Canon IR immutably.
Humans and LLMs only ever write the DSL, never Canon IR.
