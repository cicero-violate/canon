DEBUG: Checking entry: "/workspace/ai_sandbox/canon_workspace/canon/txt_1_1.patch"
error: expected item, found `" Output is a serde_json::RuntimeValue suitable for feeding directly to the"`
 --> /workspace/ai_sandbox/canon_workspace/canon/src/agent/observe.rs:8:1
  |
8 | " Output is a serde_json::RuntimeValue suitable for feeding directly to the"
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected item
  |
  = note: for a full list of items that can appear in modules, see <https://doc.rust-lang.org/reference/items.html>

error: expected item, found `" IrDelta application verification (Canon Lines 68-69)."`
 --> /workspace/ai_sandbox/canon_workspace/canon/src/runtime/delta_verifier.rs:1:1
  |
1 | " IrDelta application verification (Canon Lines 68-69)."
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected item
  |
  = note: for a full list of items that can appear in modules, see <https://doc.rust-lang.org/reference/items.html>

Error writing files: failed to resolve mod `delta_verifier`: cannot parse /workspace/ai_sandbox/canon_workspace/canon/src/runtime/delta_verifier.rs
error: expected item, found `"   Observer(Observe) → Reasoner(Learn) → Prover(Decide) → Judge(ExecutionPlan) → Mutator(Act)"tor`
 --> /workspace/ai_sandbox/canon_workspace/canon/src/agent/bootstrap.rs:6:1
  |
6 | "   Observer(Observe) → Reasoner(Learn) → Prover(Decide) → Judge(ExecutionPlan) → Mutator(Act)"tor(Act)
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected item
  |
  = note: for a full list of items that can appear in modules, see <https://doc.rust-lang.org/reference/items.html>

error: expected item after doc comment
  --> /workspace/ai_sandbox/canon_workspace/canon/src/agent/slice.rs:48:1
   |
48 | /// Builds a JSON object containing only the IR fields listed in `fields`.
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this doc comment doesn't document anything

error: expected identifier, found `" IrProof confidence on this edge: 0.0 = unverified, 1.0 = fully proven."`
  --> /workspace/ai_sandbox/canon_workspace/canon/src/agent/capability.rs:83:5
   |
80 | pub struct CapabilityEdge {
   |            -------------- while parsing this struct
...
83 |     " IrProof confidence on this edge: 0.0 = unverified, 1.0 = fully proven."
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected identifier

error: expected item, found `" IrDelta application verification (Canon Lines 68-69)."`
 --> /workspace/ai_sandbox/canon_workspace/canon/src/runtime/delta_verifier.rs:1:1
  |
1 | " IrDelta application verification (Canon Lines 68-69)."
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected item
  |
  = note: for a full list of items that can appear in modules, see <https://doc.rust-lang.org/reference/items.html>

error: expected item, found `" IrProof scopes differentiate structural law, execution law, and meta-law invariants."`
 --> /workspace/ai_sandbox/canon_workspace/canon/kernel/src/lib.rs:6:1
  |
6 | " IrProof scopes differentiate structural law, execution law, and meta-law invariants."
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected item
  |
  = note: for a full list of items that can appear in modules, see <https://doc.rust-lang.org/reference/items.html>

error: expected item, found `" RuntimeValue representation for runtime execution."`
 --> /workspace/ai_sandbox/canon_workspace/canon/src/runtime/value.rs:1:1
  |
1 | " RuntimeValue representation for runtime execution."
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected item
  |
  = note: for a full list of items that can appear in modules, see <https://doc.rust-lang.org/reference/items.html>

error: expected identifier, found `" IrProof id generated during this call, if any."`
  --> /workspace/ai_sandbox/canon_workspace/canon/src/agent/call.rs:28:5
   |
23 | pub struct AgentCallOutput {
   |            --------------- while parsing this struct
...
28 |     " IrProof id generated during this call, if any."
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected identifier

error: expected identifier, found `" IrProof confidence on an incoming edge was below the required threshold."`
  --> /workspace/ai_sandbox/canon_workspace/canon/src/agent/call.rs:54:5
   |
45 | pub enum AgentCallError {
   |          -------------- while parsing this enum
...
54 |     " IrProof confidence on an incoming edge was below the required threshold."
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected identifier
   |
   = help: enum variants can be `Variant`, `Variant = <integer>`, `Variant(Type, ..., TypeN)` or `Variant { fields: Types }`

Warning: 9 rustfmt errors occurred:
  - rustfmt failed for /workspace/ai_sandbox/canon_workspace/canon/src/agent/observe.rs: exit code 1
  - rustfmt failed for /workspace/ai_sandbox/canon_workspace/canon/src/runtime/mod.rs: exit code 1
  - rustfmt failed for /workspace/ai_sandbox/canon_workspace/canon/src/agent/bootstrap.rs: exit code 1
  - rustfmt failed for /workspace/ai_sandbox/canon_workspace/canon/src/agent/slice.rs: exit code 1
  - rustfmt failed for /workspace/ai_sandbox/canon_workspace/canon/src/agent/capability.rs: exit code 1
  - rustfmt failed for /workspace/ai_sandbox/canon_workspace/canon/src/runtime/delta_verifier.rs: exit code 1
  - rustfmt failed for /workspace/ai_sandbox/canon_workspace/canon/kernel/src/lib.rs: exit code 1
  - rustfmt failed for /workspace/ai_sandbox/canon_workspace/canon/src/runtime/value.rs: exit code 1
  - rustfmt failed for /workspace/ai_sandbox/canon_workspace/canon/src/agent/call.rs: exit code 1
Note: 34 file(s) were modified but rustfmt encountered errors
8 files via structured edits (docs:8, attrs:8)
Renamed 136 occurrences across 27 files.
Scoped renames applied successfully.
