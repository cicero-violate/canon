| Graph           | Algorithm                |
| --------------- | ------------------------ |
| Name resolution | scope walk + binding map |
| Type inference  | union–find               |
| Borrow          | dataflow fixpoint        |
| Trait           | Horn resolution          |
| CFG dataflow    | monotone framework       |

NameSolver
TypeSolver
BorrowSolver
TraitSolver
CFGDataflowSolver
DependencySolver


| Priority | Solver Name                       | Category      | Purpose                                                                       | Required For            |
| -------- | --------------------------------- | ------------- | ----------------------------------------------------------------------------- | ----------------------- |
| **1**    | **Mutation Invariant Solver**     | Safety        | No dangling nodes, no invalid edges, valid Impl targets, acyclic module graph | Safe mutate → emit      |
| **2**    | **Visibility Solver**             | Semantic      | Enforce `pub` / private access rules across modules                           | Correct name resolution |
| **3**    | **Impl Resolution Solver**        | Semantic      | Ensure `impl` targets exist, enforce orphan rules, prevent duplicate impls    | Trait correctness       |
| **4**    | **Trait Obligation Solver**       | Semantic      | Verify trait bounds, method completeness, associated types                    | Generic correctness     |
| **5**    | **Generic Constraint Solver**     | Type System   | Propagate `TypeUnifies` constraints across call graph                         | Type soundness          |
| **6**    | **Name Provenance Solver**        | Rename Safety | Track symbol origin, shadowing, re-export chains                              | Stable refactoring      |
| **7**    | **Type Cycle Diagnostic Solver**  | Type System   | Emit structured diagnostics for SCC cycles                                    | Compiler-grade errors   |
| **8**    | **Call Graph Liveness Solver**    | Optimization  | Remove unreachable/dead functions from emit_order                             | Clean output            |
| **9**    | **Borrow & Lifetime Solver**      | Safety        | Region graph + borrow conflict detection                                      | Memory correctness      |
| **10**   | **Emission Stability Solver**     | Determinism   | Deterministic ordering + stable hashing                                       | Snapshot diff stability |
| **11**   | **Const Evaluation Solver**       | Execution     | Evaluate constant expressions in IR                                           | Const correctness       |
| **12**   | **Macro Expansion Solver**        | Expansion     | Expand macro item nodes before analyze                                        | Full Rust coverage      |
| **13**   | **Pattern Exhaustiveness Solver** | Safety        | Match exhaustiveness and unreachable arm detection                            | Enum correctness        |
| **14**   | **Drop Order Solver**             | Safety        | Ensure correct destruction ordering                                           | Ownership correctness   |
| **15**   | **Unsafe Soundness Solver**       | Safety        | Validate `unsafe` blocks obey invariants                                      | Sound IR                |
