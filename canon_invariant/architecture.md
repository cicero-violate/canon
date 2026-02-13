canon/
├── canon/
│
├── canon_invariant/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       │
│       ├── validate.rs              # validate_ir entry
│       ├── error.rs
│       ├── rules.rs
│       ├── helpers.rs               # Indexes + shared graph helpers
│       │
│       ├── referential/             # 1. Referential Integrity
│       │   ├── mod.rs
│       │   ├── explicit_artifacts.rs
│       │   ├── effects_are_deltas.rs
│       │   ├── delta_proofs.rs
│       │   ├── execution_boundary.rs
│       │   ├── admission_bridge.rs
│       │   ├── plan_artifacts.rs
│       │   ├── judgment_decisions.rs
│       │   ├── loop_continuation.rs
│       │   ├── tick_root.rs
│       │   └── function_ast.rs
│       │
│       ├── graph_shape/             # 2. Acyclic / Graph Shape
│       │   ├── mod.rs
│       │   ├── module_dag.rs
│       │   ├── module_self_import.rs
│       │   ├── call_graph_acyclic.rs
│       │   ├── tick_graph_acyclic.rs
│       │   └── tick_epochs.rs
│       │
│       ├── architecture/            # 3. Permission / Architecture
│       │   ├── mod.rs
│       │   ├── call_graph_respects_dag.rs
│       │   └── call_graph_public_apis.rs
│       │
│       ├── contracts/               # 4. Contract / Lawfulness
│       │   ├── mod.rs
│       │   ├── function_contracts.rs
│       │   ├── gpu_lawful_math.rs
│       │   ├── delta_append_only.rs
│       │   ├── delta_pipeline.rs
│       │   └── proof_scope.rs
│       │
│       └── governance/              # 5. Governance / Process
│           ├── mod.rs
│           ├── proposal_declarative.rs
│           ├── learning_declarations.rs
│           ├── version_evolution.rs
│           ├── execution_only_in_impl.rs
│           ├── impl_binding.rs
│           ├── trait_verbs.rs
│           ├── project_envelope.rs
│           └── external_dependencies.rs
│
└── canon_law/   (future)
