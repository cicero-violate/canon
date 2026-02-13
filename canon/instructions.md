# Next Session Instructions — `ingest` Module Bootstrap

Canon needs a Rust source ingestor per the self-replication plan. This session kicks off the module skeleton so follow-up agents can iterate on concrete extractors.

## Goals
1. Create a new crate module `ingest` under `canon/src/ingest/`.
2. Define the public API surface the CLI will eventually call (`ingest::parse_workspace(root: &Path) -> Result<LayoutMap>`).
3. Stub the internal components (filesystem walker, Rust parser adapter, IR builder) with TODO comments tied to DAG nodes from `TODO_self_replicate_DAG.md`.

## Tasks
1. **Module scaffolding**
   - Add `pub mod ingest;` to `lib.rs`.
   - Create `canon/src/ingest/mod.rs` with:
     - `pub struct IngestOptions { pub root: PathBuf }`
     - `pub fn ingest_workspace(opts: &IngestOptions) -> Result<CanonicalIr, IngestError>`
     - `enum IngestError` with placeholders (`Io`, `Parse`, `UnsupportedFeature`).
   - Include feature-gated `syn` dependency wiring comment (actual dep will be added later).

2. **Component stubs**
   - `fs_walk.rs`: walk `src/**/*.rs`, honor `.gitignore` later. Return list of files + module names.
   - `parser.rs`: wrap `syn` parsing; for now return `todo!()` and log which Gap it resolves.
   - `builder.rs`: convert parser output into `CanonicalIr`. Stub helpers for modules, structs, impls, functions, use edges.

3. **Documentation + TODO hooks**
   - In each stub, add `// TODO(ING-001): ...` referencing the DAG node.
   - Document expectations in module-level comments (e.g., deterministically assign IDs, capture attributes, etc.).

4. **Integration placeholder**
   - Add a hidden CLI subcommand `CanonCommand::IngestStub` that calls `todo!("CLI wiring after ING-001")` so future agents have an anchor.

## Acceptance
- All new files compile (even if functions `todo!()`).
- No existing behavior regressed (`cargo check` clean apart from pre-existing warnings).
- `TODO_self_replicate_DAG.md` remains the single source of truth (reference node IDs only, do not duplicate the full gap text).
