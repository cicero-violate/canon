use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};

use canon::auto_accept_dot_proposal;
use canon::auto_accept_fn_ast;
use canon::dot_export::export_dot;
use canon::verify_dot;
use canon::{
    CanonRule, CanonicalIr, ProposalAcceptanceInput, accept_proposal, auto_accept_dsl_proposal,
    execution_events_to_observe_deltas, generate_schema,
    ir::ExecutionRecord,
    materialize, render_impl_function,
    runtime::{TickExecutionMode, TickExecutor},
    validate_ir, write_file_tree,
};

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Schema { pretty } => {
            let schema = generate_schema(pretty)?;
            println!("{schema}");
        }
        Command::IngestStub => {
            todo!("CLI wiring after ING-001");
        }
        Command::Validate { path } => {
            let data = fs::read(&path)?;
            let ir: CanonicalIr = serde_json::from_slice(&data)?;
            validate_ir(&ir)?;
            enforce_version_gate(&ir)?;
            println!("Validation passed: `{}` satisfies Canon.", path.display());
        }
        Command::Lint { ir } => {
            let ir_doc = load_ir_from_path(&ir)?;
            match validate_ir(&ir_doc) {
                Ok(()) => {
                    println!("INFO  [C-LINT] `{}` passed lint.", ir.display());
                }
                Err(errs) => {
                    for violation in errs.violations() {
                        let severity = violation_severity(violation.rule());
                        println!(
                            "{}  [{}] {}",
                            severity,
                            violation.rule().code(),
                            violation.detail()
                        );
                    }
                    return Err("lint found violations".into());
                }
            }
        }
        Command::RenderFn { ir, fn_id } => {
            let ir_doc = load_ir_from_path(&ir)?;
            let function = ir_doc
                .functions
                .iter()
                .find(|f| f.id == fn_id)
                .ok_or_else(|| format!("unknown function: {fn_id}"))?;
            let rendered = render_impl_function(function);
            println!("{rendered}");
        }
        Command::DiffIr { before, after } => {
            let before_ir = load_ir_from_path(&before)?;
            let after_ir = load_ir_from_path(&after)?;
            let diff = diff_ir(&before_ir, &after_ir);
            if diff.trim().is_empty() {
                println!("No differences detected.");
            } else {
                println!("{diff}");
            }
        }
        Command::Materialize { ir, out_dir } => {
            let data = fs::read(&ir)?;
            let ir_doc: CanonicalIr = serde_json::from_slice(&data)?;
            validate_ir(&ir_doc)?;
            enforce_version_gate(&ir_doc)?;
            let mat = materialize(&ir_doc, Some(&out_dir));
            write_file_tree(&mat.tree, &out_dir)?;
            println!(
                "Materialized Canon from `{}` into `{}`.",
                ir.display(),
                out_dir.display()
            );
        }
        Command::ObserveEvents {
            execution,
            proof,
            output,
        } => {
            let data = fs::read(&execution)?;
            let execution_record: ExecutionRecord = serde_json::from_slice(&data)?;
            let deltas = execution_events_to_observe_deltas(&execution_record, &proof);
            let rendered = serde_json::to_string_pretty(&deltas)?;
            fs::write(&output, rendered)?;
            println!(
                "Generated {} observe deltas at `{}`.",
                deltas.len(),
                output.display()
            );
        }
        Command::Decide {
            ir,
            proposal,
            proof,
            predicate,
            judgment,
            admission,
            tick,
            rationale,
            output_ir,
            materialize_dir,
        } => {
            let data = fs::read(&ir)?;
            let ir_doc: CanonicalIr = serde_json::from_slice(&data)?;
            let input = ProposalAcceptanceInput {
                proposal_id: proposal.clone(),
                proof_id: proof.clone(),
                predicate_id: predicate.clone(),
                judgment_id: judgment.clone(),
                admission_id: admission.clone(),
                tick_id: tick.clone(),
                rationale: rationale.clone(),
            };
            let mut result = accept_proposal(&ir_doc, input)?;
            validate_ir(&result.ir)?;
            enforce_version_gate(&result.ir)?;
            let mat = materialize(&result.ir, Some(&materialize_dir));
            write_file_tree(&mat.tree, &materialize_dir)?;
            result.ir.file_hashes = mat.file_hashes;
            let rendered = serde_json::to_string_pretty(&result.ir)?;
            fs::write(&output_ir, rendered)?;
            println!(
                "Accepted proposal `{}` via judgment `{}` ({} deltas).",
                proposal,
                result.judgment_id,
                result.delta_ids.len()
            );
        }
        Command::SubmitDsl {
            dsl_file,
            ir,
            output_ir,
            materialize_dir,
        } => {
            let ir_bytes = fs::read(&ir)?;
            let ir_doc: CanonicalIr = serde_json::from_slice(&ir_bytes)?;
            let dsl_source = fs::read_to_string(&dsl_file)?;
            let mut acceptance = auto_accept_dsl_proposal(&ir_doc, &dsl_source)?;
            validate_ir(&acceptance.ir)?;
            enforce_version_gate(&acceptance.ir)?;
            let mat = materialize(&acceptance.ir, Some(&materialize_dir));
            write_file_tree(&mat.tree, &materialize_dir)?;
            acceptance.ir.file_hashes = mat.file_hashes;
            let rendered = serde_json::to_string_pretty(&acceptance.ir)?;
            fs::write(&output_ir, rendered)?;
            println!(
                "DSL `{}` auto-accepted via `{}` ({} deltas).",
                dsl_file.display(),
                acceptance.judgment_id,
                acceptance.delta_ids.len()
            );
        }
        Command::ExecuteTick { ir, tick, parallel } => {
            let data = fs::read(&ir)?;
            let ir_doc: CanonicalIr = serde_json::from_slice(&data)?;
            validate_ir(&ir_doc)?;
            enforce_version_gate(&ir_doc)?;
            let executor = TickExecutor::new(&ir_doc);
            let mode = if parallel {
                TickExecutionMode::ParallelVerified
            } else {
                TickExecutionMode::Sequential
            };
            let result = executor.execute_tick_with_mode(&tick, mode)?;
            let seq_us = result.sequential_duration.as_micros();
            let parallel_msg = result
                .parallel_duration
                .map(|d| format!(", parallel={}µs", d.as_micros()))
                .unwrap_or_default();
            println!(
                "Executed tick `{}` (sequential={}µs{}) emitting {} deltas.",
                tick,
                seq_us,
                parallel_msg,
                result.emitted_deltas.len()
            );
        }
        Command::ImportDot {
            dot_file,
            ir,
            goal,
            output_ir,
            materialize_dir,
        } => {
            let ir_bytes = fs::read(&ir)?;
            let ir_doc: CanonicalIr = serde_json::from_slice(&ir_bytes)?;
            let dot_source = fs::read_to_string(&dot_file)?;
            let mut acceptance = auto_accept_dot_proposal(&ir_doc, &dot_source, &goal)?;
            validate_ir(&acceptance.ir)?;
            enforce_version_gate(&acceptance.ir)?;
            let mat = materialize(&acceptance.ir, Some(&materialize_dir));
            write_file_tree(&mat.tree, &materialize_dir)?;
            acceptance.ir.file_hashes = mat.file_hashes;
            let rendered = serde_json::to_string_pretty(&acceptance.ir)?;
            fs::write(&output_ir, rendered)?;
            println!(
                "DOT `{}` imported via judgment `{}` ({} deltas).",
                dot_file.display(),
                acceptance.judgment_id,
                acceptance.delta_ids.len()
            );
        }
        Command::ExportDot { ir, output } => {
            let data = fs::read(&ir)?;
            let ir_doc: CanonicalIr = serde_json::from_slice(&data)?;
            validate_ir(&ir_doc)?;
            enforce_version_gate(&ir_doc)?;
            let dot = export_dot(&ir_doc);
            fs::write(&output, &dot)?;
            println!("Exported DOT to `{}`.", output.display());
        }
        Command::VerifyDot { ir, original } => {
            let data = fs::read(&ir)?;
            let ir_doc: CanonicalIr = serde_json::from_slice(&data)?;
            validate_ir(&ir_doc)?;
            enforce_version_gate(&ir_doc)?;
            let dot_src = fs::read_to_string(&original)?;
            verify_dot(&ir_doc, &dot_src)
                .map_err(|e| format!("DOT round-trip verification failed:\n{e}"))?;
            println!(
                "DOT round-trip verified: `{}` matches IR `{}`.",
                original.display(),
                ir.display()
            );
        }
        Command::SubmitFnAst {
            ir,
            function_id,
            ast_file,
            output_ir,
            materialize_dir,
        } => {
            let ir_bytes = fs::read(&ir)?;
            let ir_doc: CanonicalIr = serde_json::from_slice(&ir_bytes)?;
            let ast_bytes = fs::read(&ast_file)?;
            let ast: serde_json::Value = serde_json::from_slice(&ast_bytes)?;
            let mut acceptance = auto_accept_fn_ast(&ir_doc, &function_id, ast)?;
            validate_ir(&acceptance.ir)?;
            enforce_version_gate(&acceptance.ir)?;
            let mat = materialize(&acceptance.ir, Some(&materialize_dir));
            write_file_tree(&mat.tree, &materialize_dir)?;
            acceptance.ir.file_hashes = mat.file_hashes;
            let rendered = serde_json::to_string_pretty(&acceptance.ir)?;
            fs::write(&output_ir, rendered)?;
            println!(
                "AST for `{}` accepted via judgment `{}` ({} deltas).",
                function_id,
                acceptance.judgment_id,
                acceptance.delta_ids.len()
            );
        }
    }

    Ok(())
}

#[derive(Parser)]
#[command(name = "canon", about = "Canonical IR schema + validator", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print the canonical IR JSON schema to stdout.
    Schema {
        /// Pretty-print the JSON schema.
        #[arg(long)]
        pretty: bool,
    },
    /// Validate a canonical IR document against Canon.
    Validate {
        /// Path to the canonical IR JSON document.
        path: PathBuf,
    },
    /// Placeholder for future ingest CLI (hidden).
    #[command(hide = true)]
    IngestStub,
    /// Lint a canonical IR document and print violations with severity tags.
    Lint {
        /// Path to the canonical IR JSON document.
        ir: PathBuf,
    },
    /// Render a function implementation to stdout.
    RenderFn {
        /// Path to the canonical IR JSON document.
        #[arg(long)]
        ir: PathBuf,
        /// Identifier of the function to render.
        #[arg(long = "fn-id")]
        fn_id: String,
    },
    /// Show set differences between two canonical IR documents.
    DiffIr {
        /// Path to the "before" canonical IR JSON document.
        #[arg(long)]
        before: PathBuf,
        /// Path to the "after" canonical IR JSON document.
        #[arg(long)]
        after: PathBuf,
    },
    /// Materialize a validated Canonical IR into a source tree.
    Materialize {
        /// Path to the canonical IR JSON document.
        ir: PathBuf,
        /// Output directory for the emitted FileTree under `src/`.
        out_dir: PathBuf,
    },
    /// Convert execution events into observe-stage deltas.
    ObserveEvents {
        /// Path to an ExecutionRecord JSON file.
        execution: PathBuf,
        /// Proof id to assign to generated deltas.
        proof: String,
        /// Output path for the emitted deltas JSON.
        output: PathBuf,
    },
    /// Accept a submitted proposal and emit updated Canon plus materialized sources.
    Decide {
        /// Path to the canonical IR JSON document.
        ir: PathBuf,
        /// Identifier of the proposal to accept.
        #[arg(long)]
        proposal: String,
        /// Proof identifier backing the emitted deltas.
        #[arg(long)]
        proof: String,
        /// Predicate identifier for the emitted judgment.
        #[arg(long)]
        predicate: String,
        /// Judgment identifier to record.
        #[arg(long)]
        judgment: String,
        /// Delta admission identifier.
        #[arg(long)]
        admission: String,
        /// Tick identifier that owns the admission.
        #[arg(long)]
        tick: String,
        /// Rationale for the judgment decision.
        #[arg(long)]
        rationale: String,
        /// Output path for the updated Canonical IR document.
        #[arg(long)]
        output_ir: PathBuf,
        /// Directory for materialized sources.
        #[arg(long)]
        materialize_dir: PathBuf,
    },
    /// Parse a DSL file, auto-accept it, and emit evolved Canon + materialized sources.
    SubmitDsl {
        /// Path to the DSL `.canon` file.
        dsl_file: PathBuf,
        /// Path to the canonical IR JSON document.
        #[arg(long)]
        ir: PathBuf,
        /// Output path for the updated Canonical IR document.
        #[arg(long)]
        output_ir: PathBuf,
        /// Directory for materialized sources.
        #[arg(long)]
        materialize_dir: PathBuf,
    },
    /// Execute a tick graph and optionally verify parallel execution.
    ExecuteTick {
        /// Path to the canonical IR JSON document.
        #[arg(long)]
        ir: PathBuf,
        /// Tick identifier to execute.
        #[arg(long)]
        tick: String,
        /// Enable experimental parallel verification.
        #[arg(long)]
        parallel: bool,
    },
    /// Import a Graphviz DOT file and evolve Canon with its module structure.
    ImportDot {
        /// Path to the `.dot` file to import.
        dot_file: PathBuf,
        /// Path to the base canonical IR JSON document.
        #[arg(long)]
        ir: PathBuf,
        /// Short goal label for the generated proposal.
        #[arg(long)]
        goal: String,
        /// Output path for the updated Canonical IR document.
        #[arg(long)]
        output_ir: PathBuf,
        /// Directory for materialized sources.
        #[arg(long)]
        materialize_dir: PathBuf,
    },
    /// Export a Canonical IR document as a Graphviz DOT file.
    ExportDot {
        /// Path to the canonical IR JSON document.
        ir: PathBuf,
        /// Output path for the emitted `.dot` file.
        output: PathBuf,
    },
    /// Verify that a DOT file round-trips faithfully through the IR.
    VerifyDot {
        /// Path to the canonical IR JSON document.
        ir: PathBuf,
        /// Path to the original `.dot` file to compare against.
        #[arg(long)]
        original: PathBuf,
    },
    /// Submit a JSON AST file as the body of an existing function.
    SubmitFnAst {
        /// Path to the canonical IR JSON document.
        ir: PathBuf,
        /// Function identifier to update.
        #[arg(long)]
        function_id: String,
        /// Path to the JSON file containing the AST node tree.
        #[arg(long)]
        ast_file: PathBuf,
        /// Output path for the updated Canonical IR document.
        #[arg(long)]
        output_ir: PathBuf,
        /// Directory for materialized sources.
        #[arg(long)]
        materialize_dir: PathBuf,
    },
}

fn enforce_version_gate(ir: &CanonicalIr) -> Result<(), Box<dyn std::error::Error>> {
    let runtime_version = env!("CARGO_PKG_VERSION");
    if ir.version_contract.current == runtime_version
        || ir
            .version_contract
            .compatible_with
            .iter()
            .any(|v| v == runtime_version)
    {
        Ok(())
    } else {
        Err(format!(
            "Canon version `{}` is incompatible with runtime `{}`",
            ir.version_contract.current, runtime_version
        )
        .into())
    }
}

fn load_ir_from_path(path: &Path) -> Result<CanonicalIr, Box<dyn std::error::Error>> {
    let data = fs::read(path)?;
    let ir = serde_json::from_slice(&data)?;
    Ok(ir)
}

fn diff_ir(before: &CanonicalIr, after: &CanonicalIr) -> String {
    let mut lines = Vec::new();
    diff_section(
        "module",
        before.modules.iter().map(|m| m.id.as_str().to_owned()),
        after.modules.iter().map(|m| m.id.as_str().to_owned()),
        &mut lines,
    );
    diff_section(
        "struct",
        before.structs.iter().map(|s| s.id.as_str().to_owned()),
        after.structs.iter().map(|s| s.id.as_str().to_owned()),
        &mut lines,
    );
    diff_section(
        "enum",
        before.enums.iter().map(|e| e.id.as_str().to_owned()),
        after.enums.iter().map(|e| e.id.as_str().to_owned()),
        &mut lines,
    );
    diff_section(
        "trait",
        before.traits.iter().map(|t| t.id.as_str().to_owned()),
        after.traits.iter().map(|t| t.id.as_str().to_owned()),
        &mut lines,
    );
    diff_section(
        "function",
        before.functions.iter().map(|f| f.id.as_str().to_owned()),
        after.functions.iter().map(|f| f.id.as_str().to_owned()),
        &mut lines,
    );
    diff_section(
        "delta",
        before.deltas.iter().map(|d| d.id.as_str().to_owned()),
        after.deltas.iter().map(|d| d.id.as_str().to_owned()),
        &mut lines,
    );
    lines.join("\n")
}

fn diff_section(
    label: &str,
    before: impl Iterator<Item = String>,
    after: impl Iterator<Item = String>,
    lines: &mut Vec<String>,
) {
    let before_set: BTreeSet<String> = before.collect();
    let after_set: BTreeSet<String> = after.collect();
    for item in before_set.difference(&after_set) {
        lines.push(format!("- {:<8} {}", label, item));
    }
    for item in after_set.difference(&before_set) {
        lines.push(format!("+ {:<8} {}", label, item));
    }
}

fn violation_severity(rule: CanonRule) -> &'static str {
    match rule {
        CanonRule::ModuleDag
        | CanonRule::ModuleSelfImport
        | CanonRule::CallGraphPublicApis
        | CanonRule::CallGraphRespectsDag
        | CanonRule::CallGraphAcyclic
        | CanonRule::TickGraphAcyclic
        | CanonRule::TickGraphEdgesDeclared
        | CanonRule::TickRoot
        | CanonRule::DeltaProofs
        | CanonRule::ProofScope
        | CanonRule::VersionEvolution
        | CanonRule::DeltaPipeline
        | CanonRule::DeltaAppendOnly
        | CanonRule::ExecutionBoundary
        | CanonRule::AdmissionBridge => "ERROR",
        CanonRule::FunctionContracts
        | CanonRule::ProposalDeclarative
        | CanonRule::LearningDeclarations
        | CanonRule::TickEpochs
        | CanonRule::PlanArtifacts => "WARN",
        _ => "INFO",
    }
}
