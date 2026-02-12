use std::fs;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use canon::{
    CanonicalIr, ProposalAcceptanceInput, accept_proposal, auto_accept_dsl_proposal,
    execution_events_to_observe_deltas, generate_schema,
    ir::ExecutionRecord,
    materialize,
    runtime::{TickExecutionMode, TickExecutor},
    validate_ir, write_file_tree,
};
use canon::dot_export::export_dot;
use canon::auto_accept_dot_proposal;

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
        Command::Validate { path } => {
            let data = fs::read(&path)?;
            let ir: CanonicalIr = serde_json::from_slice(&data)?;
            validate_ir(&ir)?;
            enforce_version_gate(&ir)?;
            println!("Validation passed: `{}` satisfies Canon.", path.display());
        }
        Command::Materialize { ir, out_dir } => {
            let data = fs::read(&ir)?;
            let ir_doc: CanonicalIr = serde_json::from_slice(&data)?;
            validate_ir(&ir_doc)?;
            enforce_version_gate(&ir_doc)?;
            let tree = materialize(&ir_doc);
            write_file_tree(&tree, &out_dir)?;
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
            let result = accept_proposal(&ir_doc, input)?;
            validate_ir(&result.ir)?;
            enforce_version_gate(&result.ir)?;
            let rendered = serde_json::to_string_pretty(&result.ir)?;
            fs::write(&output_ir, rendered)?;
            let tree = materialize(&result.ir);
            write_file_tree(&tree, &materialize_dir)?;
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
            let acceptance = auto_accept_dsl_proposal(&ir_doc, &dsl_source)?;
            validate_ir(&acceptance.ir)?;
            enforce_version_gate(&acceptance.ir)?;
            let rendered = serde_json::to_string_pretty(&acceptance.ir)?;
            fs::write(&output_ir, rendered)?;
            let tree = materialize(&acceptance.ir);
            write_file_tree(&tree, &materialize_dir)?;
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
            let acceptance = auto_accept_dot_proposal(&ir_doc, &dot_source, &goal)?;
            validate_ir(&acceptance.ir)?;
            enforce_version_gate(&acceptance.ir)?;
            let rendered = serde_json::to_string_pretty(&acceptance.ir)?;
            fs::write(&output_ir, rendered)?;
            let tree = materialize(&acceptance.ir);
            write_file_tree(&tree, &materialize_dir)?;
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
