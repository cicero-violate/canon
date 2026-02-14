use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "canon", about = "Canonical IR schema + validator", version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    Schema {
        #[arg(long)]
        pretty: bool,
    },
    Validate {
        path: PathBuf,
    },
    Ingest {
        #[arg(long)]
        src: PathBuf,
        #[arg(long = "semantic-out")]
        semantic_out: PathBuf,
        #[arg(long = "layout-out")]
        layout_out: Option<PathBuf>,
    },
    Lint {
        ir: PathBuf,
    },
    RenderFn {
        #[arg(long)]
        ir: PathBuf,
        #[arg(long = "fn-id")]
        fn_id: String,
    },
    DiffIr {
        #[arg(long)]
        before: PathBuf,
        #[arg(long)]
        after: PathBuf,
    },
    Materialize {
        ir: PathBuf,
        #[arg(long)]
        layout: Option<PathBuf>,
        out_dir: PathBuf,
    },
    ObserveEvents {
        execution: PathBuf,
        proof: String,
        output: PathBuf,
    },
    ExecuteTick {
        #[arg(long)]
        ir: PathBuf,
        #[arg(long)]
        tick: String,
        #[arg(long)]
        parallel: bool,
    },
    ExportDot {
        ir: PathBuf,
        #[arg(long)]
        layout: Option<PathBuf>,
        output: PathBuf,
    },
    VerifyDot {
        ir: PathBuf,
        #[arg(long)]
        layout: Option<PathBuf>,
        #[arg(long)]
        original: PathBuf,
    },
    SubmitDsl {
        dsl: PathBuf,
        #[arg(long)]
        ir: PathBuf,
        #[arg(long)]
        layout: Option<PathBuf>,
        #[arg(long = "output-ir")]
        output_ir: PathBuf,
        #[arg(long = "materialize-dir")]
        materialize_dir: PathBuf,
    },

    /// Drive one refactor pipeline run from CLI (requires pre-staged agent outputs).
    RunPipeline {
        #[arg(long)]
        ir: PathBuf,
        #[arg(long)]
        layout: Option<PathBuf>,
        /// Path to JSON file containing Vec<AgentCallOutput> (one per stage).
        #[arg(long)]
        outputs: PathBuf,
        /// Path to JSON file containing the RefactorProposal.
        #[arg(long)]
        proposal: PathBuf,
        /// Write mutated IR to this path.
        #[arg(long = "output-ir")]
        output_ir: PathBuf,
    },

    /// Fire one meta-tick over the capability graph and print the graph diff.
    MetaTick {
        /// Path to capability graph JSON (produced by save_capability_graph).
        #[arg(long)]
        graph: PathBuf,
        /// Path to reward ledger JSON.
        #[arg(long)]
        ledger: PathBuf,
        /// Write mutated graph to this path.
        #[arg(long = "output-graph")]
        output_graph: PathBuf,
    },

    /// Print ranked capability nodes with their EMA rewards.
    ShowLedger {
        #[arg(long)]
        ledger: PathBuf,
    },

    /// Print capability graph topology (nodes + edges).
    ShowGraph {
        #[arg(long)]
        graph: PathBuf,
    },

    /// Run the full Tier 7 agent loop.
    RunAgent {
        /// Starting IR path.
        #[arg(long)]
        ir: PathBuf,
        /// Starting layout path (optional, derived from IR if omitted).
        #[arg(long)]
        layout: Option<PathBuf>,
        /// Capability graph path (created as empty if missing).
        #[arg(long)]
        graph: PathBuf,
        /// Seed proposal JSON path (RefactorProposal).
        #[arg(long)]
        proposal: PathBuf,
        /// Write mutated IR here after each successful pipeline run.
        #[arg(long = "ir-out")]
        ir_out: PathBuf,
        /// Write ledger JSON here after each tick.
        #[arg(long = "ledger-out")]
        ledger_out: PathBuf,
        /// Write mutated graph here after each meta-tick.
        #[arg(long = "graph-out")]
        graph_out: PathBuf,
        /// Stop after N ticks (0 = run forever).
        #[arg(long, default_value = "0")]
        max_ticks: u64,
        /// Fire meta-tick every N ticks.
        #[arg(long, default_value = "10")]
        meta_tick_interval: u64,
        /// Update policy every N ticks.
        #[arg(long, default_value = "5")]
        policy_update_interval: u64,
    },

    /// Bootstrap the standard 5-node capability graph and a seed proposal.
    BootstrapGraph {
        /// Write capability graph JSON here.
        #[arg(long = "graph-out")]
        graph_out: PathBuf,
        /// Write seed proposal JSON here.
        #[arg(long = "proposal-out")]
        proposal_out: PathBuf,
        /// IR path — used to pick the first module as proposal target.
        #[arg(long)]
        ir: PathBuf,
    },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum LayoutStrategyArg {
    Original,
    SingleFile,
    PerType,
}
