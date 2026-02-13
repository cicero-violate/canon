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
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum LayoutStrategyArg {
    Original,
    SingleFile,
    PerType,
}
