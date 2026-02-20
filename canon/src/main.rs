mod agent;
mod cli;
mod commands;
mod decision;
mod diagnose;
mod diff;
mod dot_export;
mod dot_import;
mod evolution;
mod gpu;
mod ingest;
mod io_utils;
mod ir;
mod layout;
mod materialize;
mod observe;
mod patch_protocol;
mod proof;
mod runtime;
mod schema;
mod semantic_builder;
mod storage;
mod validate;
mod version_gate;
pub use ir::SystemState;
use clap::Parser;
use cli::Cli;
use commands::execute_command;
#[tokio::main]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    execute_command(cli.command).await
}
