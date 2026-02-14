mod cli;
mod commands;
mod diff;
mod io_utils;
mod version_gate;
mod decision;
mod dot_export;
mod dot_import;
mod evolution;
mod ingest;
mod ir;
mod layout;
mod materialize;
mod observe;
mod patch_protocol;
mod proof;
mod proposal;
mod runtime;
mod schema;
mod semantic_builder;
mod validate;

pub use ir::CanonicalIr;

use clap::Parser;
use cli::Cli;
use commands::execute_command;

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    execute_command(cli.command)
}
