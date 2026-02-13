mod cli;
mod commands;
mod io_utils;
mod diff;
mod version_gate;

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
