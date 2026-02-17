mod lean_gate;
mod shell_flow;
mod state_io;
mod tlog;
mod verify_flow;

use crate::shell_flow::run_shell_flow;
use crate::verify_flow::{CliMode, parse_cli_mode, run_system_flow};
use anyhow::{Result, anyhow};
use std::path::PathBuf;

fn main() -> Result<()> {
    // Resolve workspace root deterministically from this crate location.
    // CARGO_MANIFEST_DIR = <workspace>/canon/host_runtime
    // We want <workspace>/canon
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| anyhow!("cannot resolve host_runtime parent"))?
        .to_path_buf();
    match parse_cli_mode()? {
        CliMode::Shell(options) => run_shell_flow(&repo_root, options),
        CliMode::Execute {
            verify_only,
            intent,
        } => run_system_flow(&repo_root, verify_only, intent),
    }
}
