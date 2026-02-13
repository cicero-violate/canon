use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use hex;
use host_state_controller::{RunReceipt, StateController};
use memory_engine::delta::shell_delta::ShellDelta;
use memory_engine::delta::Delta;
use memory_engine::epoch::Epoch;
use memory_engine::primitives::{DeltaID, PageID};

use crate::lean_gate::ShellMetadata;
use crate::state_io::{gate_and_commit, load_state, shell_state_dir, SHELL_GRAPH_ID};

#[derive(Debug, Clone)]
pub struct ShellBranch {
    pub parent_id: u64,
    pub epoch: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct ShellOptions {
    pub command: String,
    pub shell_id: Option<u64>,
    pub restore_epoch: Option<u64>,
    pub branch: Option<ShellBranch>,
}

pub fn parse_shell_mode(mut args: Vec<String>) -> Result<ShellOptions> {
    let mut shell_id = None;
    let mut restore_epoch = None;
    let mut branch = None;
    let mut command_parts: Vec<String> = Vec::new();

    while !args.is_empty() {
        match args[0].as_str() {
            "--shell-id" => {
                args.remove(0);
                let value = args
                    .get(0)
                    .ok_or_else(|| anyhow!("--shell-id requires a value"))?;
                shell_id = Some(value.parse()?);
                args.remove(0);
            }
            "--restore-epoch" => {
                args.remove(0);
                let value = args
                    .get(0)
                    .ok_or_else(|| anyhow!("--restore-epoch requires a value"))?;
                restore_epoch = Some(value.parse()?);
                args.remove(0);
            }
            "--branch" => {
                args.remove(0);
                let raw = args
                    .get(0)
                    .ok_or_else(|| anyhow!("--branch requires PARENT[@EPOCH]"))?
                    .clone();
                args.remove(0);
                let (parent_str, epoch_str) = raw
                    .split_once('@')
                    .map(|(parent, epoch)| (parent, Some(epoch)))
                    .unwrap_or((raw.as_str(), None));
                let parent_id = parent_str.parse()?;
                let epoch = if let Some(epoch) = epoch_str {
                    Some(epoch.parse()?)
                } else {
                    None
                };
                branch = Some(ShellBranch { parent_id, epoch });
            }
            "--" => {
                args.remove(0);
                command_parts.extend(args.iter().cloned());
                args.clear();
                break;
            }
            _ => {
                command_parts.extend(args.iter().cloned());
                args.clear();
                break;
            }
        }
    }

    if command_parts.is_empty() {
        bail!("shell mode requires a command string");
    }

    Ok(ShellOptions {
        command: command_parts.join(" "),
        shell_id,
        restore_epoch,
        branch,
    })
}

pub fn run_shell_flow(repo_root: &Path, options: ShellOptions) -> Result<()> {
    let ShellOptions {
        command,
        shell_id,
        restore_epoch,
        branch,
    } = options;
    println!("Executing shell command: {}", command);
    if branch.is_some() && shell_id.is_some() {
        bail!("cannot combine --branch and --shell-id");
    }
    let mut state = load_state(repo_root)?;
    let shell_dir = shell_state_dir(repo_root);
    let mut controller =
        StateController::new(shell_dir).context("initialize host state controller")?;

    let shell_id = if let Some(branch) = branch {
        controller
            .ensure_shell_registered(branch.parent_id)
            .context("branch parent missing")?;
        let epoch = branch
            .epoch
            .or_else(|| controller.latest_epoch(branch.parent_id))
            .unwrap_or(0);
        println!(
            "Branching shell {} at epoch {} for new command",
            branch.parent_id, epoch
        );
        controller.branch(branch.parent_id, epoch)?
    } else if let Some(id) = shell_id {
        controller
            .ensure_shell_registered(id)
            .context("selected shell missing")?;
        id
    } else {
        controller.spawn_root().context("spawn root shell")?
    };

    if let Some(epoch) = restore_epoch {
        controller
            .restore(shell_id, epoch)
            .context("restore shell to epoch")?;
    }

    let receipt = controller
        .run(shell_id, &command)
        .context("dispatch shell command")?;
    let delta = shell_delta_from_receipt(&receipt)?;
    let metadata = shell_metadata_from_receipt(&receipt);
    gate_and_commit(
        repo_root,
        &mut state,
        SHELL_GRAPH_ID,
        &[delta],
        Some(metadata),
    )?;
    controller
        .persist_receipt(&receipt)
        .context("persist shell ledger")?;
    println!(
        "Shell {} epoch {} committed; root hash {}",
        receipt.shell_id,
        receipt.epoch,
        hex::encode(state.root_hash())
    );
    Ok(())
}

fn shell_delta_from_receipt(receipt: &RunReceipt) -> Result<Delta> {
    let id = ((receipt.shell_id as u64) << 32) | receipt.epoch;
    let delta_id = DeltaID(id);
    let page_id = PageID(id);
    let epoch = Epoch(receipt.epoch.min(u32::MAX as u64) as u32);
    let shell_delta = ShellDelta::new(
        receipt.shell_id,
        receipt.epoch,
        receipt.command.clone(),
        receipt.state_hash.clone(),
    );
    shell_delta
        .into_delta(delta_id, page_id, epoch)
        .map_err(|err| anyhow!("shell delta encode failed: {err}"))
}

fn shell_metadata_from_receipt(receipt: &RunReceipt) -> ShellMetadata {
    ShellMetadata {
        shell_id: receipt.shell_id,
        epoch: receipt.epoch,
        command: receipt.command.clone(),
        state_hash: receipt.state_hash.clone(),
    }
}
