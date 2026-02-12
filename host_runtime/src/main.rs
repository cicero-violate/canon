mod lean_gate;

use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use canon::ir::{CanonicalIr, SystemGraph};
use canon::memory::delta::shell_delta::ShellDelta;
use canon::memory::delta::Delta;
use canon::memory::epoch::Epoch;
use canon::memory::primitives::{DeltaID, PageID};
use canon::memory::{CanonicalState, TlogEntry, TlogManager};
use canon::runtime::value::ScalarValue;
use canon::runtime::{SystemExecutionEvent, SystemExecutionResult, SystemInterpreter, Value};
use hex;
use host_state_controller::{RunReceipt, StateController};
use lean_gate::ShellMetadata;
use serde_json::json;
use sha2::{Digest, Sha256};

const ZERO_PROOF_HASH: [u8; 32] = [0u8; 32];
const SHELL_GRAPH_ID: &str = "system.shell";

fn main() -> Result<()> {
    // Resolve workspace root deterministically from this crate location.
    // CARGO_MANIFEST_DIR = <workspace>/canon/host_runtime
    // We want <workspace>/canon
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent() // host_runtime
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

fn run_system_flow(repo_root: &Path, verify_only: bool, intent: Intent) -> Result<()> {
    let ir_path = resolve_fixture_path(repo_root)?;
    let ir = load_ir(&ir_path)?;

    println!("Loaded fixture {}", ir_path.display());

    let mut state = load_state(repo_root)?;
    if verify_only {
        println!("Verified root hash {}", hex::encode(state.root_hash()));
        return Ok(());
    }

    println!("Intent: lhs={}, rhs={}", intent.lhs, intent.rhs);

    let interpreter = SystemInterpreter::new(&ir);
    let inputs = build_initial_inputs(&intent);
    let graph = select_system_graph(&ir)?;
    let execution = interpreter.execute_graph(&graph.id, inputs)?;
    report_execution(&execution)?;
    run_gate_and_persist(repo_root, &mut state, &graph.id, &execution)?;
    Ok(())
}

fn run_shell_flow(repo_root: &Path, options: ShellOptions) -> Result<()> {
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

fn load_ir(path: &Path) -> Result<CanonicalIr> {
    let data = std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let ir = serde_json::from_slice(&data)?;
    Ok(ir)
}

fn select_system_graph<'a>(ir: &'a CanonicalIr) -> Result<&'a SystemGraph> {
    ir.system_graphs
        .first()
        .ok_or_else(|| anyhow!("no system graphs defined in Canonical IR"))
}

fn build_initial_inputs(intent: &Intent) -> BTreeMap<String, Value> {
    let mut inputs = BTreeMap::new();
    inputs.insert("Lhs".into(), Value::Scalar(ScalarValue::I32(intent.lhs)));
    inputs.insert("Rhs".into(), Value::Scalar(ScalarValue::I32(intent.rhs)));
    inputs
}

fn report_execution(result: &SystemExecutionResult) -> Result<()> {
    println!(
        "Executed system graph `{}` across {} nodes ({} deltas).",
        result.graph_id,
        result.execution_order.len(),
        result.emitted_deltas.len()
    );

    for event in &result.events {
        match event {
            SystemExecutionEvent::Validation { check, detail } => {
                println!("[validation] {check}: {detail}");
            }
            SystemExecutionEvent::NodeExecuted { node_id, kind } => {
                println!("[node] {node_id} ({kind:?}) executed");
            }
            SystemExecutionEvent::ProofRecorded { node_id, proof_id } => {
                println!("[proof-event] {node_id} proof_id={proof_id}");
            }
            SystemExecutionEvent::DeltaRecorded { node_id, count } => {
                println!("[delta-event] {node_id} emitted {count} deltas");
            }
        }
    }

    if result.proof_artifacts.is_empty() {
        println!("No proofs recorded.");
    } else {
        for proof in &result.proof_artifacts {
            println!(
                "[proof] node={} proof_id={} accepted={}",
                proof.node_id, proof.proof_id, proof.accepted
            );
        }
    }

    if result.delta_provenance.is_empty() {
        println!("No deltas recorded.");
    } else {
        for delta in &result.delta_provenance {
            let summary: Vec<String> = delta
                .deltas
                .iter()
                .map(|d| format!("delta_id={} bytes={}", d.delta_id.0, d.payload.len()))
                .collect();
            println!("[delta] node={} {}", delta.node_id, summary.join(", "));
        }
    }

    if let Some(last_node) = result.execution_order.last() {
        if let Some(outputs) = result.node_results.get(last_node) {
            println!("Final outputs: {outputs:?}");
        }
    }

    Ok(())
}

#[derive(Debug)]
struct Intent {
    lhs: i32,
    rhs: i32,
}

#[derive(Debug, Clone)]
struct ShellBranch {
    parent_id: u64,
    epoch: Option<u64>,
}

#[derive(Debug, Clone)]
struct ShellOptions {
    command: String,
    shell_id: Option<u64>,
    restore_epoch: Option<u64>,
    branch: Option<ShellBranch>,
}

enum CliMode {
    Execute { verify_only: bool, intent: Intent },
    Shell(ShellOptions),
}

fn parse_cli_mode() -> Result<CliMode> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if let Some(first) = args.first() {
        if first == "shell" {
            args.remove(0);
            return parse_shell_mode(args);
        }
    }
    let verify = if let Some(idx) = args.iter().position(|arg| arg == "--verify") {
        args.remove(idx);
        true
    } else {
        false
    };
    let lhs = args.get(0).and_then(|a| a.parse().ok()).unwrap_or(2);
    let rhs = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(3);
    Ok(CliMode::Execute {
        verify_only: verify,
        intent: Intent { lhs, rhs },
    })
}

fn parse_shell_mode(mut args: Vec<String>) -> Result<CliMode> {
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
    let command = command_parts.join(" ");
    Ok(CliMode::Shell(ShellOptions {
        command,
        shell_id,
        restore_epoch,
        branch,
    }))
}

fn resolve_fixture_path(repo_root: &Path) -> Result<std::path::PathBuf> {
    let fixture = env::var("CANON_FIXTURE").unwrap_or_else(|_| "simple_add".to_string());
    let path = repo_root
        .join("canon")
        .join("fixtures")
        .join(format!("{fixture}.json"));
    if path.exists() {
        Ok(path)
    } else {
        Err(anyhow!(
            "requested fixture `{fixture}` missing at {}",
            path.display()
        ))
    }
}

fn state_dir(repo_root: &Path) -> PathBuf {
    repo_root.join("canon_store")
}

fn tlog_path(repo_root: &Path) -> PathBuf {
    state_dir(repo_root).join("canon.tlog")
}

fn shell_state_dir(repo_root: &Path) -> PathBuf {
    state_dir(repo_root).join("shells")
}

pub(crate) fn load_state(repo_root: &Path) -> Result<CanonicalState> {
    std::fs::create_dir_all(state_dir(repo_root))?;
    let manager = TlogManager::new(tlog_path(repo_root)).context("open canon.tlog")?;
    let entries = manager.read_entries().context("read canon.tlog")?;
    if entries.is_empty() {
        if let Some(snapshot) =
            CanonicalState::load_from_disk(repo_root).with_context(|| "read snapshot")?
        {
            return Ok(snapshot);
        }
    }

    let mut state = CanonicalState::new_empty();
    let mut shell_events = Vec::new();
    for entry in &entries {
        verify_proof_hash(repo_root, &entry)?;
        state
            .apply_delta(&entry.delta)
            .map_err(|err| anyhow!("replay apply failed: {err}"))?;
        if state.root_hash() != entry.root_hash {
            bail!("replay hash mismatch");
        }
        if let Some(shell_delta) =
            ShellDelta::try_from_delta(&entry.delta).map_err(|err| anyhow!(err.to_string()))?
        {
            shell_events.push(shell_delta);
        }
    }
    rebuild_shell_ledgers(repo_root, &shell_events)?;
    Ok(state)
}

fn rebuild_shell_ledgers(repo_root: &Path, shell_deltas: &[ShellDelta]) -> Result<()> {
    if shell_deltas.is_empty() {
        return Ok(());
    }
    let shell_dir = shell_state_dir(repo_root);
    if shell_dir.exists() {
        for entry in std::fs::read_dir(&shell_dir)? {
            let path = entry?.path();
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("shell_") && name.ends_with(".ledger"))
                .unwrap_or(false)
            {
                std::fs::remove_file(path)?;
            }
        }
    } else {
        std::fs::create_dir_all(&shell_dir)?;
    }

    let mut controller =
        StateController::new(shell_dir).context("initialize controller for shell replay")?;
    let mut per_shell: BTreeMap<u64, Vec<RunReceipt>> = BTreeMap::new();
    for delta in shell_deltas {
        per_shell
            .entry(delta.shell_id)
            .or_default()
            .push(RunReceipt {
                epoch: delta.epoch,
                state_hash: delta.state_hash.clone(),
                shell_id: delta.shell_id,
                command: delta.command.clone(),
            });
    }
    for receipts in per_shell.values_mut() {
        receipts.sort_by_key(|receipt| receipt.epoch);
    }
    for receipts in per_shell.values() {
        for receipt in receipts {
            controller
                .persist_receipt(receipt)
                .context("replay shell ledger entry")?;
        }
    }
    Ok(())
}

fn run_gate_and_persist(
    repo_root: &Path,
    state: &mut CanonicalState,
    graph_id: &str,
    execution: &SystemExecutionResult,
) -> Result<()> {
    gate_and_commit(repo_root, state, graph_id, &execution.emitted_deltas, None)
}

fn gate_and_commit(
    repo_root: &Path,
    state: &mut CanonicalState,
    graph_id: &str,
    deltas: &[Delta],
    shell: Option<ShellMetadata>,
) -> Result<()> {
    let state_slice = state.to_slice();
    let proposal = lean_gate::Proposal {
        graph_id: graph_id.to_string(),
        deltas: deltas
            .iter()
            .map(|delta| lean_gate::ProposalDelta {
                delta_id: delta.delta_id.0,
                payload_hash: hex::encode(&delta.payload),
                bytes: delta.payload.len(),
            })
            .collect(),
        shell,
    };
    let gate_dir = repo_root.join("canon").join("proof_gate");
    let decision = lean_gate::verify_proposal(&gate_dir, &state_slice, &proposal)
        .context("Lean gate invocation failed")?;
    if !decision.accepted {
        let reason = decision
            .rejection_reason
            .unwrap_or_else(|| "proposal rejected".into());
        bail!("ProofGate rejected proposal: {reason}");
    }
    let proof_hash = persist_certificate(repo_root, &decision)?;

    let manager = TlogManager::new(tlog_path(repo_root)).context("open canon.tlog")?;
    for delta in deltas {
        state
            .apply_delta(delta)
            .map_err(|err| anyhow!("state apply failed: {err}"))?;
        manager
            .append(delta, state.root_hash(), proof_hash)
            .context("append delta to canon.tlog")?;
    }
    state
        .flush_to_disk(repo_root)
        .context("flush canonical state to disk")?;
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

fn persist_certificate(repo_root: &Path, decision: &lean_gate::ProofResult) -> Result<[u8; 32]> {
    let cert = decision
        .certificate
        .as_ref()
        .ok_or_else(|| anyhow!("ProofGate accepted but omitted certificate"))?;
    let record = json!({
        "certificate": cert,
        "proofs": decision.proofs.clone(),
    });
    let bytes = serde_json::to_vec_pretty(&record)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let proof_hash: [u8; 32] = hasher.finalize().into();
    let proofs_dir = state_dir(repo_root).join("proofs");
    std::fs::create_dir_all(&proofs_dir)?;
    let path = proofs_dir.join(format!("{}.json", hex::encode(proof_hash)));
    std::fs::write(&path, bytes)?;
    Ok(proof_hash)
}

fn verify_proof_hash(repo_root: &Path, entry: &TlogEntry) -> Result<()> {
    if entry.proof_hash == ZERO_PROOF_HASH {
        return Ok(());
    }
    let proofs_dir = state_dir(repo_root).join("proofs");
    let filename = format!("{}.json", hex::encode(entry.proof_hash));
    let path = proofs_dir.join(filename);
    let data = std::fs::read(&path)
        .with_context(|| format!("missing proof artifact {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let computed: [u8; 32] = hasher.finalize().into();
    if computed != entry.proof_hash {
        bail!("proof hash mismatch for {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ZERO_PROOF_HASH;
    use super::*;
    use canon::memory::delta::delta_types::Source;
    use canon::memory::delta::Delta;
    use canon::memory::epoch::Epoch;
    use canon::memory::primitives::{DeltaID, PageID};
    use host_state_controller::RunReceipt;
    use serde::Deserialize;
    use tempfile::TempDir;

    fn sample_delta(id: u64) -> Delta {
        let payload = vec![id as u8];
        let mask = vec![true];
        Delta::new_dense(
            DeltaID(id),
            PageID(id),
            Epoch(0),
            payload,
            mask,
            Source(format!("test::{id}")),
        )
        .expect("delta")
    }

    #[test]
    fn replay_matches_after_restart() -> Result<()> {
        let dir = TempDir::new().expect("tempdir");
        let repo_root = dir.path();
        let mut state = CanonicalState::new_empty();
        let deltas = vec![sample_delta(1), sample_delta(2)];
        persist_deltas_for_test(repo_root, &mut state, &deltas)?;
        let loaded = load_state(repo_root)?;
        assert_eq!(
            hex::encode(state.root_hash()),
            hex::encode(loaded.root_hash())
        );
        Ok(())
    }

    #[test]
    fn tamper_detection_fails() -> Result<()> {
        let dir = TempDir::new().expect("tempdir");
        let repo_root = dir.path();
        let mut state = CanonicalState::new_empty();
        let deltas = vec![sample_delta(9)];
        persist_deltas_for_test(repo_root, &mut state, &deltas)?;
        let log_path = tlog_path(repo_root);
        let mut bytes = std::fs::read(&log_path)?;
        if let Some(last) = bytes.last_mut() {
            *last ^= 0xFF;
        }
        std::fs::write(&log_path, bytes)?;
        assert!(load_state(repo_root).is_err());
        Ok(())
    }

    #[test]
    fn shell_ledgers_are_rebuilt_during_replay() -> Result<()> {
        let dir = TempDir::new().expect("tempdir");
        let repo_root = dir.path();
        let mut state = CanonicalState::new_empty();
        let receipt = RunReceipt {
            epoch: 1,
            state_hash: "hash-1".into(),
            shell_id: 7,
            command: "echo hi".into(),
        };
        let delta = shell_delta_from_receipt(&receipt)?;
        persist_deltas_for_test(repo_root, &mut state, &[delta])?;
        let shell_dir = shell_state_dir(repo_root);
        if shell_dir.exists() {
            std::fs::remove_dir_all(&shell_dir)?;
        }
        let _ = load_state(repo_root)?;
        let ledger_path = shell_state_dir(repo_root).join("shell_7.ledger");
        let contents = std::fs::read_to_string(&ledger_path)?;
        let lines: Vec<_> = contents.lines().collect();
        assert_eq!(lines.len(), 1);
        #[derive(Deserialize)]
        struct LedgerEntry {
            shell_id: u64,
            epoch: u64,
            command: String,
            state_hash: String,
        }
        let entry: LedgerEntry = serde_json::from_str(lines[0])?;
        assert_eq!(entry.shell_id, 7);
        assert_eq!(entry.epoch, 1);
        assert_eq!(entry.command, "echo hi");
        assert_eq!(entry.state_hash, "hash-1");
        Ok(())
    }

    fn persist_deltas_for_test(
        repo_root: &Path,
        state: &mut CanonicalState,
        deltas: &[Delta],
    ) -> Result<()> {
        let manager = TlogManager::new(tlog_path(repo_root)).context("open canon.tlog")?;
        for delta in deltas {
            state
                .apply_delta(delta)
                .map_err(|err| anyhow!("state apply failed: {err}"))?;
            manager
                .append(delta, state.root_hash(), ZERO_PROOF_HASH)
                .context("append delta to canon.tlog")?;
        }
        state
            .flush_to_disk(repo_root)
            .context("flush canonical state to disk")?;
        Ok(())
    }
}
