use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use database::delta::{Delta, ShellDelta};
use database::hash::gpu::create_gpu_backend;
use database::primitives::Hash;
use database::MerkleState;
use hex;
use host_state_controller::{RunReceipt, StateController};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::lean_gate::{self, ShellMetadata};
use crate::tlog::{TlogEntry, TlogManager};

pub const ZERO_PROOF_HASH: [u8; 32] = [0u8; 32];
pub const SHELL_GRAPH_ID: &str = "system.shell";

#[derive(Debug, Clone, Serialize)]
pub struct StateSlice {
    pub root_hash: String,
}

impl StateSlice {
    fn new(hash: Hash) -> Self {
        Self { root_hash: hex::encode(hash) }
    }
}

pub struct CanonicalState {
    inner: MerkleState,
}

impl CanonicalState {
    pub fn new_empty() -> Self {
        Self { inner: MerkleState::new_empty(create_gpu_backend()) }
    }

    pub fn root_hash(&self) -> Hash {
        self.inner.root_hash()
    }

    pub fn apply_delta(&mut self, delta: &Delta) -> Result<(), database::delta::DeltaError> {
        self.inner.apply_delta(delta)
    }

    pub fn to_slice(&self) -> StateSlice {
        StateSlice::new(self.inner.root_hash())
    }

    pub fn flush_to_disk(&self, repo_root: &Path) -> std::io::Result<()> {
        let path = snapshot_path(repo_root);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        self.inner.checkpoint(&path)
    }

    pub fn load_from_disk(repo_root: &Path) -> std::io::Result<Option<Self>> {
        let path = snapshot_path(repo_root);
        if !path.exists() {
            return Ok(None);
        }
        let inner = MerkleState::restore_from_checkpoint(&path, create_gpu_backend())?;
        Ok(Some(Self { inner }))
    }
}

pub fn resolve_fixture_path(repo_root: &Path) -> Result<PathBuf> {
    let fixture = env::var("CANON_FIXTURE").unwrap_or_else(|_| "simple_add".to_string());
    let path = repo_root.join("canon").join("fixtures").join(format!("{fixture}.json"));
    Ok(path)
}

pub fn state_dir(repo_root: &Path) -> PathBuf {
    repo_root.join("canon_store")
}

pub fn tlog_path(repo_root: &Path) -> PathBuf {
    state_dir(repo_root).join("canon.tlog")
}

fn snapshot_path(repo_root: &Path) -> PathBuf {
    state_dir(repo_root).join("snapshots").join("canonical_state.bin")
}

pub fn shell_state_dir(repo_root: &Path) -> PathBuf {
    state_dir(repo_root).join("shells")
}

pub fn load_state(repo_root: &Path) -> Result<CanonicalState> {
    std::fs::create_dir_all(state_dir(repo_root))?;
    let manager = TlogManager::new(tlog_path(repo_root)).context("open canon.tlog")?;
    let entries = manager.read_entries().context("read canon.tlog")?;
    if entries.is_empty() {
        if let Some(snapshot) = CanonicalState::load_from_disk(repo_root).with_context(|| "read snapshot")? {
            return Ok(snapshot);
        }
    }

    let mut state = CanonicalState::new_empty();
    let mut shell_events = Vec::new();
    for entry in &entries {
        verify_proof_hash(repo_root, entry)?;
        state.apply_delta(&entry.delta).map_err(|err| anyhow!("replay apply failed: {err}"))?;
        if state.root_hash() != entry.root_hash {
            bail!("replay hash mismatch");
        }
        if let Some(shell_delta) = ShellDelta::try_from_delta(&entry.delta).map_err(|err| anyhow!(err.to_string()))? {
            shell_events.push(shell_delta);
        }
    }
    rebuild_shell_ledgers(repo_root, &shell_events)?;
    Ok(state)
}

pub fn gate_and_commit(repo_root: &Path, state: &mut CanonicalState, graph_id: &str, deltas: &[Delta], shell: Option<ShellMetadata>) -> Result<()> {
    let state_slice = state.to_slice();
    let proposal = lean_gate::Proposal {
        graph_id: graph_id.to_string(),
        deltas: deltas.iter().map(|delta| lean_gate::ProposalDelta { delta_id: delta.delta_id.0, payload_hash: hex::encode(&delta.payload), bytes: delta.payload.len() }).collect(),
        shell,
    };
    let gate_dir = repo_root.join("canon").join("proof_gate");
    let decision = lean_gate::verify_proposal(&gate_dir, &state_slice, &proposal).context("Lean gate invocation failed")?;
    if !decision.accepted {
        let reason = decision.rejection_reason.unwrap_or_else(|| "proposal rejected".into());
        bail!("ProofGate rejected proposal: {reason}");
    }
    let proof_hash = persist_certificate(repo_root, &decision)?;

    let manager = TlogManager::new(tlog_path(repo_root)).context("open canon.tlog")?;
    for delta in deltas {
        state.apply_delta(delta).map_err(|err| anyhow!("state apply failed: {err}"))?;
        manager.append(delta, state.root_hash(), proof_hash).context("append delta to canon.tlog")?;
    }
    state.flush_to_disk(repo_root).context("flush canonical state to disk")?;
    Ok(())
}

fn rebuild_shell_ledgers(repo_root: &Path, shell_deltas: &[ShellDelta]) -> Result<()> {
    if shell_deltas.is_empty() {
        return Ok(());
    }
    let shell_dir = shell_state_dir(repo_root);
    if shell_dir.exists() {
        for entry in std::fs::read_dir(&shell_dir)? {
            let path = entry?.path();
            if path.file_name().and_then(|name| name.to_str()).map(|name| name.starts_with("shell_") && name.ends_with(".ledger")).unwrap_or(false) {
                std::fs::remove_file(path)?;
            }
        }
    } else {
        std::fs::create_dir_all(&shell_dir)?;
    }

    let mut controller = StateController::new(shell_dir).context("initialize controller for shell replay")?;
    let mut per_shell: BTreeMap<u64, Vec<RunReceipt>> = BTreeMap::new();
    for delta in shell_deltas {
        per_shell.entry(delta.shell_id).or_default().push(RunReceipt { epoch: delta.epoch, state_hash: delta.state_hash.clone(), shell_id: delta.shell_id, command: delta.command.clone() });
    }
    for receipts in per_shell.values_mut() {
        receipts.sort_by_key(|receipt| receipt.epoch);
    }
    for receipts in per_shell.values() {
        for receipt in receipts {
            controller.persist_receipt(receipt).context("replay shell ledger entry")?;
        }
    }
    Ok(())
}

fn persist_certificate(repo_root: &Path, decision: &lean_gate::ProofResult) -> Result<[u8; 32]> {
    let cert = decision.certificate.as_ref().ok_or_else(|| anyhow!("ProofGate accepted but omitted certificate"))?;
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

pub fn verify_proof_hash(repo_root: &Path, entry: &TlogEntry) -> Result<()> {
    if entry.proof_hash == ZERO_PROOF_HASH {
        return Ok(());
    }
    let proofs_dir = state_dir(repo_root).join("proofs");
    let filename = format!("{}.json", hex::encode(entry.proof_hash));
    let path = proofs_dir.join(filename);
    let data = std::fs::read(&path).with_context(|| format!("missing proof artifact {}", path.display()))?;
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
    use super::*;
    use database::delta::delta_types::Source;
    use database::epoch::Epoch;
    use database::primitives::{DeltaID, PageID};
    use host_state_controller::StateController;
    use serde::Deserialize;
    use tempfile::TempDir;

    fn sample_delta(id: u64) -> Delta {
        let payload = vec![id as u8];
        let mask = vec![true];
        Delta::new_dense(DeltaID(id), PageID(id), Epoch(0), payload, mask, Source(format!("test::{id}"))).expect("delta")
    }

    #[test]
    fn replay_matches_after_restart() -> Result<()> {
        let dir = TempDir::new().expect("tempdir");
        let repo_root = dir.path();
        let mut state = CanonicalState::new_empty();
        let deltas = vec![sample_delta(1), sample_delta(2)];
        persist_deltas_for_test(repo_root, &mut state, &deltas)?;
        let loaded = load_state(repo_root)?;
        assert_eq!(hex::encode(state.root_hash()), hex::encode(loaded.root_hash()));
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
        let receipt = RunReceipt { epoch: 1, state_hash: "hash-1".into(), shell_id: 7, command: "echo hi".into() };
        let delta = ShellDelta::new(receipt.shell_id, receipt.epoch, receipt.command.clone(), receipt.state_hash.clone()).into_delta(DeltaID(1), PageID(1), Epoch(0)).unwrap();
        persist_deltas_for_test(repo_root, &mut state, &[delta])?;
        if shell_state_dir(repo_root).exists() {
            std::fs::remove_dir_all(shell_state_dir(repo_root))?;
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

    fn persist_deltas_for_test(repo_root: &Path, state: &mut CanonicalState, deltas: &[Delta]) -> Result<()> {
        let manager = TlogManager::new(tlog_path(repo_root)).context("open canon.tlog")?;
        for delta in deltas {
            state.apply_delta(delta).map_err(|err| anyhow!("state apply failed: {err}"))?;
            manager.append(delta, state.root_hash(), ZERO_PROOF_HASH).context("append delta to canon.tlog")?;
        }
        state.flush_to_disk(repo_root).context("flush canonical state to disk")?;
        Ok(())
    }
}
