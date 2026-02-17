use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, anyhow};
use crate::state_io::StateSlice;
use serde::{Deserialize, Serialize};

/// Deterministic proposal payload shared with Lean.
#[derive(Debug, Clone, Serialize)]
pub struct Proposal {
    pub graph_id: String,
    pub deltas: Vec<ProposalDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shell: Option<ShellMetadata>,
}

/// Deterministic summary for each proposed delta.
#[derive(Debug, Clone, Serialize)]
pub struct ProposalDelta {
    pub delta_id: u64,
    pub payload_hash: String,
    pub bytes: usize,
}

/// Certificate emitted by ProofGate when a proposal is proven admissible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateCertificate {
    pub proposal_name: String,
    pub proof_tag: String,
    #[serde(default)]
    pub state_hash: Option<String>,
    #[serde(default)]
    pub delta_hash: Option<String>,
}

/// Proof result returned by the Lean gate.
#[derive(Debug)]
pub struct ProofResult {
    pub accepted: bool,
    pub certificate: Option<GateCertificate>,
    pub rejection_reason: Option<String>,
    pub proofs: Vec<InvariantProof>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantProof {
    pub name: String,
    pub status: String,
    pub detail: String,
}

/// Optional shell metadata that accompanies shell graph proposals.
#[derive(Debug, Clone, Serialize)]
pub struct ShellMetadata {
    pub shell_id: u64,
    pub epoch: u64,
    pub command: String,
    pub state_hash: String,
}

#[derive(Debug, Serialize)]
struct LeanRequest<'a> {
    state: &'a StateSlice,
    proposal: &'a Proposal,
}

#[derive(Debug, Deserialize)]
struct RawProofResponse {
    status: String,
    certificate: Option<GateCertificate>,
    error: Option<String>,
    #[serde(default)]
    proofs: Vec<InvariantProof>,
    #[serde(default)]
    rejection: Option<RawRejection>,
}

#[derive(Debug, Deserialize)]
struct RawRejection {
    reason: Option<String>,
    detail: Option<String>,
}

/// Invoke the ProofGate CLI with the canonical schema.
pub fn verify_proposal(
    gate_dir: &Path,
    state: &StateSlice,
    proposal: &Proposal,
) -> Result<ProofResult> {
    let payload = serde_json::to_vec(&LeanRequest { state, proposal })?;

    let mut child = Command::new("lake")
        .arg("exe")
        .arg("proof_gate")
        .current_dir(gate_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to invoke ProofGate at {}", gate_dir.display()))?;

    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(&payload)
            .context("unable to send request to ProofGate")?;
    } else {
        return Err(anyhow!("failed to open ProofGate stdin"));
    }

    let output = child
        .wait_with_output()
        .context("failed to await ProofGate result")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "ProofGate verification failed: {}",
            stderr.trim().to_string()
        ));
    }

    let response: RawProofResponse =
        serde_json::from_slice(&output.stdout).context("ProofGate returned invalid JSON")?;

    match response.status.as_str() {
        "accepted" => Ok(ProofResult {
            accepted: true,
            certificate: response.certificate,
            rejection_reason: None,
            proofs: response.proofs,
        }),
        "rejected" | "error" => {
            let reason = response
                .rejection
                .and_then(|rej| rej.reason.or(rej.detail))
                .or(response.error)
                .unwrap_or_else(|| "proposal rejected".to_string());
            Ok(ProofResult {
                accepted: false,
                certificate: None,
                rejection_reason: Some(reason),
                proofs: response.proofs,
            })
        }
        other => Err(anyhow!("unknown ProofGate status `{other}`")),
    }
}
