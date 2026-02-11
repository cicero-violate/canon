use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

/// Metadata describing who proposed a patch and why.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PatchMetadata {
    pub agent_id: String,
    pub prompt: String,
    pub timestamp: String,
}

/// Structured patch proposal containing the raw diff and metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PatchProposal {
    pub id: String,
    pub diff: String,
    pub metadata: PatchMetadata,
}

/// Queue storing incoming proposals from untrusted agents.
#[derive(Debug, Default)]
pub struct PatchQueue {
    proposals: VecDeque<PatchProposal>,
}

impl PatchQueue {
    pub fn new() -> Self {
        Self {
            proposals: VecDeque::new(),
        }
    }

    pub fn enqueue(
        &mut self,
        diff: String,
        metadata: PatchMetadata,
    ) -> Result<&PatchProposal, PatchError> {
        if !is_structured_patch(&diff) {
            return Err(PatchError::MalformedPatch);
        }
        let id = compute_patch_id(&diff, &metadata);
        let proposal = PatchProposal { id, diff, metadata };
        self.proposals.push_back(proposal);
        Ok(self.proposals.back().expect("proposal just pushed"))
    }

    pub fn dequeue(&mut self) -> Option<PatchProposal> {
        self.proposals.pop_front()
    }

    pub fn len(&self) -> usize {
        self.proposals.len()
    }
}

/// Simple log entry capturing Canon's gate decision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PatchLogEntry {
    pub proposal_id: String,
    pub decision: PatchDecisionData,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PatchDecisionData {
    Accepted { proof_id: String },
    Rejected { reason: String },
}

/// Gate logic enforcing policy before a patch can proceed.
#[derive(Debug, Default)]
pub struct PatchGate {
    log: Vec<PatchLogEntry>,
}

impl PatchGate {
    pub fn new() -> Self {
        Self { log: Vec::new() }
    }

    pub fn evaluate(&mut self, proposal: PatchProposal) -> PatchDecision {
        if !is_structured_patch(&proposal.diff) {
            let entry = PatchLogEntry {
                proposal_id: proposal.id.clone(),
                decision: PatchDecisionData::Rejected {
                    reason: "Patch not structured".into(),
                },
            };
            self.log.push(entry.clone());
            return PatchDecision::Rejected {
                proposal,
                reason: "Patch not structured".into(),
            };
        }

        let proof_id = compute_patch_id(&proposal.diff, &proposal.metadata);
        let verified = VerifiedPatch {
            proposal,
            proof_id: proof_id.clone(),
        };
        let entry = PatchLogEntry {
            proposal_id: verified.proposal.id.clone(),
            decision: PatchDecisionData::Accepted { proof_id },
        };
        self.log.push(entry);
        PatchDecision::Accepted { verified }
    }

    pub fn log(&self) -> &[PatchLogEntry] {
        &self.log
    }
}

/// Result of a gate evaluation.
#[derive(Debug)]
pub enum PatchDecision {
    Accepted {
        verified: VerifiedPatch,
    },
    Rejected {
        proposal: PatchProposal,
        reason: String,
    },
}

/// Canon-certified patch ready for controlled application.
#[derive(Debug, Clone)]
pub struct VerifiedPatch {
    pub proposal: PatchProposal,
    pub proof_id: String,
}

/// Registry of approved patches for replay and audit.
#[derive(Debug, Default)]
pub struct ApprovedPatchRegistry {
    patches: BTreeMap<String, VerifiedPatch>,
}

impl ApprovedPatchRegistry {
    pub fn new() -> Self {
        Self {
            patches: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, patch: VerifiedPatch) {
        self.patches.insert(patch.proposal.id.clone(), patch);
    }

    pub fn get(&self, id: &str) -> Option<&VerifiedPatch> {
        self.patches.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &VerifiedPatch> {
        self.patches.values()
    }
}

/// Controlled applier that runs tests prior to applying patch content.
pub struct PatchApplier<F>
where
    F: Fn(&VerifiedPatch) -> bool,
{
    test_runner: F,
}

impl<F> PatchApplier<F>
where
    F: Fn(&VerifiedPatch) -> bool,
{
    pub fn new(test_runner: F) -> Self {
        Self { test_runner }
    }

    pub fn apply(&self, patch: &VerifiedPatch) -> Result<(), PatchError> {
        if !(self.test_runner)(patch) {
            return Err(PatchError::TestsFailed);
        }
        Ok(())
    }
}

fn is_structured_patch(diff: &str) -> bool {
    diff.contains("*** Begin Patch") && diff.contains("*** End Patch")
}

fn compute_patch_id(diff: &str, metadata: &PatchMetadata) -> String {
    let mut hasher = Hasher::new();
    hasher.update(diff.as_bytes());
    hasher.update(metadata.agent_id.as_bytes());
    hasher.update(metadata.prompt.as_bytes());
    hasher.update(metadata.timestamp.as_bytes());
    hasher.finalize().to_hex().to_string()
}

#[derive(Debug, thiserror::Error)]
pub enum PatchError {
    #[error("patch format is invalid")]
    MalformedPatch,
    #[error("tests failed before applying patch")]
    TestsFailed,
}
