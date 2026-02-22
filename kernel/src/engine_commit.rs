use crate::Kernel;
use database::{
    delta::DeltaError,
    primitives::StateHash,
    proofs::{AdmissionProof, CommitProof},
};
use crate::kernel::{KernelError, CommitError};
impl Kernel {
    pub fn commit_delta(&self, admission: &AdmissionProof, delta_hash: &StateHash) -> Result<CommitProof, KernelError> {
        let delta = self.fetch_delta_by_hash(delta_hash).ok_or(KernelError::DeltaNotFound)?;
        self.state.write().apply_delta(&delta).map_err(|err: DeltaError| KernelError::Commit(CommitError::TlogWrite(std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))))?;
        // flush handled internally by database layer if needed
        self.epoch.increment();
        Ok(CommitProof { admission_proof_hash: admission.hash(), delta_hash: *delta_hash, state_hash: self.state.read().root_hash() })
    }
}
