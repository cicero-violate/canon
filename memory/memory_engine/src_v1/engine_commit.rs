use crate::{
    memory_engine::{MemoryEngineError, CommitError},
    proofs::{AdmissionProof, CommitProof},
    primitives::Hash,
    delta::DeltaError,
};

use crate::memory_engine::MemoryEngine;

impl MemoryEngine {
    pub fn commit_delta(
        &self,
        admission: &AdmissionProof,
        delta_hash: &Hash,
    ) -> Result<CommitProof, MemoryEngineError> {
        let delta = self
            .fetch_delta_by_hash(delta_hash)
            .ok_or(MemoryEngineError::DeltaNotFound)?;

        self.state
            .write()
            .apply_delta(&delta)
            .map_err(|err: DeltaError| {
                MemoryEngineError::Commit(CommitError::TlogWrite(
                    std::io::Error::new(std::io::ErrorKind::Other, err.to_string()),
                ))
            })?;

        self.state.write().page_store.flush().ok();

        self.epoch.increment();

        self.tlog
            .append(admission, delta.clone())
            .map_err(CommitError::TlogWrite)?;

        Ok(CommitProof {
            admission_proof_hash: admission.hash(),
            delta_hash: *delta_hash,
            state_hash: self.state.read().root_hash(),
        })
    }
}
