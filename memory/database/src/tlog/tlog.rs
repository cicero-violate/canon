//! Transaction Log — Append-only persistence for MMSB mutations
//!
//! The `TransactionLog` is part of `mmsb-memory`’s persistence layer.
//! It durably records committed deltas in an append-only binary format.
//!
//! Authority: none  
//! Responsibility: durable byte storage under memory’s control  
//!
//! Notes:
//! - No execution or judgment proofs are stored here
//! - Pages are never persisted directly
//! - Recovery is performed exclusively via replay

use crate::delta::Delta;
use crate::proofs::AdmissionProof;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;


/// Log file magic header
const MAGIC: &[u8] = b"MMSBLOG1";

/// Current on-disk log version
const VERSION: u32 = 2;

/// Append-only transaction log
#[derive(Debug)]
pub struct TransactionLog {
    entries: RwLock<VecDeque<Delta>>,
    writer: RwLock<Option<BufWriter<File>>>,
    path: PathBuf,
}

/// Lightweight summary of log contents
#[derive(Debug, Default, Clone, Copy)]
pub struct LogSummary {
    pub total_deltas: u64,
    pub total_bytes: u64,
    pub last_epoch: u32,
}

impl TransactionLog {
    /// Open or create a transaction log at `path`
    pub fn new(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();

        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;

        // Initialize header if empty
        if file.metadata()?.len() == 0 {
            file.write_all(MAGIC)?;
            file.write_all(&VERSION.to_le_bytes())?;
            file.flush()?;
        }

        Ok(Self {
            entries: RwLock::new(VecDeque::new()),
            writer: RwLock::new(Some(BufWriter::new(file))),
            path,
        })
    }

    /// Clear in-memory entries only.
    ///
    /// Does NOT modify the underlying log file.
    /// Used for state resets while preserving append-only semantics.
    pub fn clear_entries(&self) {
        self.entries.write().clear();
    }

    /// Append a delta with its admission proof witness.
    ///
    /// No execution proof is required or accepted here.
    pub fn append(&self, admission_proof: &AdmissionProof, delta: Delta) -> std::io::Result<()> {
        // Minimal sanity validation
        if admission_proof.epoch == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid admission epoch",
            ));
        }

        let serialized = bincode::serialize(&(admission_proof, &delta))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let mut writer = self.writer.write();
        if let Some(w) = writer.as_mut() {
            let w: &mut BufWriter<File> = w;
            w.write_all(&serialized.len().to_le_bytes())?;
            w.write_all(&serialized)?;
            w.flush()?;
        }

        self.entries.write().push_back(delta);
        Ok(())
    }

    /// Return a lightweight summary of the log
    pub fn summary(&self) -> std::io::Result<LogSummary> {
        // Placeholder: real implementation scans file
        Ok(LogSummary {
            total_deltas: self.entries.read().len() as u64,
            total_bytes: 0,
            last_epoch: 0,
        })
    }

    /// Replay deltas from memory (placeholder).
    ///
    /// Real replay should stream from disk via `TransactionLogReader`.
    pub fn replay(&self, _start_epoch: u32) -> std::io::Result<Vec<Delta>> {
        Ok(self.entries.read().iter().cloned().collect())
    }

    /// Return the current byte offset of the log file
    pub fn current_offset(&self) -> std::io::Result<u64> {
        Ok(File::open(&self.path)?.metadata()?.len())
    }
}

/// Validate log header and return version



#[cfg(test)]
mod tests {
    use super::TransactionLog;
    use crate::delta::delta_types::Source;
    use crate::delta::Delta;
    use crate::epoch::Epoch;
    use crate::primitives::{DeltaID, PageID};
    use crate::proofs::AdmissionProof;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn base_delta() -> Delta {
        Delta::new_dense(
            DeltaID(1),
            PageID(1),
            Epoch(0),
            vec![1u8],
            vec![true],
            Source("test".into()),
        )
        .expect("delta")
    }

    fn dummy_admission_proof() -> AdmissionProof {
        AdmissionProof {
            judgment_proof_hash: [0u8; 32],
            epoch: 1,
            nonce: 1,
        }
    }

    #[test]
    fn append_with_valid_admission_proof_succeeds() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        let path = std::env::temp_dir().join(format!("mmsb_append_ok_{nanos}.tlog"));

        let log = TransactionLog::new(&path).expect("create log");
        let delta = base_delta();
        let admission = dummy_admission_proof();

        log.append(&admission, delta).expect("append succeeds");
    }
}
