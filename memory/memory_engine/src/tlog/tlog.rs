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
use crate::memory_engine::CanonicalState;
use crate::primitives::Hash;
use crate::proofs::AdmissionProof;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

/// Compute a stable content hash for a delta
pub fn delta_hash(delta: &Delta) -> String {
    let mut hasher = Sha256::new();

    hasher.update(&delta.delta_id.0.to_le_bytes());
    hasher.update(&delta.page_id.0.to_le_bytes());
    hasher.update(&delta.epoch.0.to_le_bytes());

    for &bit in &delta.mask {
        hasher.update(&[bit as u8]);
    }

    hasher.update(&delta.payload);
    format!("{:x}", hasher.finalize())
}

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


const CANON_TLOG_MAGIC: &[u8] = b"CANON_TLOG1";
const CANON_TLOG_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlogEntry {
    pub delta: Delta,
    pub root_hash: Hash,
    pub proof_hash: Hash,
}

#[derive(Debug)]
pub struct TlogManager {
    path: PathBuf,
    writer: RwLock<BufWriter<File>>,
}

impl TlogManager {
    pub fn new(path: impl Into<PathBuf>) -> io::Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        if file.metadata()?.len() == 0 {
            file.write_all(CANON_TLOG_MAGIC)?;
            file.write_all(&CANON_TLOG_VERSION.to_le_bytes())?;
            file.flush()?;
        } else {
            validate_manager_header(&path)?;
        }
        Ok(Self {
            path,
            writer: RwLock::new(BufWriter::new(file)),
        })
    }

    pub fn append(&self, delta: &Delta, root_hash: Hash, proof_hash: Hash) -> io::Result<()> {
        let entry = TlogEntry {
            delta: delta.clone(),
            root_hash,
            proof_hash,
        };
        let blob = bincode::serialize(&entry)
            .map_err(|err| io::Error::new(ErrorKind::Other, err.to_string()))?;
        let mut guard = self.writer.write();
        guard.write_all(&(blob.len() as u32).to_le_bytes())?;
        guard.write_all(&blob)?;
        guard.flush()?;
        Ok(())
    }

    pub fn replay_all(&self, state: &mut CanonicalState) -> io::Result<Option<Hash>> {
        let entries = self.read_entries()?;
        let mut last_hash = None;
        for entry in entries {
            state
                .apply_delta(&entry.delta)
                .map_err(|err| io::Error::new(ErrorKind::InvalidData, err.to_string()))?;
            if state.root_hash() != entry.root_hash {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "root hash mismatch during replay",
                ));
            }
            last_hash = Some(entry.root_hash);
        }
        Ok(last_hash)
    }

    pub fn read_entries(&self) -> io::Result<Vec<TlogEntry>> {
        read_entries_from_path(&self.path)
    }
}

fn validate_manager_header(path: &Path) -> io::Result<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; CANON_TLOG_MAGIC.len()];
    reader.read_exact(&mut magic)?;
    if magic != CANON_TLOG_MAGIC {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "invalid canon tlog magic",
        ));
    }
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != CANON_TLOG_VERSION {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("unsupported canon tlog version {version}"),
        ));
    }
    Ok(())
}

fn read_entries_from_path(path: &Path) -> io::Result<Vec<TlogEntry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; CANON_TLOG_MAGIC.len()];
    reader.read_exact(&mut magic)?;
    if magic != CANON_TLOG_MAGIC {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "invalid canon tlog magic",
        ));
    }
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != CANON_TLOG_VERSION {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("unsupported canon tlog version {version}"),
        ));
    }
    let mut entries = Vec::new();
    loop {
        let mut len_buf = [0u8; 4];
        match reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(err),
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        let entry: TlogEntry = bincode::deserialize(&buf)
            .map_err(|err| io::Error::new(ErrorKind::InvalidData, err.to_string()))?;
        entries.push(entry);
    }
    Ok(entries)
}

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
