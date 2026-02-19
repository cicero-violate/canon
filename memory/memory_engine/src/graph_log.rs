use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Wire types (serializable, self-contained) ────────────────────────────────

/// Stable on-disk identity for a node (16-byte UUID-style key).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WireNodeId(pub [u8; 16]);

/// Stable on-disk identity for an edge.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WireEdgeId(pub [u8; 16]);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireNode {
    pub id: WireNodeId,
    pub key: String,
    pub label: String,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireEdge {
    pub id: WireEdgeId,
    pub from: WireNodeId,
    pub to: WireNodeId,
    pub kind: String,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphDelta {
    AddNode(WireNode),
    AddEdge(WireEdge),
}

// ── Snapshot (in-memory materialized view) ───────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct GraphSnapshot {
    pub nodes: Vec<WireNode>,
    pub edges: Vec<WireEdge>,
}

impl GraphSnapshot {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty()
    }
}

// ── Errors ───────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum GraphDeltaError {
    #[error("graph materialization failed: {0}")]
    Materialization(String),
    #[error("graph log io error: {0}")]
    Io(#[from] io::Error),
    #[error("graph log encode error: {0}")]
    Encode(String),
}

// ── GraphDeltaLog ─────────────────────────────────────────────────────────────

/// Append-only, length-prefixed bincode log of GraphDeltas.
///
/// On-disk format per entry:
///   [u32 le: byte length of payload] [bincode(GraphDelta)]
pub struct GraphDeltaLog {
    path: PathBuf,
    writer: BufWriter<File>,
    entry_count: u64,
}

impl GraphDeltaLog {
    /// Open or create the log at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GraphDeltaError> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(false)
            .open(&path)?;

        // Count existing entries by scanning the read side.
        let entry_count = Self::scan_count(&path)?;

        Ok(Self {
            path,
            writer: BufWriter::new(file),
            entry_count,
        })
    }

    /// Append one delta to the log.
    pub fn append(&mut self, delta: &GraphDelta) -> Result<(), GraphDeltaError> {
        let bytes = bincode::serialize(delta)
            .map_err(|e| GraphDeltaError::Encode(e.to_string()))?;
        let len = bytes.len() as u32;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&bytes)?;
        self.writer.flush()?;
        self.entry_count += 1;
        Ok(())
    }

    /// Total number of entries written so far.
    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }

    /// Replay all deltas from disk into a fresh materializer, returning the
    /// final snapshot.
    pub fn replay_all(path: impl AsRef<Path>) -> Result<GraphSnapshot, GraphDeltaError> {
        Self::replay_up_to(path, u64::MAX)
    }

    /// Replay the first `limit` deltas and return the snapshot at that point.
    pub fn replay_up_to(
        path: impl AsRef<Path>,
        limit: u64,
    ) -> Result<GraphSnapshot, GraphDeltaError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(GraphSnapshot::default());
        }

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut nodes: Vec<WireNode> = Vec::new();
        let mut edges: Vec<WireEdge> = Vec::new();
        let mut count = 0u64;

        loop {
            if count >= limit {
                break;
            }

            // Read 4-byte length prefix.
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(GraphDeltaError::Io(e)),
            }
            let len = u32::from_le_bytes(len_buf) as usize;

            let mut payload = vec![0u8; len];
            reader.read_exact(&mut payload)?;

            let delta: GraphDelta = bincode::deserialize(&payload)
                .map_err(|e| GraphDeltaError::Encode(e.to_string()))?;

            match delta {
                GraphDelta::AddNode(n) => nodes.push(n),
                GraphDelta::AddEdge(e) => edges.push(e),
            }

            count += 1;
        }

        Ok(GraphSnapshot { nodes, edges })
    }

    /// Iterate over all deltas on disk, calling `f` for each.
    pub fn scan(
        path: impl AsRef<Path>,
        mut f: impl FnMut(u64, &GraphDelta),
    ) -> Result<(), GraphDeltaError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(());
        }
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut idx = 0u64;

        loop {
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(GraphDeltaError::Io(e)),
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut payload = vec![0u8; len];
            reader.read_exact(&mut payload)?;
            let delta: GraphDelta = bincode::deserialize(&payload)
                .map_err(|e| GraphDeltaError::Encode(e.to_string()))?;
            f(idx, &delta);
            idx += 1;
        }
        Ok(())
    }

    fn scan_count(path: &Path) -> Result<u64, GraphDeltaError> {
        if !path.exists() {
            return Ok(0);
        }
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut count = 0u64;
        loop {
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(GraphDeltaError::Io(e)),
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            // Skip payload bytes.
            reader.seek(SeekFrom::Current(len as i64))?;
            count += 1;
        }
        Ok(count)
    }

    /// Path to the log file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}
