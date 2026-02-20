use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use algorithms::graph::csr::Csr;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Wire types (serializable, self-contained) ────────────────────────────────

/// Stable on-disk identity for a node (16-byte UUID-style key).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WireNodeId(pub [u8; 16]);

/// Stable on-disk identity for an edge.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WireEdgeId(pub [u8; 16]);

pub type NodeId = WireNodeId;
pub type GpuBfsResult = Vec<i32>;

impl WireNodeId {
    pub fn from_key(key: &str) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&hash.to_le_bytes());
        bytes[8..].copy_from_slice(&hash.to_be_bytes());
        Self(bytes)
    }
}

impl WireEdgeId {
    pub fn from_components(from: &WireNodeId, to: &WireNodeId, kind: &str) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        from.hash(&mut hasher);
        to.hash(&mut hasher);
        kind.hash(&mut hasher);
        let hash = hasher.finish();
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&hash.to_le_bytes());
        bytes[8..].copy_from_slice(&hash.to_be_bytes());
        Self(bytes)
    }
}

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

pub struct GraphSnapshot {
    pub nodes: Vec<WireNode>,
    pub edges: Vec<WireEdge>,
    csr_cache: Option<Csr>,
    node_index: Option<HashMap<WireNodeId, usize>>,
}

impl GraphSnapshot {
    pub fn new(nodes: Vec<WireNode>, edges: Vec<WireEdge>) -> Self {
        let (csr, index) = build_csr_cache(&nodes, &edges);
        Self { nodes, edges, csr_cache: Some(csr), node_index: Some(index) }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty()
    }

    pub fn csr(&mut self) -> &Csr {
        self.ensure_csr();
        self.csr_cache.as_ref().expect("csr cache initialized")
    }

    pub fn bfs_gpu(&mut self, start: NodeId) -> GpuBfsResult {
        self.ensure_csr();
        let Some(index_map) = self.node_index.as_ref() else {
            return Vec::new();
        };
        let Some(&start_idx) = index_map.get(&start) else {
            let len = self.csr_cache.as_ref().map(|c| c.vertex_count()).unwrap_or(0);
            return vec![-1; len];
        };

        #[cfg(feature = "cuda")]
        {
            return algorithms::graph::gpu::bfs_gpu(self.csr_cache.as_ref().unwrap(), start_idx);
        }

        #[cfg(not(feature = "cuda"))]
        {
            panic!("GraphSnapshot::bfs_gpu requires the `cuda` feature");
        }
    }

    fn ensure_csr(&mut self) {
        if self.csr_cache.is_some() && self.node_index.is_some() {
            return;
        }
        let (csr, index) = build_csr_cache(&self.nodes, &self.edges);
        self.csr_cache = Some(csr);
        self.node_index = Some(index);
    }
}

impl Default for GraphSnapshot {
    fn default() -> Self {
        Self { nodes: Vec::new(), edges: Vec::new(), csr_cache: None, node_index: None }
    }
}

impl Clone for GraphSnapshot {
    fn clone(&self) -> Self {
        Self { nodes: self.nodes.clone(), edges: self.edges.clone(), csr_cache: None, node_index: None }
    }
}

impl std::fmt::Debug for GraphSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphSnapshot")
            .field("nodes", &self.nodes)
            .field("edges", &self.edges)
            .field("csr_cache", &self.csr_cache.as_ref().map(|_| "<csr>"))
            .field("node_index", &self.node_index.as_ref().map(|m| m.len()))
            .finish()
    }
}

fn build_csr_cache(nodes: &[WireNode], edges: &[WireEdge]) -> (Csr, HashMap<WireNodeId, usize>) {
    let mut index = HashMap::with_capacity(nodes.len());
    for (idx, node) in nodes.iter().enumerate() {
        index.insert(node.id.clone(), idx);
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for edge in edges {
        let Some(&from) = index.get(&edge.from) else {
            continue;
        };
        let Some(&to) = index.get(&edge.to) else {
            continue;
        };
        adj[from].push(to);
    }

    (Csr::from_adj(&adj), index)
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
        let file = OpenOptions::new().create(true).append(true).read(false).open(&path)?;

        // Count existing entries by scanning the read side.
        let entry_count = Self::scan_count(&path)?;

        Ok(Self { path, writer: BufWriter::new(file), entry_count })
    }

    /// Append one delta to the log.
    pub fn append(&mut self, delta: &GraphDelta) -> Result<(), GraphDeltaError> {
        let bytes = bincode::serialize(delta).map_err(|e| GraphDeltaError::Encode(e.to_string()))?;
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
    pub fn replay_up_to(path: impl AsRef<Path>, limit: u64) -> Result<GraphSnapshot, GraphDeltaError> {
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

            let delta: GraphDelta = bincode::deserialize(&payload).map_err(|e| GraphDeltaError::Encode(e.to_string()))?;

            match delta {
                GraphDelta::AddNode(n) => nodes.push(n),
                GraphDelta::AddEdge(e) => edges.push(e),
            }

            count += 1;
        }

        let (csr, index) = build_csr_cache(&nodes, &edges);
        Ok(GraphSnapshot { nodes, edges, csr_cache: Some(csr), node_index: Some(index) })
    }

    /// Iterate over all deltas on disk, calling `f` for each.
    pub fn scan(path: impl AsRef<Path>, mut f: impl FnMut(u64, &GraphDelta)) -> Result<(), GraphDeltaError> {
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
            let delta: GraphDelta = bincode::deserialize(&payload).map_err(|e| GraphDeltaError::Encode(e.to_string()))?;
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
