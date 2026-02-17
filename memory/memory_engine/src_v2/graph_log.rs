//! Append-only log storing graph deltas for later materialization.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const MAGIC: &[u8] = b"GRAPHLOG1";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphDelta {
    #[serde(default)]
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphSnapshot {
    #[serde(default)]
    pub payload: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum GraphDeltaError {
    #[error("graph materialization failed: {0}")]
    Materialization(String),
}

pub struct GraphMaterializer;

impl GraphMaterializer {
    pub fn replay(entries: Vec<GraphDelta>) -> Result<GraphSnapshot, GraphDeltaError> {
        let mut snapshot = GraphSnapshot::default();
        for delta in entries {
            snapshot.payload.extend(delta.payload);
        }
        Ok(snapshot)
    }
}

#[derive(Debug)]
pub struct GraphDeltaLog {
    entries: RwLock<Vec<GraphDelta>>,
    writer: RwLock<Option<BufWriter<File>>>,
}

impl GraphDeltaLog {
    pub fn new(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;

        if file.metadata()?.len() == 0 {
            file.write_all(MAGIC)?;
        } else {
            validate_magic(&path)?;
        }

        let entries = load_entries(&path)?;

        Ok(Self {
            entries: RwLock::new(entries),
            writer: RwLock::new(Some(BufWriter::new(file))),
        })
    }

    pub fn append(&self, delta: GraphDelta) -> std::io::Result<()> {
        let blob = bincode::serialize(&delta)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut guard = self.writer.write();
        if let Some(writer) = guard.as_mut() {
            let writer: &mut BufWriter<File> = writer;
            writer.write_all(&(blob.len() as u32).to_le_bytes())?;
            writer.write_all(&blob)?;
            writer.flush()?;
        }
        self.entries.write().push(delta);
        Ok(())
    }

    pub fn entries(&self) -> Vec<GraphDelta> {
        self.entries.read().clone()
    }

    pub fn replay_snapshot(&self) -> Result<GraphSnapshot, GraphDeltaError> {
        let entries = self.entries.read();
        GraphMaterializer::replay(entries.clone())
    }
}

fn validate_magic(path: &PathBuf) -> std::io::Result<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; MAGIC.len()];
    reader.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid graph log magic",
        ));
    }
    Ok(())
}

fn load_entries(path: &PathBuf) -> std::io::Result<Vec<GraphDelta>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; MAGIC.len()];
    reader.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid graph log magic",
        ));
    }
    let mut entries = Vec::new();
    loop {
        let mut len_buf = [0u8; 4];
        match reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        let delta: GraphDelta = bincode::deserialize(&buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        entries.push(delta);
    }
    Ok(entries)
}
