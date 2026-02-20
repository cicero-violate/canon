use bincode;
use database::delta::Delta;
use database::primitives::Hash;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const CANON_TLOG_MAGIC: &[u8] = b"CANON_TLOG1";
const CANON_TLOG_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlogEntry {
    pub delta: Delta,
    pub root_hash: Hash,
    pub proof_hash: Hash,
}

pub struct TlogManager {
    path: PathBuf,
    writer: Mutex<BufWriter<File>>,
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
        Ok(Self { path, writer: Mutex::new(BufWriter::new(file)) })
    }

    pub fn append(&self, delta: &Delta, root_hash: Hash, proof_hash: Hash) -> io::Result<()> {
        let entry = TlogEntry { delta: delta.clone(), root_hash, proof_hash };
        let blob = bincode::serialize(&entry).map_err(|err| io::Error::new(ErrorKind::Other, err.to_string()))?;
        let mut guard = self.writer.lock().expect("tlog writer poisoned");
        guard.write_all(&(blob.len() as u32).to_le_bytes())?;
        guard.write_all(&blob)?;
        guard.flush()?;
        Ok(())
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
        return Err(io::Error::new(ErrorKind::InvalidData, "invalid canon tlog magic"));
    }
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != CANON_TLOG_VERSION {
        return Err(io::Error::new(ErrorKind::InvalidData, format!("unsupported canon tlog version {version}")));
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
        return Err(io::Error::new(ErrorKind::InvalidData, "invalid canon tlog magic"));
    }
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != CANON_TLOG_VERSION {
        return Err(io::Error::new(ErrorKind::InvalidData, format!("unsupported canon tlog version {version}")));
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
        let entry: TlogEntry = bincode::deserialize(&buf).map_err(|err| io::Error::new(ErrorKind::InvalidData, err.to_string()))?;
        entries.push(entry);
    }
    Ok(entries)
}
