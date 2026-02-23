use std::path::{Path, PathBuf};


#[derive(Debug, Clone)]
pub enum LinuxFact {
    Exists(PathBuf),
    File(PathBuf),
    Dir(PathBuf),
    ProcessRunning(String),
    BinaryInstalled(String),
}
