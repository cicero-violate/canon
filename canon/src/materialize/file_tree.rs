use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileTree {
    directories: BTreeSet<String>,
    files: BTreeMap<String, FileEntry>,
}

impl FileTree {
    pub fn new() -> Self {
        Self {
            directories: BTreeSet::new(),
            files: BTreeMap::new(),
        }
    }

    pub fn add_directory(&mut self, path: impl Into<String>) {
        self.directories.insert(path.into());
    }

    pub fn add_file(&mut self, path: impl Into<String>, contents: impl Into<String>) {
        self.files
            .insert(path.into(), FileEntry::new(contents.into()));
    }

    pub fn directories(&self) -> &BTreeSet<String> {
        &self.directories
    }

    pub fn files(&self) -> &BTreeMap<String, FileEntry> {
        &self.files
    }
}

impl Default for FileTree {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    pub contents: String,
}

impl FileEntry {
    pub fn new(contents: String) -> Self {
        Self { contents }
    }
}
