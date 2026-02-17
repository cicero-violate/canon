use memmap2::{MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::path::Path;

pub struct MmapLog {
    mmap: MmapMut,
    len: usize,
}

impl MmapLog {
    pub fn open(path: &Path, size: usize) -> std::io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).create(true).open(path)?;
        file.set_len(size as u64)?;
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        Ok(Self { mmap, len: 0 })
    }

    pub fn append(&mut self, data: &[u8]) {
        let end = self.len + data.len();
        self.mmap[self.len..end].copy_from_slice(data);
        self.len = end;
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.mmap.flush()
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }
}
