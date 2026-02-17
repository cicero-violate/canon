use memmap2::{MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::path::Path;

const PAGE_SIZE: usize = 4096;

#[derive(Debug)]
pub struct PageStore {
    mmap: Option<MmapMut>,
    in_memory: Vec<u8>,
}

impl PageStore {
    pub fn open(path: &Path, max_pages: u64) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(path)?;

        file.set_len(max_pages * PAGE_SIZE as u64)?;

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        Ok(Self {
            mmap: Some(mmap),
            in_memory: Vec::new(),
        })
    }

    pub fn in_memory() -> Self {
        Self {
            mmap: None,
            in_memory: vec![0u8; PAGE_SIZE * 1024],
        }
    }

    pub fn write_page(&mut self, page_id: u64, data: &[u8]) {
        let offset = page_id as usize * PAGE_SIZE;
        let len = data.len().min(PAGE_SIZE);

        if let Some(mmap) = self.mmap.as_mut() {
            assert!(offset + PAGE_SIZE <= mmap.len(), "mmap capacity exceeded");
            let page = &mut mmap[offset..offset + PAGE_SIZE];
            page[..len].copy_from_slice(&data[..len]);
        } else {
            if offset + PAGE_SIZE > self.in_memory.len() {
                self.in_memory.resize(offset + PAGE_SIZE, 0);
            }
            let page = &mut self.in_memory[offset..offset + PAGE_SIZE];
            page[..len].copy_from_slice(&data[..len]);
        }
    }

    pub fn read_page(&self, page_id: u64) -> &[u8] {
        let offset = page_id as usize * PAGE_SIZE;
        if let Some(mmap) = &self.mmap {
            &mmap[offset..offset + PAGE_SIZE]
        } else {
            &self.in_memory[offset..offset + PAGE_SIZE]
        }
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        if let Some(mmap) = self.mmap.as_mut() {
            mmap.flush()
        } else {
            Ok(())
        }
    }
}
