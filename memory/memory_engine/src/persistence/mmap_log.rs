use memmap2::{MmapMut, MmapOptions};
use std::{
    fs::{File, OpenOptions},
    io::{Seek, SeekFrom},
    path::Path,
};

use super::root_header::RootHeader;
use crc32fast::Hasher;

const HEADER_SLOTS: usize = 2;
const HEADER_SIZE: usize = std::mem::size_of::<RootHeader>();
const HEADER_REGION: usize = HEADER_SLOTS * HEADER_SIZE;

pub struct MmapLog {
    file: File,
    mmap: MmapMut,
    write_offset: usize,
}

impl MmapLog {
    pub fn open(path: &Path, size: usize) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        let initial = size.max(HEADER_REGION);
        file.set_len(initial as u64)?;

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        Ok(Self {
            file,
            mmap,
            write_offset: HEADER_REGION,
        })
    }

    pub fn append(&mut self, data: &[u8]) -> std::io::Result<()> {
        // Frame = [len:u64][payload][crc:u32]
        let framed_len = 8 + data.len() + 4;
        let end = self.write_offset + framed_len;
        if end > self.mmap.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "WAL full; compaction required",
            ));
        }

        let len_bytes = (data.len() as u64).to_le_bytes();
        self.mmap[self.write_offset..self.write_offset + 8].copy_from_slice(&len_bytes);

        let payload_start = self.write_offset + 8;
        self.mmap[payload_start..payload_start + data.len()].copy_from_slice(data);

        // CRC32 over payload
        let mut hasher = Hasher::new();
        hasher.update(data);
        let crc = hasher.finalize().to_le_bytes();

        let crc_offset = payload_start + data.len();
        self.mmap[crc_offset..crc_offset + 4].copy_from_slice(&crc);

        self.mmap.flush_range(self.write_offset, framed_len)?;

        self.write_offset = end;
        Ok(())
    }

    pub fn write_root_header(&mut self, header: &RootHeader) -> std::io::Result<()> {
        // Enforce strict monotonic generation fencing
        if let Some(current) = self.read_latest_root()? {
            if header.generation <= current.generation {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "non-monotonic root header generation",
                ));
            }
        }

        let bytes = bytemuck::bytes_of(header);
        let slot = (header.generation as usize) % HEADER_SLOTS;
        let offset = slot * HEADER_SIZE;

        self.mmap[offset..offset + HEADER_SIZE].copy_from_slice(bytes);
        self.mmap.flush_range(offset, HEADER_SIZE)?;

        self.file.sync_data()?;
        Ok(())
    }

    pub fn read_latest_root(&self) -> std::io::Result<Option<RootHeader>> {
        let mut best: Option<RootHeader> = None;
        for i in 0..HEADER_SLOTS {
            let offset = i * HEADER_SIZE;
            let bytes = &self.mmap[offset..offset + HEADER_SIZE];
            let header: RootHeader = *bytemuck::from_bytes(bytes);
            if header.is_valid() {
                if best
                    .map(|b| header.generation > b.generation)
                    .unwrap_or(true)
                {
                    best = Some(header);
                }
            }
        }
        Ok(best)
    }

    pub fn scan_records(&self) -> Vec<Vec<u8>> {
        let mut records = Vec::new();
        let mut offset = HEADER_REGION;

        while offset + 8 <= self.mmap.len() {
            let len =
                u64::from_le_bytes(self.mmap[offset..offset + 8].try_into().unwrap()) as usize;

            let record_total = 8 + len + 4;

            if len == 0 || offset + record_total > self.mmap.len() {
                break;
            }

            let start = offset + 8;
            let end = start + len;

            let payload = &self.mmap[start..end];

            // Verify CRC
            let crc_offset = end;
            let stored_crc =
                u32::from_le_bytes(self.mmap[crc_offset..crc_offset + 4].try_into().unwrap());

            let mut hasher = Hasher::new();
            hasher.update(payload);
            let computed_crc = hasher.finalize();

            if stored_crc != computed_crc {
                break; // Stop at first corrupt record
            }

            records.push(payload.to_vec());
            offset += record_total;
        }

        records
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.mmap.flush()?;
        self.file.sync_all()
    }

    pub fn truncate(&mut self) -> std::io::Result<()> {
        self.file.set_len(HEADER_REGION as u64)?;
        self.file.sync_all()?;

        self.file.seek(SeekFrom::Start(0))?;
        self.mmap = unsafe { MmapOptions::new().map_mut(&self.file)? };
        self.write_offset = HEADER_REGION;
        Ok(())
    }
}
