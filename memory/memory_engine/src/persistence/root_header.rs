use std::fs::File;
use std::io::{Read, Write, Seek, SeekFrom};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RootHeader {
    pub root: [u8; 32],
    pub tree_size: u64,
    pub epoch: u64,
    pub checksum: u64,
}

impl RootHeader {
    pub fn compute_checksum(&self) -> u64 {
        self.root.iter().fold(0u64, |acc, b| acc.wrapping_add(*b as u64))
            ^ self.tree_size
            ^ self.epoch
    }

    pub fn write_double_buffer(file: &mut File, h: &RootHeader) -> std::io::Result<()> {
        let mut header = *h;
        header.checksum = header.compute_checksum();

        file.seek(SeekFrom::Start(0))?;
        file.write_all(bytemuck::bytes_of(&header))?;
        file.write_all(bytemuck::bytes_of(&header))?;
        file.flush()?;
        Ok(())
    }

    pub fn read_double_buffer(file: &mut File) -> std::io::Result<Option<Self>> {
        file.seek(SeekFrom::Start(0))?;
        let mut buf = [0u8; std::mem::size_of::<RootHeader>()];
        file.read_exact(&mut buf)?;
        let h: RootHeader = *bytemuck::from_bytes(&buf);
        if h.compute_checksum() == h.checksum {
            return Ok(Some(h));
        }
        Ok(None)
    }
}
