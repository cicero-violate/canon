use std::fs::{File, OpenOptions};
use std::io::{Write, Read, Seek, SeekFrom};
use std::path::Path;

pub struct Journal {
    file: File,
}

impl Journal {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(path)?;

        Ok(Self { file })
    }

    pub fn begin(&mut self, tx_id: u64) -> std::io::Result<()> {
        writeln!(self.file, "BEGIN {}", tx_id)?;
        self.file.flush()
    }

    pub fn commit(&mut self, tx_id: u64) -> std::io::Result<()> {
        writeln!(self.file, "COMMIT {}", tx_id)?;
        self.file.flush()
    }

    pub fn recover(&mut self) -> std::io::Result<Vec<u64>> {
        self.file.seek(SeekFrom::Start(0))?;

        let mut contents = String::new();
        self.file.read_to_string(&mut contents)?;

        let mut open = std::collections::HashSet::new();

        for line in contents.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                continue;
            }

            let tx_id: u64 = parts[1].parse().unwrap_or(0);

            match parts[0] {
                "BEGIN" => { open.insert(tx_id); }
                "COMMIT" => { open.remove(&tx_id); }
                _ => {}
            }
        }

        Ok(open.into_iter().collect())
    }
}
