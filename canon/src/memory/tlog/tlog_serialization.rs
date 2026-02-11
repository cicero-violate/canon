use crate::memory::delta::{Delta, Source};
use crate::memory::epoch::Epoch;
use crate::memory::primitives::{DeltaID, PageID};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

pub fn read_log(path: impl AsRef<Path>) -> std::io::Result<Vec<Delta>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != b"MMSBLOG1" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid magic",
        ));
    }
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);

    let mut deltas = Vec::new();
    loop {
        let mut delta_id = [0u8; 8];
        if reader.read_exact(&mut delta_id).is_err() {
            break;
        }
        let mut page_id = [0u8; 8];
        reader.read_exact(&mut page_id)?;
        let mut epoch = [0u8; 4];
        reader.read_exact(&mut epoch)?;

        let mut mask_len_bytes = [0u8; 4];
        reader.read_exact(&mut mask_len_bytes)?;
        let mask_len = u32::from_le_bytes(mask_len_bytes) as usize;
        let mut mask_bytes = vec![0u8; mask_len];
        reader.read_exact(&mut mask_bytes)?;
        let mask = mask_bytes.iter().map(|b| *b != 0).collect();

        let mut payload_len_bytes = [0u8; 4];
        reader.read_exact(&mut payload_len_bytes)?;
        let payload_len = u32::from_le_bytes(payload_len_bytes) as usize;
        let mut payload = vec![0u8; payload_len];
        reader.read_exact(&mut payload)?;

        let mut sparse_flag = [0u8; 1];
        reader.read_exact(&mut sparse_flag)?;
        let is_sparse = sparse_flag[0] != 0;

        let mut timestamp_bytes = [0u8; 8];
        reader.read_exact(&mut timestamp_bytes)?;
        let timestamp = u64::from_le_bytes(timestamp_bytes);

        let mut source_len_bytes = [0u8; 4];
        reader.read_exact(&mut source_len_bytes)?;
        let source_len = u32::from_le_bytes(source_len_bytes) as usize;
        let mut source_buf = vec![0u8; source_len];
        reader.read_exact(&mut source_buf)?;
        let source = Source(String::from_utf8_lossy(&source_buf).to_string());

        let intent_metadata = if version >= 2 {
            let mut metadata_len_bytes = [0u8; 4];
            if reader.read_exact(&mut metadata_len_bytes).is_err() {
                break;
            }
            let metadata_len = u32::from_le_bytes(metadata_len_bytes) as usize;
            if metadata_len == 0 {
                None
            } else {
                let mut metadata_buf = vec![0u8; metadata_len];
                reader.read_exact(&mut metadata_buf)?;
                Some(String::from_utf8_lossy(&metadata_buf).to_string())
            }
        } else {
            None
        };

        deltas.push(Delta {
            delta_id: DeltaID(u64::from_le_bytes(delta_id)),
            page_id: PageID(u64::from_le_bytes(page_id)),
            epoch: Epoch(u32::from_le_bytes(epoch)),
            mask,
            payload,
            is_sparse,
            timestamp,
            source,
            intent_metadata,
        });
    }

    Ok(deltas)
}
