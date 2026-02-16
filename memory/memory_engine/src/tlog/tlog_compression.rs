use crate::delta::Delta;

/// RLE (Run-Length Encoding) for sparse masks
fn encode_rle(mask: &[bool]) -> Vec<u8> {
    let mut encoded = Vec::new();
    if mask.is_empty() {
        return encoded;
    }

    let mut current = mask[0];
    let mut count: u8 = 1;

    for &bit in &mask[1..] {
        if bit == current && count < 255 {
            count += 1;
        } else {
            encoded.push(if current { count } else { count | 0x80 });
            current = bit;
            count = 1;
        }
    }
    encoded.push(if current { count } else { count | 0x80 });
    encoded
}


/// Bitpack dense boolean masks
fn bitpack_mask(mask: &[bool]) -> Vec<u8> {
    let num_bytes = (mask.len() + 7) / 8;
    let mut packed = vec![0u8; num_bytes];

    for (i, &bit) in mask.iter().enumerate() {
        if bit {
            packed[i / 8] |= 1 << (i % 8);
        }
    }
    packed
}



#[derive(Debug, Clone, Copy)]
pub enum CompressionMode {
    None,
    Rle,
    Bitpack,
}

pub struct CompressionStats {
    pub original_size: usize,
    pub compressed_size: usize,
    pub ratio: f64,
}

pub fn compress_delta_mask(mask: &[bool], mode: CompressionMode) -> (Vec<u8>, CompressionStats) {
    let original = mask.len();

    let compressed = match mode {
        CompressionMode::None => mask.iter().map(|&b| b as u8).collect(),
        CompressionMode::Rle => encode_rle(mask),
        CompressionMode::Bitpack => bitpack_mask(mask),
    };

    let stats = CompressionStats {
        original_size: original,
        compressed_size: compressed.len(),
        ratio: original as f64 / compressed.len().max(1) as f64,
    };

    (compressed, stats)
}

pub fn compact(deltas: &[Delta]) -> Vec<Delta> {
    if deltas.len() <= 1 {
        return deltas.to_vec();
    }
    let mut result = Vec::with_capacity(deltas.len());
    let mut iter = deltas.iter();
    if let Some(first) = iter.next() {
        result.push(first.clone());
        for delta in iter {
            if let Some(last) = result.last_mut() {
                if last.page_id == delta.page_id {
                    if let Ok(merged) = last.merge(delta) {
                        *last = merged;
                        continue;
                    }
                }
            }
            result.push(delta.clone());
        }
    }
    result
}
