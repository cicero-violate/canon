use crate::delta::{DeltaError, Source};
use crate::epoch::Epoch;
use crate::page::DeltaAppliable;
use crate::page::Page;
use crate::page::PageError;
use crate::primitives::{DeltaID, Hash, PageID};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize}; // Required for bincode
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)] // ← Added derives
pub struct Delta {
    pub delta_id: DeltaID,
    pub page_id: PageID,
    pub epoch: Epoch,
    pub mask: Vec<bool>,
    pub payload: Vec<u8>,
    pub is_sparse: bool,
    pub timestamp: u64,
    pub source: Source,
    pub intent_metadata: Option<String>,
}

impl Delta {
    pub fn new_dense(delta_id: DeltaID, page_id: PageID, epoch: Epoch, data: Vec<u8>, mask: Vec<bool>, source: Source) -> Result<Self, DeltaError> {
        if data.len() != mask.len() {
            return Err(DeltaError::SizeMismatch { mask_len: mask.len(), payload_len: data.len() });
        }

        Ok(Self { delta_id, page_id, epoch, mask, payload: data, is_sparse: false, timestamp: now_ns(), source, intent_metadata: None })
    }

    pub fn new_sparse(delta_id: DeltaID, page_id: PageID, epoch: Epoch, mask: Vec<bool>, payload: Vec<u8>, source: Source) -> Result<Self, DeltaError> {
        let changed = mask.iter().filter(|&&m| m).count();
        if changed != payload.len() {
            return Err(DeltaError::SizeMismatch { mask_len: mask.len(), payload_len: payload.len() });
        }

        Ok(Self { delta_id, page_id, epoch, mask, payload, is_sparse: true, timestamp: now_ns(), source, intent_metadata: None })
    }

    pub fn merge(&self, other: &Delta) -> Result<Delta, DeltaError> {
        if self.page_id != other.page_id {
            return Err(DeltaError::PageIDMismatch { expected: self.page_id, found: other.page_id });
        }

        if self.mask.len() != other.mask.len() {
            return Err(DeltaError::MaskSizeMismatch { expected: self.mask.len(), found: other.mask.len() });
        }

        let mut merged_mask = self.mask.clone();
        let mut merged_payload = self.to_dense();
        let other_dense = other.to_dense();

        for (idx, &flag) in other.mask.iter().enumerate() {
            if flag {
                merged_mask[idx] = true;
                merged_payload[idx] = other_dense[idx];
            }
        }

        Ok(Delta {
            delta_id: other.delta_id,
            page_id: other.page_id,
            epoch: Epoch(other.epoch.0.max(self.epoch.0)),
            mask: merged_mask,
            payload: merged_payload,
            is_sparse: false,
            timestamp: other.timestamp.max(self.timestamp),
            source: other.source.clone(),
            intent_metadata: other.intent_metadata.clone().or_else(|| self.intent_metadata.clone()),
        })
    }

    pub fn to_dense(&self) -> Vec<u8> {
        if !self.is_sparse {
            return self.payload.clone();
        }

        let mut dense = vec![0u8; self.mask.len()];
        let mut payload_idx = 0;
        for (idx, &flag) in self.mask.iter().enumerate() {
            if flag {
                dense[idx] = self.payload[payload_idx];
                payload_idx += 1;
            }
        }
        dense
    }

    pub fn apply_to(&self, page: &mut Page) -> Result<(), PageError> {
        if let Err(err) = super::delta_validation::validate_delta(self) {
            return Err(match err {
                DeltaError::SizeMismatch { .. } => PageError::MaskSizeMismatch,
                DeltaError::PageIDMismatch { .. } => PageError::PageIDMismatch,
                DeltaError::MaskSizeMismatch { .. } => PageError::MaskSizeMismatch,
            });
        }
        page.apply_delta(self)
    }
    pub fn hash(&self) -> Hash {
        let mut hasher = Sha256::new();
        hasher.update(&self.page_id.0.to_be_bytes());
        hasher.update(&self.epoch.0.to_be_bytes());
        hasher.update(self.mask.iter().map(|b| *b as u8).collect::<Vec<_>>());
        hasher.update(&self.payload);
        hasher.finalize().into()
    }
}

fn now_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64
}
