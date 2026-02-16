use super::*;
use crate::delta::Delta;
use crate::epoch::{Epoch, EpochCell};
use crate::primitives::PageID;

impl std::fmt::Debug for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Page")
            .field("id", &self.id)
            .field("epoch", &self.epoch.load())
            .field("len", &self.data.len())
            .field("location", &self.location)
            .finish()
    }
}

pub struct Page {
    id: PageID,
    epoch: EpochCell,
    data: Vec<u8>,
    location: PageLocation,
}

impl Page {
    pub fn new(id: PageID, size: usize, location: PageLocation) -> Result<Self, PageError> {
        if size == 0 {
            return Err(PageError::InvalidSize(size));
        }

        Ok(Self {
            id,
            epoch: EpochCell::new(0),
            data: vec![0u8; size],
            location,
        })
    }
}

impl PageAccess for Page {
    fn id(&self) -> PageID {
        self.id
    }
    fn size(&self) -> usize {
        self.data.len()
    }
    fn location(&self) -> PageLocation {
        self.location
    }

    fn epoch(&self) -> Epoch {
        self.epoch.load()
    }
    fn set_epoch(&self, epoch: Epoch) {
        self.epoch.store(epoch);
    }

    fn data_slice(&self) -> &[u8] {
        &self.data
    }

    fn data_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

impl DeltaAppliable for Page {
    fn apply_delta(&mut self, delta: &Delta) -> Result<(), PageError> {
        if delta.page_id != self.id {
            return Err(PageError::PageIDMismatch);
        }

        let data_slice = self.data_mut_slice();

        if delta.mask.len() != data_slice.len() {
            return Err(PageError::MaskSizeMismatch);
        }

        let dense = delta.to_dense();

        for (i, &mask_bit) in delta.mask.iter().enumerate() {
            if mask_bit {
                data_slice[i] = dense[i];
            }
        }

        self.set_epoch(delta.epoch);
        Ok(())
    }
}

impl PageViewProvider for Page {
    fn view(&self) -> PageView {
        PageView {
            id: self.id,
            location: self.location,
            data: self.data.clone(),
        }
    }
}
