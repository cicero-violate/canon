use super::*;
use crate::epoch::epoch_types::{Epoch, EpochCell};
use crate::primitives::PageID;

impl std::fmt::Debug for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Page").field("id", &self.id).field("epoch", &self.epoch.load()).field("len", &self.data.len()).field("location", &self.location).finish()
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

        Ok(Self { id, epoch: EpochCell::new(0), data: vec![0u8; size], location })
    }
}

impl PageAccess for Page {
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

// DeltaAppliable impl removed.
// Page owns mutation logic directly via inherent methods.
// Delta application removed from page layer.
// Page is now pure storage + epoch carrier.
