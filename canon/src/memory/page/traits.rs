use super::view::PageView;
use super::{PageError, PageLocation};
use crate::memory::delta::Delta;
use crate::memory::epoch::Epoch;
use crate::memory::primitives::PageID;

/// Read/write access capability
pub trait PageAccess {
    fn id(&self) -> PageID;
    fn size(&self) -> usize;
    fn location(&self) -> PageLocation;

    fn epoch(&self) -> Epoch;
    fn set_epoch(&self, epoch: Epoch);

    fn data_slice(&self) -> &[u8];
    fn data_mut_slice(&mut self) -> &mut [u8];
}

/// Delta mutation capability
pub trait DeltaAppliable {
    fn apply_delta(&mut self, delta: &Delta) -> Result<(), PageError>;
}

/// Authority-gated view minting
pub trait PageViewProvider {
    fn view(&self) -> PageView;
}
