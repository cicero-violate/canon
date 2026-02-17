use super::{PageError, PageLocation};
use crate::delta::Delta;
use crate::epoch::Epoch;

/// Read/write access capability
pub trait PageAccess {
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
