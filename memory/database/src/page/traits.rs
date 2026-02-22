use super::{PageError, PageLocation};
// Delta no longer imported here to avoid cyclic dependency.

// ApplyDelta trait removed to eliminate page ↔ delta dependency.
use crate::epoch::epoch_types::Epoch;

/// Read/write access capability
pub trait PageAccess {
    fn location(&self) -> PageLocation;

    fn epoch(&self) -> Epoch;
    fn set_epoch(&self, epoch: Epoch);

    fn data_slice(&self) -> &[u8];
    fn data_mut_slice(&mut self) -> &mut [u8];
}

// DeltaAppliable removed.
// Page layer owns delta application directly.
// Break page ↔ delta cycle:
// Page layer must not depend on concrete Delta type.
// Delta application stays in delta layer.
