use super::*;
use crate::primitives::PageID;

pub trait PageAllocatorLike {
    fn allocate(&self, id: PageID, size: usize) -> Result<(), PageError>;
}
