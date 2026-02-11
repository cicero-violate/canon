use super::page_impl::Page;
use super::*;
use crate::memory::primitives::PageID;

pub trait PageAllocatorLike {
    fn allocate(&self, id: PageID, size: usize) -> Result<(), PageError>;
    fn free(&self, id: PageID);
    fn get(&self, id: PageID) -> Option<&Page>;
}
