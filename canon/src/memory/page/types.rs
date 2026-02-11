use thiserror::Error;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageLocation {
    Cpu = 0,
    Gpu = 1,
    Unified = 2,
}

impl PageLocation {
    pub fn from_tag(tag: i32) -> Result<Self, PageError> {
        match tag {
            0 => Ok(PageLocation::Cpu),
            1 => Ok(PageLocation::Gpu),
            2 => Ok(PageLocation::Unified),
            _ => Err(PageError::InvalidLocation(tag)),
        }
    }
}

#[derive(Debug, Error)]
pub enum PageError {
    #[error("Invalid page size: {0}")]
    InvalidSize(usize),

    #[error("Invalid location tag: {0}")]
    InvalidLocation(i32),

    #[error("Mask/payload size mismatch")]
    MaskSizeMismatch,

    #[error("PageID mismatch")]
    PageIDMismatch,

    #[error("Allocation failed")]
    AllocationFailed,

    #[error("Metadata decode error")]
    MetadataDecode(String),

    #[error("Page not found: {0:?}")]
    PageNotFound(crate::memory::primitives::PageID),
}

/// Configuration for page allocator
#[derive(Debug, Clone)]
pub struct PageAllocatorConfig {
    pub default_location: PageLocation,
    pub initial_capacity: usize,
}

impl Default for PageAllocatorConfig {
    fn default() -> Self {
        Self {
            default_location: PageLocation::Cpu,
            initial_capacity: 1024,
        }
    }
}

/// Snapshot data for checkpoint/restore
#[derive(Debug, Clone)]
pub struct PageSnapshotData {
    pub page_id: crate::memory::primitives::PageID,
    pub data: Vec<u8>,
    pub location: PageLocation,
    pub epoch: crate::memory::epoch::Epoch,
}
