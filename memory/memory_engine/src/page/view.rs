use super::PageLocation;
use crate::primitives::PageID;

#[derive(Debug, Clone)]
pub struct PageView {
    pub id: PageID,
    pub location: PageLocation,
    pub data: Vec<u8>,
}

impl PageView {
    pub fn data_slice(&self) -> &[u8] {
        &self.data
    }
}
