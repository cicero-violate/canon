use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct RootHeader {
    pub generation: u64,
    pub tree_size: u64,
    pub root_hash: [u8; 32],
}

impl RootHeader {
    pub fn is_valid(&self) -> bool {
        self.tree_size > 0 && self.generation > 0
    }
}
