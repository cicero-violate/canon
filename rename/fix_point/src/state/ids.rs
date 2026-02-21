impl NodeId {
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
    pub fn as_bytes(&self) -> [u8; 16] {
        self.0
    }
    pub fn from_key(key: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&hash.to_le_bytes());
        bytes[8..].copy_from_slice(&hash.to_be_bytes());
        Self(bytes)
    }
    pub fn low_u64_le(&self) -> u64 {
        u64::from_le_bytes(self.0[..8].try_into().unwrap())
    }
}


impl NodeId {
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
    pub fn as_bytes(&self) -> [u8; 16] {
        self.0
    }
    pub fn from_key(key: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&hash.to_le_bytes());
        bytes[8..].copy_from_slice(&hash.to_be_bytes());
        Self(bytes)
    }
    pub fn low_u64_le(&self) -> u64 {
        u64::from_le_bytes(self.0[..8].try_into().unwrap())
    }
}


impl EdgeId {
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
    pub fn as_bytes(&self) -> [u8; 16] {
        self.0
    }
    pub fn from_components(from: &NodeId, to: &NodeId, kind: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        from.hash(&mut hasher);
        to.hash(&mut hasher);
        kind.hash(&mut hasher);
        let hash = hasher.finish();
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&hash.to_le_bytes());
        bytes[8..].copy_from_slice(&hash.to_be_bytes());
        Self(bytes)
    }
}


impl EdgeId {
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
    pub fn as_bytes(&self) -> [u8; 16] {
        self.0
    }
    pub fn from_components(from: &NodeId, to: &NodeId, kind: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        from.hash(&mut hasher);
        to.hash(&mut hasher);
        kind.hash(&mut hasher);
        let hash = hasher.finish();
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&hash.to_le_bytes());
        bytes[8..].copy_from_slice(&hash.to_be_bytes());
        Self(bytes)
    }
}


impl NodeId {
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
    pub fn as_bytes(&self) -> [u8; 16] {
        self.0
    }
    pub fn from_key(key: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&hash.to_le_bytes());
        bytes[8..].copy_from_slice(&hash.to_be_bytes());
        Self(bytes)
    }
    pub fn low_u64_le(&self) -> u64 {
        u64::from_le_bytes(self.0[..8].try_into().unwrap())
    }
}


impl NodeId {
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
    pub fn as_bytes(&self) -> [u8; 16] {
        self.0
    }
    pub fn from_key(key: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&hash.to_le_bytes());
        bytes[8..].copy_from_slice(&hash.to_be_bytes());
        Self(bytes)
    }
    pub fn low_u64_le(&self) -> u64 {
        u64::from_le_bytes(self.0[..8].try_into().unwrap())
    }
}


impl EdgeId {
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
    pub fn as_bytes(&self) -> [u8; 16] {
        self.0
    }
    pub fn from_components(from: &NodeId, to: &NodeId, kind: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        from.hash(&mut hasher);
        to.hash(&mut hasher);
        kind.hash(&mut hasher);
        let hash = hasher.finish();
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&hash.to_le_bytes());
        bytes[8..].copy_from_slice(&hash.to_be_bytes());
        Self(bytes)
    }
}


impl EdgeId {
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
    pub fn as_bytes(&self) -> [u8; 16] {
        self.0
    }
    pub fn from_components(from: &NodeId, to: &NodeId, kind: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        from.hash(&mut hasher);
        to.hash(&mut hasher);
        kind.hash(&mut hasher);
        let hash = hasher.finish();
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&hash.to_le_bytes());
        bytes[8..].copy_from_slice(&hash.to_be_bytes());
        Self(bytes)
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId([u8; 16]);
