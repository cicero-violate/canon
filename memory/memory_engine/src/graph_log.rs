use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphDelta {
    #[serde(default)]
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphSnapshot {
    #[serde(default)]
    pub payload: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum GraphDeltaError {
    #[error("graph materialization failed: {0}")]
    Materialization(String),
}
