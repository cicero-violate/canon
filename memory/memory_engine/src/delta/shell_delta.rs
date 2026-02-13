use serde::{Deserialize, Serialize};
use serde_json;
use thiserror::Error;

use crate::delta::{Delta, DeltaError, Source};
use crate::epoch::Epoch;
use crate::primitives::{DeltaID, PageID};

const SHELL_DELTA_MAGIC: &[u8] = b"CANON_SHELL_DELTA";
const SHELL_DELTA_VERSION: u8 = 1;

/// Human-readable delta for shell command mutations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShellDelta {
    pub shell_id: u64,
    pub epoch: u64,
    pub command: String,
    pub state_hash: String,
}

#[derive(Debug, Error)]
pub enum ShellDeltaError {
    #[error("payload missing shell delta header")]
    MissingHeader,
    #[error("payload uses unsupported shell delta version {0}")]
    UnsupportedVersion(u8),
    #[error("shell delta serialization failed: {0}")]
    Serialization(String),
    #[error("delta encoding failed: {0}")]
    Delta(#[from] DeltaError),
}

impl From<serde_json::Error> for ShellDeltaError {
    fn from(err: serde_json::Error) -> Self {
        ShellDeltaError::Serialization(err.to_string())
    }
}

impl ShellDelta {
    pub fn new(
        shell_id: u64,
        epoch: u64,
        command: impl Into<String>,
        state_hash: impl Into<String>,
    ) -> Self {
        Self {
            shell_id,
            epoch,
            command: command.into(),
            state_hash: state_hash.into(),
        }
    }

    /// Convert this shell delta into a canonical [`Delta`] payload.
    pub fn into_delta(
        &self,
        delta_id: DeltaID,
        page_id: PageID,
        epoch: Epoch,
    ) -> Result<Delta, ShellDeltaError> {
        let payload = self.encode_payload()?;
        let mask = vec![true; payload.len()];
        Delta::new_dense(
            delta_id,
            page_id,
            epoch,
            payload,
            mask,
            Source(format!("canon.shell/{}", self.shell_id)),
        )
        .map_err(ShellDeltaError::from)
    }

    /// Attempt to decode a [`ShellDelta`] from a canonical [`Delta`].
    pub fn try_from_delta(delta: &Delta) -> Result<Option<ShellDelta>, ShellDeltaError> {
        match Self::decode_payload(&delta.payload)? {
            Some(shell) => Ok(Some(shell)),
            None => Ok(None),
        }
    }

    fn encode_payload(&self) -> Result<Vec<u8>, ShellDeltaError> {
        let mut payload = Vec::with_capacity(SHELL_DELTA_MAGIC.len() + 1);
        payload.extend_from_slice(SHELL_DELTA_MAGIC);
        payload.push(SHELL_DELTA_VERSION);
        let json = serde_json::to_vec(self)?;
        payload.extend_from_slice(&json);
        Ok(payload)
    }

    fn decode_payload(bytes: &[u8]) -> Result<Option<ShellDelta>, ShellDeltaError> {
        if bytes.len() < SHELL_DELTA_MAGIC.len() + 1 {
            return Ok(None);
        }
        if !bytes.starts_with(SHELL_DELTA_MAGIC) {
            return Ok(None);
        }
        let version = bytes[SHELL_DELTA_MAGIC.len()];
        if version != SHELL_DELTA_VERSION {
            return Err(ShellDeltaError::UnsupportedVersion(version));
        }
        let body = &bytes[SHELL_DELTA_MAGIC.len() + 1..];
        let shell: ShellDelta = serde_json::from_slice(body)?;
        Ok(Some(shell))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_shell_delta() {
        let shell = ShellDelta::new(42, 7, "echo hello", "abc123");
        let delta = shell
            .into_delta(DeltaID(1), PageID(1), Epoch(0))
            .expect("encode");
        let decoded = ShellDelta::try_from_delta(&delta)
            .expect("decode")
            .expect("shell payload");
        assert_eq!(decoded.shell_id, 42);
        assert_eq!(decoded.epoch, 7);
        assert_eq!(decoded.command, "echo hello");
        assert_eq!(decoded.state_hash, "abc123");
    }

    #[test]
    fn non_shell_delta_returns_none() {
        let payload = vec![1, 2, 3];
        let mask = vec![true; payload.len()];
        let delta = Delta::new_dense(
            DeltaID(99),
            PageID(1),
            Epoch(0),
            payload,
            mask,
            Source("test".into()),
        )
        .expect("delta");
        assert!(ShellDelta::try_from_delta(&delta)
            .expect("decode")
            .is_none());
    }
}
