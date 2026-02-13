use memory_engine::delta::Delta;
use memory_engine::epoch::Epoch;
use memory_engine::primitives::{DeltaID, PageID};

use crate::runtime::value::DeltaValue;

use super::{SystemInterpreter, SystemInterpreterError};

impl<'a> SystemInterpreter<'a> {
    pub(super) fn materialize_delta(
        &self,
        value: &DeltaValue,
        sequence: u64,
    ) -> Result<Delta, SystemInterpreterError> {
        let payload = value.payload_hash.as_bytes().to_vec();
        let mask = vec![true; payload.len()];
        let page_id = PageID(sequence.saturating_add(1));
        let parsed_id = value
            .delta_id
            .parse::<u64>()
            .unwrap_or(sequence.saturating_add(1));
        Delta::new_dense(
            DeltaID(parsed_id),
            page_id,
            Epoch(0),
            payload,
            mask,
            memory_engine::delta::Source(format!("system_interpreter:{}", value.delta_id)),
        )
        .map_err(|err| SystemInterpreterError::DeltaMaterialization(err.to_string()))
    }
}
