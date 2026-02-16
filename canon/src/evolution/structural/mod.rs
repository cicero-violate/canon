mod apply;
mod guard;
mod rename;

use super::EvolutionError;
use crate::ir::{CanonicalIr, Delta};

pub(super) fn apply_structural_delta(
    ir: &mut CanonicalIr,
    delta: &Delta,
) -> Result<(), EvolutionError> {
    if let Some(payload) = &delta.payload {
        apply::apply_delta_payload(ir, payload, &delta.id)?;
    }
    Ok(())
}
