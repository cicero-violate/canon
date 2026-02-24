//! Unsafe Soundness Solver (S15) — stub pending unsafe_ flag on IR nodes.
//!
//! Variables (future):
//!   U = { v | v.unsafe_ = true }   (Fn, Impl, Trait)
//!   safe_caller(u, v) <=> u calls v ∧ ¬v.unsafe_ ∧ ¬u.unsafe_block
//!
//! Equation (future):
//!   sound(v) <=> v ∈ U => all callers are in unsafe context
//!
//! Status: no-op until IR gap E12 (unsafe_ flag) is resolved.

use anyhow::Result;
use model::ir::model_ir::ModelIR;

pub fn solve(_ir: &ModelIR) -> Result<()> {
    // TODO(S15): implement when NodeKind gains unsafe_ bool (IR gap E12).
    Ok(())
}
