//! Pattern Exhaustiveness Solver (S13) — stub pending NodeKind::Enum.
//!
//! Variables (future):
//!   E  = { v | NodeKind::Enum { variants } }
//!   P  = patterns in match arms (from Body::Blocks stmts)
//!   covered(e, P) <=> ∀ variant ∈ e.variants: ∃ p ∈ P matching variant
//!
//! Equation (future):
//!   exhaustive(match) <=> covered(enum, arms) ∨ has_wildcard(arms)
//!
//! Status: no-op until IR gap E6 (NodeKind::Enum) is resolved.

use anyhow::Result;
use model::ir::model_ir::ModelIR;

pub fn solve(_ir: &ModelIR) -> Result<()> {
    // TODO(S13): implement when NodeKind::Enum is added (IR gap E6).
    Ok(())
}
