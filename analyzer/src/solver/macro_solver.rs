//! Macro Expansion Solver (S12) — stub pending NodeKind::MacroCall.
//!
//! Variables (future):
//!   M = { v | NodeKind::MacroCall { tokens } }
//!   expand(v) = parse(v.tokens) -> Vec<NodeKind>
//!
//! Equation (future):
//!   IR' = IR ∪ ⋃ expand(m) for m ∈ M
//!
//! Status: structural shell complete. G_macro is now wired.
//!         Activates when NodeKind::MacroCall added (IR gap E14).

use anyhow::Result;
use model::ir::model_ir::ModelIR;

pub fn solve(_ir: &mut ModelIR) -> Result<()> {
    // G_macro = ir.macro_graph (Expands edges).
    // DFS on G_macro => expansion order; cycle => recursive macro error.
    // TODO(S12): implement when NodeKind::MacroCall lands (IR gap E14).
    Ok(())
}
