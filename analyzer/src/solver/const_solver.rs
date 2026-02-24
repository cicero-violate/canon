//! Const Evaluation Solver (S11) — stub pending NodeKind::Const/Static.
//!
//! Variables (future):
//!   C = { v | NodeKind::Const { value } }
//!   eval(v) = constant-fold value string
//!
//! Equation (future):
//!   eval(c) = interpret(c.value) ∈ ℤ ∪ ℝ ∪ bool ∪ str
//!
//! Status: structural shell complete. G_value is now wired.
//!         Activates when NodeKind::Const/Static added (IR gap E5).

use anyhow::Result;
use model::ir::model_ir::ModelIR;

pub fn solve(_ir: &ModelIR) -> Result<()> {
    // G_value = ir.value_graph (ConstDep edges).
    // Topo-sort G_value => evaluation order; cycle => error.
    // TODO(S11): implement when NodeKind::Const/Static land (IR gap E5).
    Ok(())
}
