//! Solver — Phase 2 of the analysis pipeline.
//!
//! Variables:
//!   ir : &mut ModelIR   — mutated in place by each sub-solver
//!
//! Pipeline:
//!   solve(ir)
//!     -> name_solver::solve(ir)    — topo order + rename propagation
//!     -> type_solver::solve(ir)    — SCC-based unification
//!     -> call_solver::solve(ir)    — DFS reachability on call graph
//!     -> module_solver::solve(ir)  — topo order containment validation
//!     -> cfg_solver::solve(ir)     — dominators / unreachable block removal

use anyhow::Result;
use model::ir::model_ir::ModelIR;

pub mod call_solver;
pub mod cfg_solver;
pub mod module_solver;
pub mod name_solver;
pub mod type_solver;
pub mod use_solver;

/// Run all solvers in dependency order.
pub fn solve(ir: &mut ModelIR) -> Result<()> {
    module_solver::solve(ir)?;   // containment must be settled first
    name_solver::solve(ir)?;     // rename constraints depend on module order
    type_solver::solve(ir)?;     // type unification depends on resolved names
    call_solver::solve(ir)?;     // call reachability depends on resolved types
    cfg_solver::solve(ir)?;      // CFG dominators depend on call resolution
    use_solver::solve(ir)?;      // inject Use nodes after all names are resolved
    Ok(())
}

// ── shared helper ────────────────────────────────────────────────────────────

/// Build a plain adjacency list from any CsrGraph for algorithm crate consumption.
///
/// Variables:
///   V = graph.vertex_count()
///   adj[v] = [dst.index() for (dst, _) in graph.neighbours(v)]
///
/// Equation:
///   adj[v] = col_idx[ row_ptr[v] .. row_ptr[v+1] ]  (cast to usize)
pub(crate) fn csr_to_adj<ND, ED>(graph: &model::ir::csr_graph::CsrGraph<ND, ED>) -> Vec<Vec<usize>> {
    let v = graph.vertex_count();
    (0..v)
        .map(|i| {
            graph
                .neighbours(model::ir::node::NodeId(i as u32))
                .map(|(dst, _)| dst.index())
                .collect()
        })
        .collect()
}
