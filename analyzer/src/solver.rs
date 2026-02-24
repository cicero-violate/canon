//! Phase 2: solve constraint graphs using topological order.
//!
//! Variables:
//!   order : Vec<usize>   — topological order of G_name
//!   V, E                 — vertex/edge counts from CsrGraph
//!
//! Equations:
//!   order = topo(G_name.row_ptr, G_name.col_idx)
//!   for v in order: propagate rename constraints through name_graph neighbours(v)

use anyhow::Result;
use model::ir::model_ir::ModelIR;
use algorithms::graph::topological_sort::topological_sort;

/// Phase 2: solve derived graphs and annotate ModelIR.
pub fn solve(ir: &mut ModelIR) -> Result<()> {
    let v = ir.name_graph.vertex_count();
    if v == 0 {
        return Ok(());
    }

    // Build adjacency list from name_graph CSR for topo sort.
    let adj: Vec<Vec<usize>> = (0..v)
        .map(|i| {
            ir.name_graph
                .neighbours(model::ir::node::NodeId(i as u32))
                .map(|(dst, _)| dst.index())
                .collect()
        })
        .collect();

    let _order = topological_sort(&adj);
    // Constraint propagation over _order will be implemented per-pass.

    Ok(())
}
