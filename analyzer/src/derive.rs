//! Phase 1: derive constraint graphs from the ModelIR node arena.
//!
//! Variables:
//!   ir.nodes          — flat arena, length V
//!   ir.module_graph   — G_module, built from Contains edges
//!   ir.call_graph     — G_call,   built from Calls edges in node kinds
//!   ir.name_graph     — G_name,   built from Renames/Resolves edges
//!
//! Equation:
//!   for each (src, dst, EdgeKind::Contains) in raw_edges -> module_graph
//!   for each (src, dst, EdgeKind::Calls)    in raw_edges -> call_graph
//!   for each (src, dst, EdgeKind::Renames)  in raw_edges -> name_graph

use anyhow::Result;
use model::ir::{
    model_ir::ModelIR,
    edge::EdgeKind,
    csr_graph::CsrGraph,
    node::NodeId,
};

/// Phase 1: walk the node arena and populate all five CSR graphs.
pub fn derive(ir: &mut ModelIR) -> Result<()> {
    let v = ir.nodes.len();

    // Collect raw edges from node kinds.
    // In a full implementation, capture_rustc fills these via push_node + explicit edge lists.
    // Here we prepare the infrastructure: empty graphs sized to V.
    let node_ids: Vec<NodeId> = (0..v as u32).map(NodeId).collect();

    // module_graph: Contains edges — placeholder, no edges yet.
    ir.module_graph = CsrGraph::from_edges(node_ids.clone(), vec![]);
    // call_graph
    ir.call_graph   = CsrGraph::from_edges(node_ids.clone(), vec![]);
    // name_graph
    ir.name_graph   = CsrGraph::from_edges(node_ids.clone(), vec![]);
    // type_graph
    ir.type_graph   = CsrGraph::from_edges(node_ids.clone(), vec![]);
    // cfg_graph
    ir.cfg_graph    = CsrGraph::from_edges(node_ids, vec![]);

    Ok(())
}
