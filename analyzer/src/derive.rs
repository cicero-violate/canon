//! Phase 1 — derive all five CSR graphs from the ModelIR node arena.
//!
//! Variables:
//!   V           = ir.nodes.len()
//!   edge_hints  = explicit (src, dst, EdgeKind) triples from JSON
//!
//! Each hint is routed to the matching builder:
//!   Contains | ImplFor              -> module_graph
//!   Calls                           -> call_graph
//!   Renames | Resolves              -> name_graph
//!   TypeOf  | TypeUnifies           -> type_graph
//!   CfgEdge | CfgBranch             -> cfg_graph
//!
//! CFG edges are also auto-derived from Body::Blocks terminators.

use anyhow::Result;
use model::ir::{
    edge::EdgeKind,
    model_ir::ModelIR,
    node::{NodeId, NodeKind},
};
use crate::graph::{
    call_graph::CallGraphBuilder,
    cfg_graph::CfgGraphBuilder,
    module_graph::ModuleGraphBuilder,
    name_graph::NameGraphBuilder,
    type_graph::TypeGraphBuilder,
};

pub fn derive(ir: &mut ModelIR) -> Result<()> {
    let v = ir.nodes.len();

    let mut module_b = ModuleGraphBuilder::new(v);
    let mut call_b   = CallGraphBuilder::new(v);
    let mut name_b   = NameGraphBuilder::new(v);
    let mut type_b   = TypeGraphBuilder::new(v);
    let mut cfg_b    = CfgGraphBuilder::new(v);

    // Route edge_hints into builders.
    for hint in &ir.edge_hints {
        let src = NodeId(hint.src);
        let dst = NodeId(hint.dst);
        match &hint.kind {
            EdgeKind::Contains | EdgeKind::ImplFor => {
                module_b.add_contains(src, dst);
            }
            EdgeKind::Calls => {
                call_b.add_call(src, dst);
            }
            EdgeKind::Renames => {
                name_b.add_rename(src, dst);
            }
            EdgeKind::Resolves => {
                name_b.add_resolves(src, dst);
            }
            EdgeKind::TypeOf => {
                type_b.add_type_of(src, dst);
            }
            EdgeKind::TypeUnifies => {
                type_b.add_unifies(src, dst);
            }
            EdgeKind::CfgEdge => {
                cfg_b.add_cfg_edge(src, dst);
            }
            EdgeKind::CfgBranch { label } => {
                cfg_b.add_branch(src, dst, label.clone());
            }
        }
    }

    // Auto-derive CFG edges from Body::Blocks terminators.
    let nodes = ir.nodes.clone();
    for node in &nodes {
        match &node.kind {
            NodeKind::Function { body, .. } | NodeKind::Method { body, .. } => {
                cfg_b.add_from_body(node.id, body);
            }
            _ => {}
        }
    }

    ir.module_graph = module_b.build();
    ir.call_graph   = call_b.build();
    ir.name_graph   = name_b.build();
    ir.type_graph   = type_b.build();
    ir.cfg_graph    = cfg_b.build();

    Ok(())
}
