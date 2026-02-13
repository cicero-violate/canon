use std::collections::HashSet;

use thiserror::Error;

use crate::{
    dot_import::{
        DotImportError, dot_graph_to_file_topology, dot_graph_to_imported_types,
        dot_graph_to_proposal, dot_graph_to_routing_hints, parse_dot,
    },
    ir::CanonicalIr,
    layout::{LayoutGraph, LayoutNode, apply_topology_to_layout},
    proposal::sanitize_identifier,
};

use super::{
    DSL_PREDICATE_ID, DSL_PROOF_ID, DSL_TICK_ID,
    accept::{AcceptProposalError, ProposalAcceptance, ProposalAcceptanceInput, accept_proposal},
    bootstrap::{ensure_dsl_predicate, ensure_dsl_proof, ensure_dsl_tick},
};

#[derive(Debug, Error)]
pub enum AutoAcceptDotError {
    #[error(transparent)]
    Import(#[from] DotImportError),
    #[error(transparent)]
    Accept(#[from] AcceptProposalError),
    #[error("dot graph contains no clusters")]
    Empty,
    #[error("unable to infer tick graph for bootstrap tick")]
    MissingTickGraph,
}

/// Parse a DOT source, bootstrap proof/predicate/tick if absent,
/// accept the generated proposal, then patch file topology and
/// imported_types onto the resulting IR — all in one call.
pub fn auto_accept_dot_proposal(
    ir: &CanonicalIr,
    layout: &LayoutGraph,
    dot_source: &str,
    goal: &str,
) -> Result<ProposalAcceptance, AutoAcceptDotError> {
    let graph = parse_dot(dot_source)?;
    if graph.clusters.is_empty() {
        return Err(AutoAcceptDotError::Empty);
    }

    let proposal = dot_graph_to_proposal(&graph, goal)?;
    let proposal_id = proposal.id.clone();
    let goal_slug = sanitize_identifier(goal);
    let judgment_id = format!("judgment.dot.{goal_slug}");
    let admission_id = format!("admission.dot.{goal_slug}");

    let mut working = ir.clone();
    ensure_dsl_proof(&mut working);
    ensure_dsl_predicate(&mut working);
    ensure_dsl_tick(&mut working).map_err(|_| AutoAcceptDotError::MissingTickGraph)?;
    working.proposals.push(proposal);

    let mut acceptance = accept_proposal(
        &working,
        layout,
        ProposalAcceptanceInput {
            proposal_id,
            proof_id: DSL_PROOF_ID.to_string(),
            predicate_id: DSL_PREDICATE_ID.to_string(),
            judgment_id,
            admission_id,
            tick_id: DSL_TICK_ID.to_string(),
            rationale: format!("Auto-accepted DOT proposal for goal `{goal}`."),
        },
    )?;

    let topology = dot_graph_to_file_topology(&graph);
    let routing_hints = dot_graph_to_routing_hints(&graph);

    // patch imported_types
    let type_map = dot_graph_to_imported_types(&graph);
    for edge in acceptance.ir.module_edges.iter_mut() {
        if let Some(types) = type_map.get(&(edge.source.clone(), edge.target.clone())) {
            edge.imported_types = types.clone();
        }
    }

    apply_topology_to_layout(&mut acceptance.layout, &acceptance.ir.modules, topology);
    apply_routing_hints(&mut acceptance.layout, &routing_hints);

    Ok(acceptance)
}

fn apply_routing_hints(
    layout: &mut LayoutGraph,
    hints: &std::collections::HashMap<String, String>,
) {
    if hints.is_empty() {
        return;
    }
    let mut known_files: HashSet<&str> = HashSet::new();
    for module in &layout.modules {
        for file in &module.files {
            known_files.insert(file.id.as_str());
        }
    }
    for assignment in &mut layout.routing {
        let node_id = layout_node_id(&assignment.node);
        if let Some(file_id) = hints.get(node_id) {
            if known_files.contains(file_id.as_str()) {
                assignment.file_id = file_id.clone();
                assignment.rationale = "LAY-005: DOT routing hint".to_owned();
            }
        }
    }
}

fn layout_node_id(node: &LayoutNode) -> &str {
    match node {
        LayoutNode::Struct(id) => id.as_str(),
        LayoutNode::Enum(id) => id.as_str(),
        LayoutNode::Trait(id) => id.as_str(),
        LayoutNode::Impl(id) => id.as_str(),
        LayoutNode::Function(id) => id.as_str(),
    }
}
