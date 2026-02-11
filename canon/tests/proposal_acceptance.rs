use canon::{
    CanonicalIr, ProposalAcceptanceInput, accept_proposal,
    ir::{
        Proposal, ProposalGoal, ProposalStatus, ProposedApi, ProposedEdge, ProposedNode,
        ProposedNodeKind, Word,
    },
};
use std::path::PathBuf;

fn load_fixture() -> CanonicalIr {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("valid_ir.json");
    let data = std::fs::read(path).expect("fixture");
    serde_json::from_slice(&data).expect("canonical ir")
}

fn make_node(
    id: Option<&str>,
    name: &str,
    module: Option<&str>,
    kind: ProposedNodeKind,
) -> ProposedNode {
    ProposedNode {
        id: id.map(|value| value.to_string()),
        name: Word::new(name).expect("word"),
        module: module.map(|m| m.to_string()),
        kind,
    }
}

fn make_proposal() -> Proposal {
    Proposal {
        id: "proposal.future".to_string(),
        goal: ProposalGoal {
            id: Word::new("FutureGoal").expect("word"),
            description: "Advance Canon structure.".to_string(),
        },
        nodes: vec![
            make_node(
                Some("module.future"),
                "Future",
                None,
                ProposedNodeKind::Module,
            ),
            make_node(
                Some("struct.future_state"),
                "FutureState",
                Some("module.future"),
                ProposedNodeKind::Struct,
            ),
            make_node(
                Some("trait.future_ops"),
                "FutureOps",
                Some("module.future"),
                ProposedNodeKind::Trait,
            ),
        ],
        apis: vec![ProposedApi {
            trait_id: "trait.future_ops".to_string(),
            functions: vec!["trait_fn.compute_future".to_string()],
        }],
        edges: vec![ProposedEdge {
            from: "module.future".to_string(),
            to: "module.core".to_string(),
            rationale: "Future interacts with core.".to_string(),
        }],
        status: ProposalStatus::Submitted,
    }
}

#[test]
fn accept_proposal_generates_structural_artifacts() {
    let mut ir = load_fixture();
    ir.proposals.push(make_proposal());
    let predicate = ir
        .judgment_predicates
        .first()
        .expect("predicate")
        .id
        .clone();
    let tick = ir.ticks.first().expect("tick").id.clone();
    let acceptance = accept_proposal(
        &ir,
        ProposalAcceptanceInput {
            proposal_id: "proposal.future".to_string(),
            proof_id: "proof.law".to_string(),
            predicate_id: predicate,
            judgment_id: "judgment.future".to_string(),
            admission_id: "admission.future".to_string(),
            tick_id: tick,
            rationale: "Future structure accepted.".to_string(),
        },
    )
    .expect("accepted");
    assert_eq!(acceptance.delta_ids.len(), 7);

    let next = acceptance.ir;
    let proposal = next
        .proposals
        .iter()
        .find(|p| p.id == "proposal.future")
        .expect("proposal");
    assert_eq!(proposal.status, ProposalStatus::Accepted);
    assert!(next.modules.iter().any(|m| m.id == "module.future"));
    assert!(next.structs.iter().any(|s| s.id == "struct.future_state"));
    assert!(next.traits.iter().any(|t| t.id == "trait.future_ops"));
    assert!(next
        .impl_blocks
        .iter()
        .any(|blk| blk.struct_id == "struct.future_state" && blk.trait_id == "trait.future_ops"));
    assert!(
        next.functions
            .iter()
            .any(|f| f.trait_function == "trait_fn.compute_future")
    );
    assert!(
        next.module_edges
            .iter()
            .any(|edge| edge.source == "module.future" && edge.target == "module.core")
    );
    assert!(
        next.judgments
            .iter()
            .any(|j| j.id == "judgment.future" && j.proposal == "proposal.future")
    );
    assert!(next.admissions.iter().any(|a| a.id == "admission.future"));
}
