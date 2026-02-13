use canon::{CanonicalIr, auto_accept_dsl_proposal, create_proposal_from_dsl, ir::ProposalStatus};
use std::path::PathBuf;

#[path = "support.rs"]
mod support;
use support::default_layout_for;

fn load_fixture() -> CanonicalIr {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("valid_ir.json");
    let data = std::fs::read(path).expect("fixture");
    serde_json::from_slice(&data).expect("canonical ir")
}

#[test]
fn create_proposal_from_dsl_generates_structures() {
    let source = r#"
        module parse
        module lint imports parse

        goal pipeline: parse -> lint
    "#;
    let artifacts = create_proposal_from_dsl(source).expect("proposal");
    assert_eq!(artifacts.proposal.status, ProposalStatus::Submitted);
    assert!(
        artifacts
            .proposal
            .nodes
            .iter()
            .any(|node| node.id.as_deref() == Some("module.parse"))
    );
    assert!(
        artifacts
            .proposal
            .nodes
            .iter()
            .any(|node| node.id.as_deref() == Some("struct.module_parse.state"))
    );
    assert!(
        artifacts
            .proposal
            .apis
            .iter()
            .any(|api| api.trait_id.contains("parse"))
    );
}

#[test]
fn auto_accept_dsl_proposal_builds_canon() {
    let ir = load_fixture();
    let layout = default_layout_for(&ir);
    let source = r#"
        module parse
        module lint imports parse

        goal pipeline: parse -> lint
    "#;
    let acceptance = auto_accept_dsl_proposal(&ir, &layout, source).expect("accepted");
    assert!(
        acceptance
            .ir
            .modules
            .iter()
            .any(|module| module.id == "module.parse")
    );
}
