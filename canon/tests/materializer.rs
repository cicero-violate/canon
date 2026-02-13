use canon::{CanonicalIr, materialize};
use std::path::PathBuf;

#[path = "support.rs"]
mod support;
use support::default_layout_for;

fn load_fixture(name: &str) -> CanonicalIr {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name);
    let data = std::fs::read(path).expect("fixture must exist");
    serde_json::from_slice(&data).expect("fixture must be valid CanonicalIr")
}

#[test]
fn materializer_builds_module_tree() {
    let ir = load_fixture("valid_ir.json");
    let layout = default_layout_for(&ir);
    let result = materialize(&ir, &layout, None);
    let tree = &result.tree;
    assert!(tree.directories().contains("src"));
    assert!(tree.directories().contains("src/Core"));
    assert!(tree.directories().contains("src/Delta"));

    let lib = tree.files().get("src/lib.rs").expect("lib.rs expected");
    assert!(lib.contents.contains("pub mod Core;"));
    assert!(lib.contents.contains("pub mod Delta;"));

    let core_mod = tree
        .files()
        .get("src/Core/mod.rs")
        .expect("Core mod expected");
    assert!(core_mod.contents.contains("pub struct State"));
    assert!(core_mod.contents.contains("pub trait Compute"));
    assert!(core_mod.contents.contains("impl Compute for State"));
    assert!(
        core_mod
            .contents
            .contains("canon_runtime::execute_function")
    );

    let cargo = tree.files().get("Cargo.toml").expect("Cargo file expected");
    assert!(cargo.contents.contains("[package]"));
    assert!(cargo.contents.contains("name = \"CanonDemo\""));
    assert!(cargo.contents.contains("[dependencies]"));
}

#[test]
fn materializer_is_deterministic() {
    let ir = load_fixture("valid_ir.json");
    let layout = default_layout_for(&ir);
    let a = materialize(&ir, &layout, None);
    let b = materialize(&ir, &layout, None);
    assert_eq!(a.tree, b.tree);
    assert_eq!(a.file_hashes, b.file_hashes);
}
