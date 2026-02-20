use canon::{materialize, write_file_tree, CanonicalIr, FileTree};
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

#[path = "support.rs"]
mod support;
use support::default_layout_for;

fn load_fixture(name: &str) -> CanonicalIr {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join(name);
    let data = fs::read(path).expect("fixture must exist");
    serde_json::from_slice(&data).expect("fixture must be valid CanonicalIr")
}

#[test]
fn writes_file_tree_to_disk() {
    let ir = load_fixture("valid_ir.json");
    let layout = default_layout_for(&ir);
    let result = materialize(&ir, &layout, None);
    let dir = tempdir().expect("tempdir");
    write_file_tree(&result.tree, dir.path()).expect("write succeeds");

    let lib_path = dir.path().join("src/lib.rs");
    let core_path = dir.path().join("src/Core/mod.rs");
    let cargo_path = dir.path().join("Cargo.toml");
    assert!(lib_path.exists(), "lib.rs must exist");
    assert!(core_path.exists(), "Core mod must exist");
    assert!(cargo_path.exists(), "Cargo file must exist");

    let lib_contents = fs::read_to_string(lib_path).expect("read lib");
    assert!(lib_contents.contains("pub mod Core;"));

    let core_contents = fs::read_to_string(core_path).expect("read core");
    assert!(core_contents.contains("impl Compute for State"));

    let round_trip = read_tree(dir.path());
    assert_eq!(result.tree, round_trip, "file tree must round-trip exactly");
}

fn read_tree(root: &Path) -> FileTree {
    fn walk(tree: &mut FileTree, root: &Path, current: &Path) {
        if let Ok(entries) = fs::read_dir(current) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if let Ok(rel) = path.strip_prefix(root) {
                        if !rel.as_os_str().is_empty() {
                            tree.add_directory(rel.to_string_lossy().replace('\\', "/"));
                        }
                    }
                    walk(tree, root, &path);
                } else if path.is_file() {
                    if let Ok(rel) = path.strip_prefix(root) {
                        let key = rel.to_string_lossy().replace('\\', "/");
                        let contents = fs::read_to_string(&path).expect("read file");
                        tree.add_file(key, contents);
                    }
                }
            }
        }
    }

    let mut tree = FileTree::new();
    walk(&mut tree, root, root);
    tree
}
