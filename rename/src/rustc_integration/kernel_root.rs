use std::path::PathBuf;
/// Canonical persistence root for graph_kernel artifacts.
///
/// Invariant:
///   <project_root>/target/graph_kernel
pub fn kernel_root(project_root: &PathBuf) -> PathBuf {
    project_root.join("target").join("graph_kernel")
}
/// Locate the Cargo project root by walking upward to Cargo.toml.
///
/// Panics if no Cargo.toml is found.
pub fn resolve_project_root() -> PathBuf {
    let cwd = std::env::current_dir().expect("failed to read current_dir");
    let mut cur = cwd.as_path();
    loop {
        if cur.join("Cargo.toml").exists() {
            return cur.to_path_buf();
        }
        match cur.parent() {
            Some(p) => cur = p,
            None => panic!("could not locate project root (Cargo.toml)"),
        }
    }
}
