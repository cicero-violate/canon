use crate::core::project_editor::refactor::MoveSet;
use crate::core::symbol_id::normalize_symbol_id;
use crate::fs;
use crate::module_path::{compute_new_file_path, ModulePath};
use crate::state::NodeRegistry;
use anyhow::{anyhow, Result};
use database::graph_log::{GraphSnapshot, WireNode, WireNodeId};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(crate) struct FilePlan {
    pub path: PathBuf,
    pub content: String,
}

pub(crate) struct Plan1 {
    pub files: Vec<FilePlan>,
}

pub(crate) fn apply_moves_to_snapshot(snapshot: &mut GraphSnapshot, moveset: &MoveSet) -> Result<()> {
    if moveset.entries.is_empty() {
        return Ok(());
    }
    let mut node_by_key: HashMap<String, usize> = HashMap::new();
    for (idx, node) in snapshot.nodes.iter().enumerate() {
        node_by_key.insert(node.key.clone(), idx);
    }
    for (symbol_id, (old_module, new_module)) in &moveset.entries {
        let name = symbol_id.rsplit("::").next().unwrap_or(symbol_id);
        let old_key = format!("{old_module}::{name}");
        let Some(idx) = node_by_key.get(&old_key).cloned() else {
            continue;
        };
        let node_id = {
            let node = &mut snapshot.nodes[idx];
            node.key = format!("{new_module}::{name}");
            node.metadata.insert("module_path".to_string(), new_module.clone());
            if let Some(module) = node.metadata.get("module") {
                if normalize_symbol_id(module) == normalize_symbol_id(old_module) {
                    node.metadata.insert("module".to_string(), new_module.clone());
                }
            }
            if node.metadata.contains_key("module_id") {
                let new_id = format!("mod:{}", new_module.trim_start_matches("crate::"));
                node.metadata.insert("module_id".to_string(), new_id);
            }
            node.id.clone()
        };
        update_containment_edge(snapshot, node_id, old_module, new_module);
    }
    Ok(())
}

fn update_containment_edge(snapshot: &mut GraphSnapshot, node_id: WireNodeId, old_module: &str, new_module: &str) {
    let from_old = find_node_id_by_key(snapshot, old_module);
    let from_new = find_node_id_by_key(snapshot, new_module);
    let (Some(from_old), Some(from_new)) = (from_old, from_new) else {
        return;
    };
    for edge in &mut snapshot.edges {
        if edge.kind == "contains" && edge.from == from_old && edge.to == node_id {
            edge.from = from_new.clone();
        }
    }
}

fn find_node_id_by_key(snapshot: &GraphSnapshot, key: &str) -> Option<WireNodeId> {
    snapshot.nodes.iter().find(|n| n.key == key).map(|n| n.id.clone())
}

pub(crate) fn project_plan(snapshot: &GraphSnapshot, project_root: &Path) -> Result<Plan1> {
    let mut modules: HashMap<String, WireNode> = HashMap::new();
    let mut items_by_module: HashMap<String, Vec<WireNode>> = HashMap::new();

    for node in &snapshot.nodes {
        let node_kind = node.metadata.get("node_kind").map(|s| s.as_str()).unwrap_or("");
        if node_kind == "module" {
            modules.insert(node.key.clone(), node.clone());
            continue;
        }
        if !is_top_level_item(node) {
            continue;
        }
        let module_path = node
            .metadata
            .get("module_path")
            .or_else(|| node.metadata.get("module"))
            .cloned()
            .unwrap_or_else(|| "crate".to_string());
        let module_path = normalize_symbol_id(&module_path);
        items_by_module.entry(module_path).or_default().push(node.clone());
    }

    if !modules.contains_key("crate") {
        modules.insert(
            "crate".to_string(),
            WireNode {
                id: WireNodeId::from_key("crate"),
                key: "crate".to_string(),
                label: "crate".to_string(),
                metadata: Default::default(),
            },
        );
    }

    let mut files = Vec::new();
    let module_paths: Vec<String> = modules.keys().cloned().collect();
    let mut module_set: HashSet<String> = module_paths.into_iter().collect();
    for module_path in items_by_module.keys() {
        module_set.insert(module_path.clone());
    }
    module_set.insert("crate".to_string());

    let mut module_list: Vec<String> = module_set.into_iter().collect();
    module_list.sort();
    for module_path in module_list {
        let file_path = compute_new_file_path(&ModulePath::from_string(&module_path), project_root)?;
        let mut file_items = Vec::new();

        let mut child_modules: Vec<String> = modules
            .keys()
            .filter(|m| is_direct_child(m, &module_path))
            .cloned()
            .collect();
        child_modules.sort();
        for child in child_modules {
            let vis = module_visibility(&modules, &child)?;
            let name = child.rsplit("::").next().unwrap_or(&child);
            file_items.push(format!("{vis}mod {name};"));
        }

        if let Some(module_items) = items_by_module.get_mut(&module_path) {
            module_items.sort_by(|a, b| a.key.cmp(&b.key));
            for node in module_items.drain(..) {
                if let Some(snippet) = node.metadata.get("source_snippet") {
                    let snippet = snippet.trim();
                    if !snippet.is_empty() {
                        let vis = normalize_visibility(node.metadata.get("visibility").map(|s| s.as_str()))?;
                        let rendered = if vis.is_empty() { snippet.to_string() } else { format!("{vis}{snippet}") };
                        file_items.push(rendered);
                    }
                }
            }
        }

        let content = file_items.join("\n\n");
        files.push(FilePlan { path: file_path, content });
    }

    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(Plan1 { files })
}

fn is_top_level_item(node: &WireNode) -> bool {
    let container = node.metadata.get("container_kind").map(|s| s.as_str()).unwrap_or("");
    if container != "module" {
        return false;
    }
    match node.metadata.get("node_kind").map(|s| s.as_str()) {
        Some("module") => false,
        Some("struct" | "enum" | "union" | "trait" | "impl" | "function" | "const" | "static" | "type_alias") => true,
        _ => false,
    }
}

fn is_direct_child(module: &str, parent: &str) -> bool {
    if !module.starts_with(&format!("{parent}::")) {
        return false;
    }
    let rest = &module[parent.len() + 2..];
    !rest.is_empty() && !rest.contains("::")
}

fn module_visibility(modules: &HashMap<String, WireNode>, module: &str) -> Result<String> {
    let Some(node) = modules.get(module) else {
        return Ok(String::new());
    };
    normalize_visibility(node.metadata.get("visibility").map(|s| s.as_str()))
}

fn normalize_visibility(value: Option<&str>) -> Result<String> {
    let Some(value) = value else {
        return Err(anyhow!("missing visibility metadata"));
    };
    if value.is_empty() {
        return Err(anyhow!("empty visibility metadata"));
    }
    match value {
        "private" => Ok(String::new()),
        "public" => Ok("pub ".to_string()),
        "crate" => Ok("pub(crate) ".to_string()),
        v if v.starts_with("restricted:") => {
            let path = v.trim_start_matches("restricted:");
            if path.is_empty() {
                return Err(anyhow!("restricted visibility missing path"));
            }
            Ok(format!("pub({}) ", path))
        }
        other => Err(anyhow!("unknown visibility value: {other}")),
    }
}

pub(crate) fn emit_plan(
    registry: &mut NodeRegistry,
    plan: Plan1,
    project_root: &Path,
    allow_delete: bool,
) -> Result<HashSet<PathBuf>> {
    let mut touched = HashSet::new();
    enforce_root_guard(project_root, &plan)?;
    let existing = fs::collect_rs_files(project_root)?;
    let plan_paths: HashSet<PathBuf> = plan.files.iter().map(|f| f.path.clone()).collect();
    let deletion_candidates: Vec<PathBuf> = existing
        .into_iter()
        .filter(|file| !plan_paths.contains(file))
        .collect();
    if allow_delete {
        for file in deletion_candidates {
            let _ = std::fs::remove_file(&file);
        }
    } else if !deletion_candidates.is_empty() {
        eprintln!("Plan emission would delete {} files (suppressed):", deletion_candidates.len());
        for file in deletion_candidates {
            eprintln!("  {}", file.display());
        }
    }
    registry.asts.clear();
    registry.sources.clear();
    for file in plan.files {
        let ast = syn::parse_file(&file.content).map_err(|e| anyhow!("failed to parse {}: {e}", file.path.display()))?;
        let source = Arc::new(file.content);
        registry.insert_ast(file.path.clone(), ast);
        registry.insert_source(file.path.clone(), source);
        if let Some(parent) = file.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&file.path, registry.sources.get(&file.path).unwrap().as_str())?;
        touched.insert(file.path);
    }
    Ok(touched)
}

fn enforce_root_guard(project_root: &Path, plan: &Plan1) -> Result<()> {
    if !project_root.join("Cargo.toml").exists() {
        return Err(anyhow!("project_root missing Cargo.toml"));
    }
    for file in &plan.files {
        let rel = file
            .path
            .strip_prefix(project_root)
            .map_err(|_| anyhow!("plan path escapes project_root: {}", file.path.display()))?;
        let mut comps = rel.components();
        if let (Some(first), Some(second)) = (comps.next(), comps.next()) {
            if first.as_os_str() == "src" && second.as_os_str() == "src" {
                return Err(anyhow!("plan path would create nested src/src: {}", file.path.display()));
            }
        }
    }
    Ok(())
}

pub(crate) fn ensure_emission_branch(project_root: &Path) -> Result<()> {
    let current = git_cmd(project_root, &["rev-parse", "--abbrev-ref", "HEAD"])?;
    if current.trim() == "refactor-plan-emission" {
        return Ok(());
    }
    let exists = git_cmd(project_root, &["show-ref", "--verify", "--quiet", "refs/heads/refactor-plan-emission"]).is_ok();
    if exists {
        git_cmd(project_root, &["checkout", "refactor-plan-emission"])?;
        git_cmd(project_root, &["reset", "--hard", "main"])?;
    } else {
        git_cmd(project_root, &["checkout", "-b", "refactor-plan-emission"])?;
    }
    Ok(())
}

pub(crate) fn allow_plan_deletions(project_root: &Path) -> Result<bool> {
    let current = git_cmd(project_root, &["rev-parse", "--abbrev-ref", "HEAD"])?;
    if !current.trim().starts_with("refactor-") {
        return Ok(false);
    }
    Ok(std::env::args().any(|a| a == "--commit"))
}

fn git_cmd(project_root: &Path, args: &[&str]) -> Result<String> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(project_root)
        .output()?;
    if !output.status.success() {
        return Err(anyhow!("git command failed: git {}", args.join(" ")));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

pub(crate) fn rebuild_graph_snapshot(project_root: &Path) -> Result<GraphSnapshot> {
    use compiler_capture::frontends::rustc::RustcFrontend;
    use compiler_capture::multi_capture::capture_project;
    use compiler_capture::project::CargoProject;
    use database::{MemoryEngine, MemoryEngineConfig};

    let cargo = CargoProject::from_entry(project_root)?;
    let frontend = RustcFrontend::new();
    let _artifacts = capture_project(&frontend, &cargo, &[])
        .map_err(|e| anyhow!("rustc capture failed: {e}"))?;
    let state_dir = cargo.workspace_root().join(".rename");
    std::fs::create_dir_all(&state_dir)?;
    let tlog_path = state_dir.join("state.tlog");
    let engine = MemoryEngine::new(MemoryEngineConfig { tlog_path })?;
    let snapshot = engine.materialized_graph()?;
    Ok(snapshot)
}

pub(crate) fn compare_snapshots(left: &GraphSnapshot, right: &GraphSnapshot) -> Result<()> {
    let mut left_nodes = left.nodes.clone();
    let mut right_nodes = right.nodes.clone();
    left_nodes.sort_by(|a, b| a.id.0.cmp(&b.id.0));
    right_nodes.sort_by(|a, b| a.id.0.cmp(&b.id.0));
    if left_nodes.len() != right_nodes.len() {
        return Err(anyhow!("snapshot node count mismatch"));
    }
    for (l, r) in left_nodes.iter().zip(right_nodes.iter()) {
        if l.id != r.id || l.key != r.key || l.label != r.label || l.metadata != r.metadata {
            return Err(anyhow!("snapshot node mismatch"));
        }
    }
    let mut left_edges = left.edges.clone();
    let mut right_edges = right.edges.clone();
    left_edges.sort_by(|a, b| a.id.0.cmp(&b.id.0));
    right_edges.sort_by(|a, b| a.id.0.cmp(&b.id.0));
    if left_edges.len() != right_edges.len() {
        return Err(anyhow!("snapshot edge count mismatch"));
    }
    for (l, r) in left_edges.iter().zip(right_edges.iter()) {
        if l.id != r.id || l.from != r.from || l.to != r.to || l.kind != r.kind || l.metadata != r.metadata {
            return Err(anyhow!("snapshot edge mismatch"));
        }
    }
    Ok(())
}
