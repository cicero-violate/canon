use crate::core::project_editor::invariants::{
    assert_edge_endpoints_exist, assert_module_path_consistency, assert_no_duplicate_edges, assert_unique_def_paths, assert_unique_node_ids, assert_unique_node_keys,
};
use crate::core::project_editor::refactor::MoveSet;
use crate::core::symbol_id::normalize_symbol_id;
use crate::module_path::ModulePath;
use crate::state::NodeRegistry;
use crate::structured::ast_render::render_node;
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

pub(crate) struct EmissionReport {
    pub written: Vec<PathBuf>,
    pub unchanged: Vec<PathBuf>,
    pub deletion_candidates: Vec<PathBuf>,
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
    let from_old = find_node_id_by_module(snapshot, old_module);
    let from_new = find_node_id_by_module(snapshot, new_module);
    let (Some(from_old), Some(from_new)) = (from_old, from_new) else {
        return;
    };
    for edge in &mut snapshot.edges {
        if edge.kind == "contains" && edge.from == from_old && edge.to == node_id {
            edge.from = from_new.clone();
        }
    }
}

fn find_node_id_by_module(snapshot: &GraphSnapshot, module_path: &str) -> Option<WireNodeId> {
    snapshot.nodes.iter().find(|n| n.key == module_path || n.metadata.get("def_path").map(|v| normalize_symbol_id(v) == module_path).unwrap_or(false)).map(|n| n.id.clone())
}

pub(crate) fn project_plan(snapshot: &GraphSnapshot, project_root: &Path) -> Result<Plan1> {
    // Enforce structural identity invariants before projection
    assert_unique_node_keys(snapshot);
    assert_unique_def_paths(snapshot);
    assert_unique_node_ids(snapshot);
    assert_edge_endpoints_exist(snapshot);
    assert_no_duplicate_edges(snapshot);
    assert_module_path_consistency(snapshot);

    let mut modules: HashMap<String, WireNode> = HashMap::new();
    let mut items_by_module: HashMap<String, Vec<WireNode>> = HashMap::new();
    let mut ast_cache: HashMap<PathBuf, syn::File> = HashMap::new();
    let mut seen_item_keys: HashSet<String> = HashSet::new();
    let debug_plan = std::env::var("RENAME_DEBUG_PLAN").ok().as_deref() == Some("1");
    let mut node_kind_counts: HashMap<String, usize> = HashMap::new();
    let mut container_kind_counts: HashMap<String, usize> = HashMap::new();

    for node in &snapshot.nodes {
        let node_kind = node.metadata.get("node_kind").map(|s| s.as_str()).unwrap_or("");
        *node_kind_counts.entry(node_kind.to_string()).or_insert(0) += 1;
        let container_kind = node.metadata.get("container_kind").cloned().unwrap_or_default();
        *container_kind_counts.entry(container_kind).or_insert(0) += 1;
        if node_kind == "module" {
            let module_id = node.metadata.get("def_path").or_else(|| node.metadata.get("module_path")).cloned().unwrap_or_else(|| node.key.clone());
            let module_id = normalize_symbol_id(&module_id);
            if !is_local_module_path(&module_id) {
                continue;
            }
            if !is_local_source(node, project_root) {
                continue;
            }
            modules.insert(module_id, node.clone());
            continue;
        }
        if !is_top_level_item(node) {
            continue;
        }
        if !is_local_source(node, project_root) {
            continue;
        }
        if debug_plan && node.metadata.get("source_snippet").is_none() {
            eprintln!("[plan] top-level item missing source_snippet: key={} kind={}", node.key, node_kind);
        }
        let module_path = normalize_symbol_id(&select_module_path(node));
        if !is_local_module_path(&module_path) {
            continue;
        }
        if seen_item_keys.insert(node.key.clone()) {
            items_by_module.entry(module_path).or_default().push(node.clone());
        }
    }

    if !modules.contains_key("crate") {
        modules.insert("crate".to_string(), WireNode { id: WireNodeId::from_key("crate"), key: "crate".to_string(), label: "crate".to_string(), metadata: Default::default() });
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
    if debug_plan {
        eprintln!("[plan] node_kind_counts={:?}", node_kind_counts);
        eprintln!("[plan] container_kind_counts={:?}", container_kind_counts);
        eprintln!("[plan] modules={} items_by_module={}", modules.len(), items_by_module.len());
    }
    for module_path in module_list {
        let has_children = modules.keys().any(|m| is_direct_child(m, &module_path));
        let file_path = module_file_path(&module_path, has_children, project_root)?;
        if debug_plan {
            eprintln!("[plan] module={} has_children={} file={}", module_path, has_children, file_path.display());
        }
        let mut file_items = Vec::new();

        // Collect use statements from the original source file for this module.
        let use_items: Vec<String> = {
            let mut uses = Vec::new();
            // Derive the original source path deterministically from the module path.
            // module_file_path already knows the src/ layout; we just need to point
            // at the original tree (project_root/src/…) not the output (fix_point/src/…).
            if let Ok(orig_path) = module_file_path(&module_path, has_children, project_root) {
                // If computed path is mod.rs but original used file-module layout (e.g. src/fs.rs),
                // fall back to the sibling .rs file.
                let orig_path = if !orig_path.exists() && orig_path.file_name().map(|f| f == "mod.rs").unwrap_or(false) {
                    let sibling = orig_path.parent().map(|p| p.with_extension("rs")).filter(|p| p.exists()).unwrap_or(orig_path);
                    sibling
                } else {
                    orig_path
                };
                if !ast_cache.contains_key(&orig_path) {
                    if let Ok(content) = std::fs::read_to_string(&orig_path) {
                        if let Ok(parsed) = syn::parse_file(&content) {
                            ast_cache.insert(orig_path.clone(), parsed);
                        }
                    }
                }
                if let Some(ast) = ast_cache.get(&orig_path) {
                    for item in &ast.items {
                        if let syn::Item::Use(_) = item {
                            uses.push(render_node(item));
                        }
                    }
                }
            }
            uses
        };

        let mut child_modules: Vec<String> = modules.keys().filter(|m| is_direct_child(m, &module_path)).cloned().collect();
        child_modules.sort();
        for child in child_modules {
            let vis = module_visibility(&modules, &child)?;
            let name = child.rsplit("::").next().unwrap_or(&child);
            file_items.push(format!("{vis}mod {name};"));
        }

        for use_item in use_items {
            file_items.push(use_item);
        }

        if let Some(module_items) = items_by_module.get_mut(&module_path) {
            module_items.sort_by(|a, b| a.key.cmp(&b.key));
            for node in module_items.drain(..) {
                let snippet = extract_item_snippet(&node, project_root, &mut ast_cache);
                if let Some(snippet) = snippet {
                    let snippet = snippet.trim();
                    if !snippet.is_empty() {
                        let vis = normalize_visibility(node.metadata.get("visibility").map(|s| s.as_str()))?;
                        if let Ok(mut item) = syn::parse_str::<syn::Item>(snippet) {
                            apply_visibility(&mut item, vis.trim());
                            file_items.push(render_node(&item));
                        } else {
                            let snippet = strip_leading_visibility(snippet);
                            let rendered = if vis.is_empty() { snippet } else { format!("{vis}{snippet}") };
                            file_items.push(rendered);
                        }
                    }
                }
            }
        }

        let content = file_items.join("\n\n");
        if debug_plan {
            eprintln!("[plan] file={} items={}", file_path.display(), file_items.len());
        }
        files.push(FilePlan { path: file_path, content });
    }

    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(Plan1 { files })
}

fn is_top_level_item(node: &WireNode) -> bool {
    let node_kind = node.metadata.get("node_kind").map(|s| s.as_str()).unwrap_or("");
    if node_kind == "module" {
        return false;
    }
    let is_emit_kind = matches!(node_kind, "struct" | "enum" | "union" | "trait" | "impl" | "function" | "const" | "static" | "type_alias");
    if !is_emit_kind {
        return false;
    }
    let container = node.metadata.get("container_kind").map(|s| s.as_str()).unwrap_or("");
    container.is_empty() || container == "module" || container == "unknown"
}

fn is_local_module_path(module_path: &str) -> bool {
    if module_path.is_empty() {
        return false;
    }
    if module_path.contains('<') || module_path.contains('>') {
        return false;
    }
    module_path == "crate" || module_path.starts_with("crate::")
}

fn select_module_path(node: &WireNode) -> String {
    let candidates = [node.metadata.get("module_path"), node.metadata.get("module")];
    for cand in candidates {
        if let Some(value) = cand {
            if !value.is_empty() && !value.contains('<') && !value.contains('>') {
                return value.clone();
            }
        }
    }
    "crate".to_string()
}

fn extract_item_snippet(node: &WireNode, project_root: &Path, ast_cache: &mut HashMap<PathBuf, syn::File>) -> Option<String> {
    let node_kind = node.metadata.get("node_kind").map(|s| s.as_str()).unwrap_or("");
    if !matches!(node_kind, "struct" | "enum" | "union" | "trait" | "impl" | "function" | "const" | "static" | "type_alias") {
        return None;
    }
    if let Some(snippet) = node.metadata.get("source_snippet") {
        if syn::parse_str::<syn::Item>(snippet).is_ok() {
            return Some(snippet.clone());
        }
    }
    render_item_from_ast(node, project_root, ast_cache)
}

fn apply_visibility(item: &mut syn::Item, vis_str: &str) {
    let vis = if vis_str.is_empty() {
        syn::Visibility::Inherited
    } else if let Ok(parsed) = syn::parse_str::<syn::Visibility>(vis_str) {
        parsed
    } else {
        syn::Visibility::Inherited
    };
    match item {
        syn::Item::Const(i) => i.vis = vis,
        syn::Item::Enum(i) => i.vis = vis,
        syn::Item::ExternCrate(i) => i.vis = vis,
        syn::Item::Fn(i) => i.vis = vis,
        syn::Item::Mod(i) => i.vis = vis,
        syn::Item::Static(i) => i.vis = vis,
        syn::Item::Struct(i) => i.vis = vis,
        syn::Item::Trait(i) => i.vis = vis,
        syn::Item::TraitAlias(i) => i.vis = vis,
        syn::Item::Type(i) => i.vis = vis,
        syn::Item::Union(i) => i.vis = vis,
        syn::Item::Use(i) => i.vis = vis,
        syn::Item::Impl(_) => {}
        _ => {}
    }
}

fn render_item_from_ast(node: &WireNode, project_root: &Path, ast_cache: &mut HashMap<PathBuf, syn::File>) -> Option<String> {
    let source_file = node.metadata.get("source_file")?;
    if source_file.contains('<') || source_file.contains('>') {
        return None;
    }
    let path = PathBuf::from(source_file);
    if !path.starts_with(project_root) {
        return None;
    }
    let ast = if let Some(ast) = ast_cache.get(&path) {
        ast
    } else {
        let content = std::fs::read_to_string(&path).ok()?;
        let parsed = syn::parse_file(&content).ok()?;
        ast_cache.insert(path.clone(), parsed);
        ast_cache.get(&path)?
    };
    let name = node.metadata.get("name").cloned().or_else(|| node.metadata.get("def_path").and_then(|d| d.split("::").last().map(|s| s.to_string()))).unwrap_or_default();
    let node_kind = node.metadata.get("node_kind").map(|s| s.as_str()).unwrap_or("");
    match node_kind {
        "struct" => ast.items.iter().find_map(|i| match i {
            syn::Item::Struct(s) if s.ident == name => Some(render_node(s)),
            _ => None,
        }),
        "enum" => ast.items.iter().find_map(|i| match i {
            syn::Item::Enum(e) if e.ident == name => Some(render_node(e)),
            _ => None,
        }),
        "union" => ast.items.iter().find_map(|i| match i {
            syn::Item::Union(u) if u.ident == name => Some(render_node(u)),
            _ => None,
        }),
        "trait" => ast.items.iter().find_map(|i| match i {
            syn::Item::Trait(t) if t.ident == name => Some(render_node(t)),
            _ => None,
        }),
        "type_alias" => ast.items.iter().find_map(|i| match i {
            syn::Item::Type(t) if t.ident == name => Some(render_node(t)),
            _ => None,
        }),
        "const" => ast.items.iter().find_map(|i| match i {
            syn::Item::Const(c) if c.ident == name => Some(render_node(c)),
            _ => None,
        }),
        "static" => ast.items.iter().find_map(|i| match i {
            syn::Item::Static(s) if s.ident == name => Some(render_node(s)),
            _ => None,
        }),
        "function" => ast.items.iter().find_map(|i| match i {
            syn::Item::Fn(f) if f.sig.ident == name => Some(render_node(f)),
            _ => None,
        }),
        "impl" => {
            use quote::ToTokens;
            ast.items.iter().find_map(|i| match i {
                syn::Item::Impl(imp) => {
                    let self_ty = imp.self_ty.to_token_stream().to_string();
                    if !name.is_empty() && !self_ty.contains(&name) {
                        return None;
                    }
                    Some(render_node(imp))
                }
                _ => None,
            })
        }
        _ => None,
    }
}

fn is_local_source(node: &WireNode, project_root: &Path) -> bool {
    let Some(source_file) = node.metadata.get("source_file") else {
        return false;
    };
    if source_file.contains('<') || source_file.contains('>') {
        return false;
    }
    let path = Path::new(source_file);
    if path.is_absolute() {
        path.starts_with(project_root)
    } else {
        project_root.join(path).starts_with(project_root)
    }
}

fn is_direct_child(module: &str, parent: &str) -> bool {
    if !module.starts_with(&format!("{parent}::")) {
        return false;
    }
    let rest = &module[parent.len() + 2..];
    !rest.is_empty() && !rest.contains("::")
}

fn module_file_path(module_path: &str, has_children: bool, project_root: &Path) -> Result<PathBuf> {
    let mut path = project_root.join("src");
    let module = ModulePath::from_string(module_path);
    let segments: Vec<_> = module.segments.iter().skip_while(|s| *s == "crate").collect();
    if segments.is_empty() {
        return Ok(path.join("lib.rs"));
    }
    for segment in &segments[..segments.len().saturating_sub(1)] {
        path = path.join(segment);
    }
    if let Some(last) = segments.last() {
        if has_children {
            path = path.join(last).join("mod.rs");
        } else {
            path = path.join(format!("{last}.rs"));
        }
    }
    Ok(path)
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
        "crate" => Ok("pub ".to_string()),
        v if v.starts_with("restricted:") => Ok("pub ".to_string()),
        other => Err(anyhow!("unknown visibility value: {other}")),
    }
}

fn strip_leading_visibility(snippet: &str) -> String {
    let s = snippet.trim_start();
    if let Some(rest) = s.strip_prefix("pub ") {
        return rest.trim_start().to_string();
    }
    if let Some(rest) = s.strip_prefix("pub(") {
        if let Some(idx) = rest.find(')') {
            let after = &rest[idx + 1..];
            return after.trim_start().to_string();
        }
    }
    s.to_string()
}

pub(crate) fn emit_plan(registry: &mut NodeRegistry, plan: Plan1, project_root: &Path, _allow_delete: bool) -> Result<EmissionReport> {
    let mut written = Vec::new();
    let mut unchanged = Vec::new();
    enforce_root_guard(project_root, &plan)?;
    let output_root = project_root.join("fix_point");
    let deletion_candidates: Vec<PathBuf> = Vec::new();
    let deletion_count = 0;
    registry.asts.clear();
    registry.sources.clear();
    for file in plan.files {
        let rel = file.path.strip_prefix(project_root).map_err(|_| anyhow!("plan path escapes project_root: {}", file.path.display()))?;
        let out_path = output_root.join(rel);
        let ast = syn::parse_file(&file.content).map_err(|e| {
            let preview = file.content.lines().take(20).collect::<Vec<_>>().join("\n");
            anyhow!("failed to parse {}: {e}\n---\n{preview}", file.path.display())
        })?;
        let source = Arc::new(file.content);
        registry.insert_ast(out_path.clone(), ast);
        registry.insert_source(out_path.clone(), source);
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let new_content = registry.sources.get(&out_path).unwrap().as_str();
        let mut changed = true;
        if let Ok(existing) = std::fs::read_to_string(&out_path) {
            if existing == new_content {
                changed = false;
            }
        }
        if changed {
            std::fs::write(&out_path, new_content)?;
            written.push(out_path);
        } else {
            unchanged.push(out_path);
        }
    }
    eprintln!("Plan emission summary: written={}, unchanged={}, deletions={}", written.len(), unchanged.len(), deletion_count);
    Ok(EmissionReport { written, unchanged, deletion_candidates })
}

fn enforce_root_guard(project_root: &Path, plan: &Plan1) -> Result<()> {
    if !project_root.join("Cargo.toml").exists() {
        return Err(anyhow!("project_root missing Cargo.toml"));
    }
    if !project_root.join("src").exists() {
        return Err(anyhow!("project_root/src missing"));
    }
    if plan.files.is_empty() {
        return Err(anyhow!("plan has no files"));
    }
    let output_root = project_root.join("fix_point");
    for file in &plan.files {
        let rel_any = file.path.strip_prefix(project_root).map_err(|_| anyhow!("plan path escapes project_root: {}", file.path.display()))?;
        for comp in rel_any.components() {
            if comp.as_os_str() == ".." {
                return Err(anyhow!("plan path contains ..: {}", file.path.display()));
            }
        }
        let rel = file.path.strip_prefix(project_root).map_err(|_| anyhow!("plan path escapes project_root: {}", file.path.display()))?;
        let rel = file.path.strip_prefix(project_root).map_err(|_| anyhow!("plan path escapes project_root: {}", file.path.display()))?;
        let out_path = output_root.join(rel);
        if out_path.components().any(|c| c.as_os_str() == "..") {
            return Err(anyhow!("output path contains ..: {}", out_path.display()));
        }
    }
    Ok(())
}

pub(crate) fn rollback_emission(project_root: &Path, written: &[PathBuf]) -> Result<()> {
    for path in written {
        let _ = std::fs::remove_file(path);
    }
    Ok(())
}

pub(crate) fn maybe_commit_emission(project_root: &Path) -> Result<()> {
    Ok(())
}

pub(crate) fn rebuild_graph_snapshot(project_root: &Path) -> Result<GraphSnapshot> {
    use compiler_capture::frontends::rustc::RustcFrontend;
    use compiler_capture::multi_capture::capture_project;
    use compiler_capture::project::CargoProject;
    use kernel::kernel::{Kernel as MemoryEngine, MemoryEngineConfig};

    let cargo = CargoProject::from_entry(project_root)?;
    let frontend = RustcFrontend::new();
    let _artifacts = capture_project(&frontend, &cargo, &[]).map_err(|e| anyhow!("rustc capture failed: {e}"))?;
    let workspace_root = cargo.metadata().map(|m| m.workspace_root).unwrap_or_else(|_| cargo.workspace_root().to_path_buf());
    let state_dir = workspace_root.join(".rename");
    std::fs::create_dir_all(&state_dir)?;
    let tlog_path = state_dir.join("state.tlog");
    let engine = MemoryEngine::new(MemoryEngineConfig { tlog_path })?;
    let snapshot = engine.materialized_graph()?;
    Ok(snapshot)
}

pub(crate) fn compare_snapshots(left: &GraphSnapshot, right: &GraphSnapshot) -> Result<()> {
    let mut left_map: HashMap<String, WireNode> = HashMap::new();
    let mut right_map: HashMap<String, WireNode> = HashMap::new();
    for node in &left.nodes {
        let Some(def_path) = node.metadata.get("def_path") else {
            continue;
        };
        let kind = node.metadata.get("node_kind").map(|s| s.as_str()).unwrap_or("");
        let key = format!("{def_path}#{kind}");
        left_map.insert(key, node.clone());
    }
    for node in &right.nodes {
        let Some(def_path) = node.metadata.get("def_path") else {
            continue;
        };
        let kind = node.metadata.get("node_kind").map(|s| s.as_str()).unwrap_or("");
        let key = format!("{def_path}#{kind}");
        right_map.insert(key, node.clone());
    }
    if left_map.len() != right_map.len() {
        let left_keys: HashSet<_> = left_map.keys().cloned().collect();
        let right_keys: HashSet<_> = right_map.keys().cloned().collect();
        let extra: Vec<_> = right_keys.difference(&left_keys).take(10).cloned().collect();
        let missing: Vec<_> = left_keys.difference(&right_keys).take(10).cloned().collect();
        return Err(anyhow!("snapshot node count mismatch (left={}, right={}, extra_right={:?}, missing_right={:?})", left_map.len(), right_map.len(), extra, missing));
    }
    for (key, left_node) in &left_map {
        let Some(right_node) = right_map.get(key) else {
            return Err(anyhow!("snapshot node missing for key={}", key));
        };
        let left_meta = filtered_metadata(left_node);
        let right_meta = filtered_metadata(right_node);
        if left_meta != right_meta {
            return Err(anyhow!("snapshot node mismatch for key={}", key));
        }
    }

    let mut left_edges: HashSet<(String, String, String)> = HashSet::new();
    let mut right_edges: HashSet<(String, String, String)> = HashSet::new();
    let left_id_to_key: HashMap<_, _> = left_map.iter().map(|(k, v)| (v.id.clone(), k.clone())).collect();
    let right_id_to_key: HashMap<_, _> = right_map.iter().map(|(k, v)| (v.id.clone(), k.clone())).collect();
    for edge in &left.edges {
        let (Some(from), Some(to)) = (left_id_to_key.get(&edge.from), left_id_to_key.get(&edge.to)) else {
            continue;
        };
        left_edges.insert((from.clone(), to.clone(), edge.kind.clone()));
    }
    for edge in &right.edges {
        let (Some(from), Some(to)) = (right_id_to_key.get(&edge.from), right_id_to_key.get(&edge.to)) else {
            continue;
        };
        right_edges.insert((from.clone(), to.clone(), edge.kind.clone()));
    }
    if left_edges != right_edges {
        return Err(anyhow!("snapshot edge mismatch"));
    }
    Ok(())
}

fn filtered_metadata(node: &WireNode) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for key in ["def_path", "node_kind", "module", "module_path", "module_id", "name", "visibility", "kind", "path"] {
        if let Some(value) = node.metadata.get(key) {
            out.insert(key.to_string(), value.clone());
        }
    }
    out
}
