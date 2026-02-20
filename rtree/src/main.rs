use anyhow::Result;
use algorithms::graph::csr::Csr;
use colored::*;
use serde_json::Value as JsonValue;
use std::collections::{HashMap, HashSet};
use std::env;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Command;

struct CrateMeta {
    pkg_name: String,
    src_root: PathBuf,
    workspace_deps: Vec<String>,
}

fn is_workspace_root(path: &Path) -> bool {
    let toml = match std::fs::read_to_string(path.join("Cargo.toml")) {
        Ok(s) => s,
        Err(_) => return false,
    };
    toml.contains("[workspace]")
}

fn cargo_metadata(root: &Path) -> Result<JsonValue> {
    let out = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version=1"])
        .current_dir(root)
        .output()?;
    if !out.status.success() {
        anyhow::bail!("cargo metadata failed");
    }
    Ok(serde_json::from_slice(&out.stdout)?)
}

fn collect_workspace_crates(root: &Path) -> Result<Vec<CrateMeta>> {
    let meta = cargo_metadata(root)?;

    let member_ids: HashSet<String> = meta["workspace_members"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("invalid workspace_members"))?
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();

    let packages = meta["packages"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("invalid packages"))?;

    let mut crates = Vec::new();

    for pkg in packages {
        let id = pkg["id"].as_str().unwrap_or("");
        if !member_ids.contains(id) {
            continue;
        }

        let pkg_name = pkg["name"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing name"))?
            .to_string();

        let manifest = PathBuf::from(
            pkg["manifest_path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing manifest_path"))?,
        );

        let src_root = manifest.parent().unwrap().join("src");
        if !src_root.exists() {
            continue;
        }

        let workspace_deps = pkg["dependencies"]
            .as_array()
            .map(|deps| {
                deps.iter()
                    .filter_map(|d| {
                        let dep_name = d["name"].as_str()?;
                        if packages.iter().any(|p| {
                            p["name"].as_str() == Some(dep_name)
                                && member_ids.contains(p["id"].as_str().unwrap_or(""))
                        }) {
                            Some(dep_name.to_string())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        crates.push(CrateMeta {
            pkg_name,
            src_root,
            workspace_deps,
        });
    }

    Ok(crates)
}

fn run_workspace(root: &Path, dot: bool, cfg_style: bool) -> Result<()> {
    let crates = collect_workspace_crates(root)?;

    let mut unified = Graph::new();
    let mut name_to_node: HashMap<String, usize> = HashMap::new();

    for c in &crates {
        let g = build_module_graph(&c.src_root, &c.pkg_name)?;
        for node in 0..g.modules.len() {
            let n = g.modules[node].name.clone();
            name_to_node.entry(n.clone()).or_insert_with(|| {
                unified.add_node(Module { name: n })
            });
        }
        for (src, dst) in g.edges() {
            let src_name = &g.modules[src].name;
            let dst_name = &g.modules[dst].name;
            let src_idx = name_to_node[src_name];
            let dst_idx = name_to_node[dst_name];
            unified.add_edge(src_idx, dst_idx);
        }
    }

    for c in &crates {
        let self_lib = format!("{}::lib", c.pkg_name);
        let self_main = format!("{}::main", c.pkg_name);

        let from = name_to_node
            .get(&self_lib)
            .or_else(|| name_to_node.get(&self_main))
            .copied();

        for dep in &c.workspace_deps {
            let dep_lib = format!("{}::lib", dep);
            let dep_main = format!("{}::main", dep);

            let to = name_to_node
                .get(&dep_lib)
                .or_else(|| name_to_node.get(&dep_main))
                .copied();

            if let (Some(f), Some(t)) = (from, to) {
                unified.add_edge(f, t);
            }
        }
    }

    let unified = collapse_cycles(&unified)?;

    if dot {
        print_dot(&unified)?;
    } else if cfg_style {
        print_cfg_dot(&unified)?;
    } else {
        print_flat_dependencies(&unified, None)?;
    }

    Ok(())
}

mod dep_info;
mod metadata;

use dep_info::build_file_dependency_map;
use metadata::get_project_metadata;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Module {
    name: String,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let dot = args.iter().any(|a| a == "--dot");
    let cfg_style = args.iter().any(|a| a == "--cfg-style");
    let path_arg = args.iter().skip(1).find(|a| !a.starts_with("--"));
    if path_arg.is_none() {
        eprintln!("Usage: rtree [--dot] [--cfg-style] <directory_path>");
        std::process::exit(1);
    }

    let dir_path = Path::new(path_arg.unwrap());

    if cfg!(feature = "cuda") {
        eprintln!("rtree: cuda feature enabled");
    } else {
        eprintln!("rtree: cuda feature disabled");
    }

    if is_workspace_root(dir_path) {
        return run_workspace(dir_path, dot, cfg_style);
    }

    let project_meta = get_project_metadata(
        dir_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("invalid path"))?,
    )?;

    // Metadata is intentionally observed to keep it semantically live
    eprintln!(
        "package={} root={} src={}",
        project_meta.package_name,
        project_meta.root.display(),
        project_meta.src_root.display()
    );

    let module_graph = build_module_graph(&project_meta.src_root, "")?;

    if module_graph.node_count() == 0 {
        eprintln!("No Rust modules found");
        std::process::exit(1);
    }

    if dot {
        print_dot(&module_graph)?;
    } else if cfg_style {
        print_cfg_dot(&module_graph)?;
    } else {
        print_flat_dependencies(&module_graph, Some(&project_meta.src_root))?;
    }

    Ok(())
}

fn build_module_graph(src_root: &Path, prefix: &str) -> Result<Graph> {
    let file_deps = build_file_dependency_map(src_root)?;

    let mut graph = Graph::new();
    let mut module_to_node: HashMap<String, usize> = HashMap::new();
    let mut file_to_module: HashMap<PathBuf, String> = HashMap::new();

    // First pass: create nodes
    for (file, _) in &file_deps {
        if !file.starts_with(src_root) {
            continue;
        }

        let mut module = file_to_module_name(file, src_root)?;
        if !prefix.is_empty() {
            module = format!("{}::{}", prefix, module);
        }
        module_to_node.entry(module.clone()).or_insert_with(|| {
            graph.add_node(Module { name: module.clone() })
        });
        file_to_module.insert(file.clone(), module);
    }

    // Second pass: create edges
    for (file, deps) in &file_deps {
        let from = match file_to_module.get(file) {
            Some(m) => m,
            None => continue,
        };
        let from_idx = module_to_node[from];

        for dep in deps {
            if let Some(to) = file_to_module.get(dep) {
                // Suppress child→ancestor edges (e.g. ingest::parser → ingest).
                // A module referencing its own parent via `crate::parent::X` is
                // not a true dependency — the parent already owns the child via
                // `mod child`.  Including it creates false cycles.
                let is_ancestor = {
                    let ancestor_prefix = format!("{}::", to);
                    from.starts_with(&ancestor_prefix) || from == to.as_str()
                };
                if from != to && !is_ancestor {
                    graph.add_edge(from_idx, module_to_node[to]);
                }
            }
        }
    }

    collapse_cycles(&graph)
}

fn file_to_module_name(path: &Path, src_root: &Path) -> Result<String> {
    let rel = path.strip_prefix(src_root)?.with_extension("");
    let parts: Vec<&str> = rel
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();

    let name = if parts.last() == Some(&"mod") {
        parts[..parts.len() - 1].join("::")
    } else if parts.last() == Some(&"lib") {
        "lib".to_string()
    } else if parts.last() == Some(&"main") {
        "main".to_string()
    } else {
        parts.join("::")
    };

    Ok(name)
}

fn print_dot(graph: &Graph) -> Result<()> {
    println!("digraph dependencies {{");
    println!("  rankdir=LR;");
    println!("  node [shape=box fontname=\"monospace\"];");

    for node in 0..graph.modules.len() {
        let label = graph.modules[node].name.replace('"', "\\\"");
        println!("  \"{}\";", label);
    }

    for (src, dst) in graph.edges() {
        let src = graph.modules[src].name.replace('"', "\\\"");
        let dst = graph.modules[dst].name.replace('"', "\\\"");
        println!("  \"{}\" -> \"{}\";", src, dst);
    }

    println!("}}");
    Ok(())
}

fn print_flat_dependencies(graph: &Graph, src_root: Option<&Path>) -> Result<()> {
    let mut modules: Vec<_> = (0..graph.modules.len()).collect();
    modules.sort_by_key(|&n| &graph.modules[n].name);

    for node in modules {
        let module = &graph.modules[node];
        let name = colorize(&module.name);
        let filename = if let Some(root) = src_root {
            module_to_filename(&module.name, root)?
        } else {
            String::new()
        };

        let mut deps: Vec<_> = graph.adj[node]
            .iter()
            .map(|&n| &graph.modules[n].name)
            .collect();
        deps.sort();

        if deps.is_empty() {
            if filename.is_empty() {
                println!("{} → (no dependencies)", name);
            } else {
                println!("{} → (no dependencies) {}", name, filename.bright_black());
            }
        } else {
            if filename.is_empty() {
                println!("{} →", name);
            } else {
                println!("{} → {}", name, filename.bright_black());
            }
            for (i, dep) in deps.iter().enumerate() {
                let is_last = i + 1 == deps.len();
                let connector = if is_last {
                    "  └── "
                } else {
                    "  ├── "
                };
                println!("{}{}", connector, colorize(dep));
            }
        }
    }
    Ok(())
}

fn module_to_filename(module_name: &str, src_root: &Path) -> Result<String> {
    let path = if module_name == "lib" {
        src_root.join("lib.rs")
    } else if module_name == "main" {
        src_root.join("main.rs")
    } else if module_name.starts_with("bin::") {
        let bin_name = module_name.strip_prefix("bin::").unwrap();
        src_root.join("bin").join(format!("{}.rs", bin_name))
    } else {
        let parts: Vec<&str> = module_name.split("::").collect();
        let mut candidate = src_root.join(parts.join("/")).with_extension("rs");

        if !candidate.exists() {
            candidate = src_root.join(parts.join("/")).join("mod.rs");
        }

        candidate
    };

    Ok(path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string())
}

fn collapse_cycles(graph: &Graph) -> Result<Graph> {
    let sccs = compute_sccs(graph);
    let mut new_graph = Graph::new();
    let mut old_to_new: HashMap<usize, usize> = HashMap::new();

    for scc in sccs {
        if scc.len() == 1 {
            let old = scc[0];
            let idx = new_graph.add_node(graph.modules[old].clone());
            old_to_new.insert(old, idx);
        } else {
            let names: Vec<String> = scc.iter().map(|&i| graph.modules[i].name.clone()).collect();
            let idx = new_graph.add_node(Module {
                name: format!("[cycle: {}]", names.join(", ")),
            });
            for &old in &scc {
                old_to_new.insert(old, idx);
            }
        }
    }

    let mut seen = HashSet::new();
    for (src, dst) in graph.edges() {
        let src = old_to_new[&src];
        let dst = old_to_new[&dst];
        if src != dst && seen.insert((src, dst)) {
            new_graph.add_edge(src, dst);
        }
    }

    Ok(new_graph)
}

fn module_dir(name: &str) -> &str {
    match name.rfind("::") {
        Some(pos) => &name[..pos],
        None => "",
    }
}

fn print_cfg_dot(graph: &Graph) -> Result<()> {
    let mut dir_to_nodes: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    for node in 0..graph.modules.len() {
        let dir = module_dir(&graph.modules[node].name).to_string();
        dir_to_nodes.entry(dir).or_default().push(node);
    }

    println!("digraph dependencies {{");
    println!("  rankdir=LR;");
    println!("  node [shape=box fontname=\"monospace\" fillcolor=lightblue style=filled];");
    println!("  edge [fontname=\"monospace\"];");
    println!();

    for (cluster_idx, (dir, nodes)) in dir_to_nodes.iter().enumerate() {
        let cluster_label = if dir.is_empty() {
            "(root)".to_string()
        } else {
            dir.replace('"', "\\\"")
        };

        println!("  subgraph cluster_{cluster_idx} {{");
        println!("    label=\"{cluster_label}\";");
        println!("    style=filled;");
        println!("    fillcolor=lightgray;");
        println!("    color=black;");

        for node in nodes {
            let idx = *node;
            let label = graph.modules[*node].name.replace('"', "\\\"");
            println!("    n{idx} [label=\"{label}\"];");
        }

        println!("  }}");
        println!();
    }

    for (src, dst) in graph.edges() {
        println!("  n{src} -> n{dst};");
    }

    println!("}}");
    Ok(())
}

#[derive(Default)]
struct Graph {
    modules: Vec<Module>,
    adj: Vec<Vec<usize>>,
}

impl Graph {
    fn new() -> Self {
        Self { modules: Vec::new(), adj: Vec::new() }
    }

    fn node_count(&self) -> usize {
        self.modules.len()
    }

    fn add_node(&mut self, module: Module) -> usize {
        let idx = self.modules.len();
        self.modules.push(module);
        self.adj.push(Vec::new());
        idx
    }

    fn add_edge(&mut self, from: usize, to: usize) {
        if from == to {
            return;
        }
        if !self.adj[from].contains(&to) {
            self.adj[from].push(to);
        }
    }

    fn edges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.adj.iter().enumerate().flat_map(|(src, dsts)| {
            dsts.iter().copied().map(move |dst| (src, dst))
        })
    }
}

fn compute_sccs(graph: &Graph) -> Vec<Vec<usize>> {
    let n = graph.modules.len();
    if n == 0 {
        return Vec::new();
    }

    let csr = Csr::from_adj(&graph.adj);
    let rev_adj = build_reverse_adj(&graph.adj);
    let rev_csr = Csr::from_adj(&rev_adj);

    let mut assigned = vec![false; n];
    let mut sccs = Vec::new();

    for start in 0..n {
        if assigned[start] {
            continue;
        }
        let reach_from = reachable_from(&csr, &graph.adj, start);
        let reach_to = reachable_from(&rev_csr, &rev_adj, start);

        let mut scc = Vec::new();
        for v in 0..n {
            if !assigned[v] && reach_from[v] && reach_to[v] {
                assigned[v] = true;
                scc.push(v);
            }
        }
        sccs.push(scc);
    }

    sccs
}

fn build_reverse_adj(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut rev = vec![Vec::new(); adj.len()];
    for (src, dsts) in adj.iter().enumerate() {
        for &dst in dsts {
            rev[dst].push(src);
        }
    }
    rev
}

#[cfg(feature = "cuda")]
fn reachable_from(csr: &Csr, _adj: &[Vec<usize>], start: usize) -> Vec<bool> {
    let levels = algorithms::graph::gpu::bfs_gpu(csr, start);
    levels.into_iter().map(|d| d >= 0).collect()
}

#[cfg(not(feature = "cuda"))]
fn reachable_from(_csr: &Csr, adj: &[Vec<usize>], start: usize) -> Vec<bool> {
    // CPU fallback
    let mut seen = vec![false; adj.len()];
    let mut queue = VecDeque::new();
    seen[start] = true;
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        for &next in &adj[node] {
            if !seen[next] {
                seen[next] = true;
                queue.push_back(next);
            }
        }
    }
    seen
}

fn colorize(name: &str) -> ColoredString {
    if name == "lib" || name == "main" {
        name.cyan().bold()
    } else if name.starts_with("[cycle:") {
        name.yellow().bold()
    } else {
        name.green()
    }
}
// duplicate is_workspace_root removed (kept earlier definition)
