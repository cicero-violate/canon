use std::collections::HashMap;

use thiserror::Error;

use crate::ir::{
    FileEdge, FileNode, ModuleEdge, ModuleId, Proposal, ProposalGoal, ProposalStatus,
    ProposedApi, ProposedEdge, ProposedNode, ProposedNodeKind, Word, WordError,
};

// ── parser types ────────────────────────────────────────────────────────────

/// One subgraph cluster parsed from the DOT source.
#[derive(Debug, Clone)]
pub struct DotCluster {
    pub id: String,
    pub label: String,
    pub nodes: Vec<DotNode>,
    pub intra_edges: Vec<DotIntraEdge>,
}

/// A node inside a cluster (represents a .rs file).
#[derive(Debug, Clone)]
pub struct DotNode {
    pub id: String,
    pub label: String,
}

/// An edge between two nodes within the same cluster.
#[derive(Debug, Clone)]
pub struct DotIntraEdge {
    pub from: String,
    pub to: String,
}

/// An edge between nodes in different clusters.
#[derive(Debug, Clone)]
pub struct DotInterEdge {
    pub from_cluster: String,
    pub to_cluster: String,
    pub imported_types: Vec<String>,
}

/// Full parsed representation of a DOT graph.
#[derive(Debug, Clone)]
pub struct DotGraph {
    pub clusters: Vec<DotCluster>,
    pub inter_edges: Vec<DotInterEdge>,
}

// ── errors ───────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum DotImportError {
    #[error("word error: {0}")]
    Word(#[from] WordError),
    #[error("no clusters found in DOT source")]
    NoClusters,
    #[error("node `{0}` referenced in edge but not declared in any cluster")]
    UnknownNode(String),
    #[error("cluster label `{0}` cannot be converted to a canonical Word")]
    BadClusterLabel(String),
}

// ── public entry point ───────────────────────────────────────────────────────

/// Parse a DOT source string into a `DotGraph`.
pub fn parse_dot(source: &str) -> Result<DotGraph, DotImportError> {
    let mut clusters: Vec<DotCluster> = Vec::new();
    let mut pending_inter: Vec<(String, String, Vec<String>)> = Vec::new();

    // node_id -> cluster_id lookup built as we parse
    let mut node_to_cluster: HashMap<String, String> = HashMap::new();

    let mut current_cluster: Option<DotCluster> = None;

    for raw in source.lines() {
        let line = raw.trim();

        // ── open cluster ────────────────────────────────────────────────────
        if line.starts_with("subgraph cluster_") {
            let cluster_id = line
                .trim_start_matches("subgraph ")
                .trim_end_matches(" {")
                .trim_end_matches('{')
                .trim()
                .to_string();
            current_cluster = Some(DotCluster {
                id: cluster_id,
                label: String::new(),
                nodes: Vec::new(),
                intra_edges: Vec::new(),
            });
            continue;
        }

        // ── cluster label ────────────────────────────────────────────────────
        if let Some(ref mut cluster) = current_cluster {
            if line.starts_with("label=") {
                cluster.label = line
                    .trim_start_matches("label=")
                    .trim_matches('"')
                    .trim_end_matches(';')
                    .to_string();
                continue;
            }
        }

        // ── node declaration: id [label="..."] ───────────────────────────────
        if let Some(ref mut cluster) = current_cluster {
            if line.contains("[label=") && !line.contains("->") {
                if let Some(node) = parse_node_decl(line) {
                    node_to_cluster.insert(node.id.clone(), cluster.id.clone());
                    cluster.nodes.push(node);
                    continue;
                }
            }
        }

        // ── edge line: a -> b [...] ──────────────────────────────────────────
        if line.contains("->") {
            if let Some((from, to, label)) = parse_edge_line(line) {
                pending_inter.push((from, to, label));
            }
            continue;
        }

        // ── close cluster ────────────────────────────────────────────────────
        if line == "}" {
            if let Some(cluster) = current_cluster.take() {
                clusters.push(cluster);
            }
        }
    }

    if clusters.is_empty() {
        return Err(DotImportError::NoClusters);
    }

    // ── resolve edges into intra / inter ────────────────────────────────────
    let mut inter_edges: Vec<DotInterEdge> = Vec::new();

    for (from, to, label_types) in pending_inter {
        let from_cluster = node_to_cluster
            .get(&from)
            .ok_or_else(|| DotImportError::UnknownNode(from.clone()))?
            .clone();
        let to_cluster = node_to_cluster
            .get(&to)
            .ok_or_else(|| DotImportError::UnknownNode(to.clone()))?
            .clone();

        if from_cluster == to_cluster {
            // intra-cluster edge: attach to the cluster
            if let Some(cluster) = clusters.iter_mut().find(|c| c.id == from_cluster) {
                cluster.intra_edges.push(DotIntraEdge { from, to });
            }
        } else {
            // inter-cluster: merge with any existing edge between the same pair
            if let Some(existing) = inter_edges
                .iter_mut()
                .find(|e| e.from_cluster == from_cluster && e.to_cluster == to_cluster)
            {
                for t in label_types {
                    if !existing.imported_types.contains(&t) {
                        existing.imported_types.push(t);
                    }
                }
            } else {
                inter_edges.push(DotInterEdge {
                    from_cluster: from_cluster.clone(),
                    to_cluster: to_cluster.clone(),
                    imported_types: label_types,
                });
            }
        }
    }

    Ok(DotGraph {
        clusters,
        inter_edges,
    })
}

/// Convert a `DotGraph` into a `Proposal` that can be fed into `accept_proposal`.
pub fn dot_graph_to_proposal(graph: &DotGraph, goal: &str) -> Result<Proposal, DotImportError> {
    let mut nodes: Vec<ProposedNode> = Vec::new();
    let mut edges: Vec<ProposedEdge> = Vec::new();
    let mut apis: Vec<ProposedApi> = Vec::new();
    // track emitted trait ids to avoid duplicates
    let mut seen_traits: std::collections::HashSet<String> = std::collections::HashSet::new();

    for cluster in &graph.clusters {
        let label = if cluster.label.is_empty() {
            &cluster.id
        } else {
            &cluster.label
        };
        let name = cluster_label_to_word(label)?;
        let module_id = format!("module.{}", slugify(label));

        nodes.push(ProposedNode {
            id: Some(module_id.clone()),
            name,
            module: None,
            kind: ProposedNodeKind::Module,
        });
    }

    for inter in &graph.inter_edges {
        let from_label = cluster_label(&graph.clusters, &inter.from_cluster);
        let to_label = cluster_label(&graph.clusters, &inter.to_cluster);
        let from_id = format!("module.{}", slugify(from_label));
        let to_id = format!("module.{}", slugify(to_label));
        let rationale = if inter.imported_types.is_empty() {
            format!("{} imports {}", from_label, to_label)
        } else {
            format!(
                "{} imports {} from {}",
                from_label,
                inter.imported_types.join(", "),
                to_label
            )
        };
        edges.push(ProposedEdge {
            from: from_id,
            to: to_id,
            rationale,
        });

        // synthesize one Trait node + ProposedApi per type label on this edge
        for type_label in &inter.imported_types {
            let trait_slug = slugify(type_label);
            let trait_id = format!("trait.{}.{}", slugify(to_label), trait_slug);
            if seen_traits.insert(trait_id.clone()) {
                let trait_name = cluster_label_to_word(type_label)?;
                let to_module_id = format!("module.{}", slugify(to_label));
                nodes.push(ProposedNode {
                    id: Some(trait_id.clone()),
                    name: trait_name,
                    module: Some(to_module_id),
                    kind: ProposedNodeKind::Trait,
                });
                let fn_id = format!("trait_fn.{}.{}", slugify(to_label), trait_slug);
                apis.push(ProposedApi {
                    trait_id,
                    functions: vec![fn_id],
                });
            }
        }
    }

    // enforce_proposal_ready requires non-empty apis — if no type labels
    // existed on any inter-edge, synthesize one stub api from the goal itself
    if apis.is_empty() {
        let first_module_id = graph
            .clusters
            .first()
            .map(|c| {
                let lbl = if c.label.is_empty() { &c.id } else { &c.label };
                format!("module.{}", slugify(lbl))
            })
            .unwrap_or_else(|| format!("module.{}", slugify(goal)));
        let trait_slug = slugify(goal);
        let trait_id = format!("trait.{}.{}", slugify(goal), trait_slug);
        let trait_name = cluster_label_to_word(goal)?;
        nodes.push(ProposedNode {
            id: Some(trait_id.clone()),
            name: trait_name,
            module: Some(first_module_id),
            kind: ProposedNodeKind::Trait,
        });
        apis.push(ProposedApi {
            trait_id,
            functions: vec![format!("trait_fn.{}.{}", slugify(goal), trait_slug)],
        });
    }

    let goal_word = cluster_label_to_word(goal)?;
    let goal_slug = slugify(goal);

    Ok(Proposal {
        id: format!("proposal.dot.{goal_slug}"),
        goal: ProposalGoal {
            id: goal_word,
            description: format!("Imported from DOT: {goal}"),
        },
        nodes,
        apis,
        edges,
        status: ProposalStatus::Submitted,
    })
}

/// Convert a `DotGraph` into the flat IR types needed to populate
/// `Module.files`, `Module.file_edges`, and `ModuleEdge.imported_types`
/// after the proposal has been accepted.
pub fn dot_graph_to_file_topology(
    graph: &DotGraph,
) -> HashMap<String, (Vec<FileNode>, Vec<FileEdge>)> {
    let mut out: HashMap<String, (Vec<FileNode>, Vec<FileEdge>)> = HashMap::new();

    for cluster in &graph.clusters {
        let label = if cluster.label.is_empty() {
            &cluster.id
        } else {
            &cluster.label
        };
        let module_id = format!("module.{}", slugify(label));

        let files: Vec<FileNode> = cluster
            .nodes
            .iter()
            .map(|n| FileNode {
                id: n.id.clone(),
                name: n.label.clone(),
            })
            .collect();

        let file_edges: Vec<FileEdge> = cluster
            .intra_edges
            .iter()
            .map(|e| FileEdge {
                from: e.from.clone(),
                to: e.to.clone(),
            })
            .collect();

        out.insert(module_id, (files, file_edges));
    }

    out
}

/// Extract `imported_types` per `(from_module_id, to_module_id)` pair,
/// ready to be patched onto `ModuleEdge` entries after acceptance.
pub fn dot_graph_to_imported_types(
    graph: &DotGraph,
) -> HashMap<(String, String), Vec<String>> {
    let mut out: HashMap<(String, String), Vec<String>> = HashMap::new();

    for inter in &graph.inter_edges {
        let from_label = cluster_label(&graph.clusters, &inter.from_cluster);
        let to_label = cluster_label(&graph.clusters, &inter.to_cluster);
        let from_id = format!("module.{}", slugify(from_label));
        let to_id = format!("module.{}", slugify(to_label));
        out.insert((from_id, to_id), inter.imported_types.clone());
    }

    out
}

// ── private helpers ──────────────────────────────────────────────────────────

fn parse_node_decl(line: &str) -> Option<DotNode> {
    // e.g.  mc_hash  [label="hash.rs"];
    let bracket = line.find('[')?;
    let id = line[..bracket].trim().to_string();
    if id.is_empty() {
        return None;
    }
    let label = extract_attr(line, "label").unwrap_or_else(|| id.clone());
    Some(DotNode { id, label })
}

fn parse_edge_line(line: &str) -> Option<(String, String, Vec<String>)> {
    // e.g.  mc_hash -> mc_tree;
    // e.g.  mc_lib -> ps_partition  [label="MerkleNode, Hash" ...];
    let arrow = line.find("->")?;
    let from = line[..arrow].trim().to_string();
    let rest = line[arrow + 2..].trim();

    let (to_raw, label_str) = if let Some(bracket) = rest.find('[') {
        (&rest[..bracket], extract_attr(rest, "label").unwrap_or_default())
    } else {
        (rest, String::new())
    };

    let to = to_raw.trim().trim_end_matches(';').to_string();
    if from.is_empty() || to.is_empty() {
        return None;
    }

    let types: Vec<String> = label_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    Some((from, to, types))
}

fn extract_attr(line: &str, attr: &str) -> Option<String> {
    // finds attr="value" in a line
    let key = format!("{attr}=\"");
    let start = line.find(&key)? + key.len();
    let end = line[start..].find('"')? + start;
    Some(line[start..end].to_string())
}

fn cluster_label<'a>(clusters: &'a [DotCluster], cluster_id: &'a str) -> &'a str {
    clusters
        .iter()
        .find(|c| c.id == cluster_id)
        .map_or(cluster_id, |c| {
            if c.label.is_empty() {
                c.id.as_str()
            } else {
                c.label.as_str()
            }
        })
}

fn cluster_label_to_word(label: &str) -> Result<Word, DotImportError> {
    let mut builder = String::new();
    for ch in label.chars() {
        if ch == '-' || ch == '_' || ch == ' ' {
            continue;
        }
        if !ch.is_ascii_alphanumeric() {
            continue;
        }
        if builder.is_empty() {
            builder.push(ch.to_ascii_uppercase());
        } else {
            builder.push(ch);
        }
    }
    if builder.is_empty() {
        return Err(DotImportError::BadClusterLabel(label.to_string()));
    }
    Word::new(builder).map_err(DotImportError::Word)
}

fn slugify(label: &str) -> String {
    label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}
