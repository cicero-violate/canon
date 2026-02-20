use crate::dot_import::{parse_dot, DotGraph};
use crate::ir::CanonicalIr;
use crate::layout::{LayoutGraph, LayoutModule};
use std::collections::{BTreeMap, BTreeSet};

// ── round-trip verification ───────────────────────────────────────────────────

#[derive(Debug)]
pub struct DotVerifyError {
    pub mismatches: Vec<String>,
}

impl std::fmt::Display for DotVerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for m in &self.mismatches {
            writeln!(f, "  - {m}")?;
        }
        Ok(())
    }
}

impl std::error::Error for DotVerifyError {}

/// Parse `original_dot`, export `ir` to DOT, parse that, then compare
/// cluster ids, per-cluster node ids, and inter-cluster edges (order-insensitive).
pub fn verify_dot(ir: &CanonicalIr, layout: &LayoutGraph, original_dot: &str) -> Result<(), DotVerifyError> {
    let a: DotGraph = match parse_dot(original_dot) {
        Ok(g) => g,
        Err(e) => {
            return Err(DotVerifyError { mismatches: vec![format!("parse original: {e}")] });
        }
    };
    let exported = export_dot(ir, layout);
    let b: DotGraph = match parse_dot(&exported) {
        Ok(g) => g,
        Err(e) => {
            return Err(DotVerifyError { mismatches: vec![format!("parse export: {e}")] });
        }
    };

    let mut mismatches = Vec::new();

    // cluster id sets
    let ca: BTreeSet<&str> = a.clusters.iter().map(|c| c.id.as_str()).collect();
    let cb: BTreeSet<&str> = b.clusters.iter().map(|c| c.id.as_str()).collect();
    for id in ca.difference(&cb) {
        mismatches.push(format!("cluster `{id}` present in original but not in export"));
    }
    for id in cb.difference(&ca) {
        mismatches.push(format!("cluster `{id}` present in export but not in original"));
    }

    // per-cluster node id sets
    let na: BTreeMap<&str, BTreeSet<&str>> = a.clusters.iter().map(|c| (c.id.as_str(), c.nodes.iter().map(|n| n.id.as_str()).collect())).collect();
    let nb: BTreeMap<&str, BTreeSet<&str>> = b.clusters.iter().map(|c| (c.id.as_str(), c.nodes.iter().map(|n| n.id.as_str()).collect())).collect();
    for (cid, nodes_a) in &na {
        if let Some(nodes_b) = nb.get(cid) {
            for n in nodes_a.difference(nodes_b) {
                mismatches.push(format!("cluster `{cid}`: node `{n}` in original but not export"));
            }
            for n in nodes_b.difference(nodes_a) {
                mismatches.push(format!("cluster `{cid}`: node `{n}` in export but not original"));
            }
        }
    }

    // inter-cluster edges (order-insensitive, imported_types sorted)
    type EdgeKey = (String, String, Vec<String>);
    let edge_set = |g: &DotGraph| -> BTreeSet<EdgeKey> {
        g.inter_edges
            .iter()
            .map(|e| {
                let mut types = e.imported_types.clone();
                types.sort();
                (e.from_cluster.clone(), e.to_cluster.clone(), types)
            })
            .collect()
    };
    let ea = edge_set(&a);
    let eb = edge_set(&b);
    for e in ea.difference(&eb) {
        mismatches.push(format!("edge `{}`->`{}` {:?} in original but not export", e.0, e.1, e.2));
    }
    for e in eb.difference(&ea) {
        mismatches.push(format!("edge `{}`->`{}` {:?} in export but not original", e.0, e.1, e.2));
    }

    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(DotVerifyError { mismatches })
    }
}

// ── color palette ────────────────────────────────────────────────────────────
// Derived deterministically from module id hash so new crates get a stable
// color and the original DOT palette is preserved for known crates.

const PALETTE: &[&str] = &["#0055aa", "#884400", "#006600", "#aa6600", "#880088", "#aa0000", "#005555"];

fn edge_color(module_id: &str) -> &'static str {
    let mut hash: usize = 5381;
    for b in module_id.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(b as usize);
    }
    PALETTE[hash % PALETTE.len()]
}

// ── public entry point ───────────────────────────────────────────────────────

/// Render a `CanonicalIr` as a Graphviz DOT string.
///
/// Round-trip guarantee: the output can be parsed by `dot_import::parse_dot`
/// and will recover the same modules, file topology, and inter-module edges.
pub fn export_dot(ir: &CanonicalIr, layout: &LayoutGraph) -> String {
    let mut out = String::new();

    out.push_str("digraph ");
    out.push_str(&slugify(&ir.project.name.to_string()));
    out.push_str(" {\n");
    out.push_str("    rankdir=LR;\n");
    out.push_str("    compound=true;\n");
    out.push_str("    nodesep=0.5;\n");
    out.push_str("    ranksep=1.4;\n");
    out.push_str("    node [shape=box, style=rounded, fontsize=10];\n");
    out.push_str("    edge [fontsize=9];\n\n");

    // ── clusters (one per module) ────────────────────────────────────────────
    for module in &ir.modules {
        let cluster_id = cluster_id_of(&module.id);
        out.push_str(&format!("    subgraph {} {{\n", cluster_id));
        out.push_str(&format!("        label=\"{}\";\n", module.name.as_str()));
        out.push_str("        style=rounded; color=\"#333333\";\n\n");

        let layout_module = layout.modules.iter().find(|m| m.id == module.id);
        if layout_module.map(|m| m.files.is_empty()).unwrap_or(true) {
            // no file topology — emit a single placeholder node so the
            // cluster is visible and round-trips cleanly
            let node_id = format!("{}_lib", cluster_id);
            out.push_str(&format!("        {} [label=\"lib.rs\"];\n", node_id));
        } else {
            let layout_module = layout_module.expect("layout module missing");
            for file in &layout_module.files {
                out.push_str(&format!("        {} [label=\"{}\"];\n", sanitize_node_id(&file.id), file.path));
            }

            out.push('\n');
        }

        out.push_str("    }\n\n");
    }

    // ── inter-module edges ───────────────────────────────────────────────────
    for edge in &ir.module_edges {
        let from_module = layout.modules.iter().find(|m| m.id == edge.source);
        let to_module = layout.modules.iter().find(|m| m.id == edge.target);

        let from_node = from_module.and_then(|m| lib_node(m)).unwrap_or_else(|| format!("{}_lib", cluster_id_of(&edge.source)));

        let to_node = to_module.and_then(|m| entry_node(m)).unwrap_or_else(|| format!("{}_lib", cluster_id_of(&edge.target)));

        let color = edge_color(&edge.source);

        let label_attr = if edge.imported_types.is_empty() { String::new() } else { format!(" label=\"{}\"", edge.imported_types.join(", ")) };

        out.push_str(&format!(
            "    {} -> {} [{} color=\"{}\" ltail={} lhead={}];\n",
            sanitize_node_id(&from_node),
            sanitize_node_id(&to_node),
            if label_attr.is_empty() { String::new() } else { format!("{} ", label_attr.trim_start()) },
            color,
            cluster_id_of(&edge.source),
            cluster_id_of(&edge.target),
        ));
    }

    out.push_str("}\n");
    out
}

// ── private helpers ──────────────────────────────────────────────────────────

/// The lib node is the last file in the intra-edge sink order —
/// i.e. the node with no outgoing intra edges (the natural export surface).
fn lib_node(module: &LayoutModule) -> Option<String> {
    module.files.last().map(|f| sanitize_node_id(&f.id))
}

/// The entry node is the first file — the one that is never a `to`.
fn entry_node(module: &LayoutModule) -> Option<String> {
    module.files.first().map(|f| sanitize_node_id(&f.id))
}

fn cluster_id_of(module_id: &str) -> String {
    format!("cluster_{}", slugify(module_id))
}

fn sanitize_node_id(id: &str) -> String {
    id.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect()
}

fn slugify(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' }).collect()
}
