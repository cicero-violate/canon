use crate::ir::CanonicalIr;

// ── color palette ────────────────────────────────────────────────────────────
// Derived deterministically from module id hash so new crates get a stable
// color and the original DOT palette is preserved for known crates.

const PALETTE: &[&str] = &[
    "#0055aa",
    "#884400",
    "#006600",
    "#aa6600",
    "#880088",
    "#aa0000",
    "#005555",
];

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
pub fn export_dot(ir: &CanonicalIr) -> String {
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
        out.push_str(&format!(
            "        label=\"{}\";\n",
            module.name.as_str()
        ));
        out.push_str("        style=rounded; color=\"#333333\";\n\n");

        if module.files.is_empty() {
            // no file topology — emit a single placeholder node so the
            // cluster is visible and round-trips cleanly
            let node_id = format!("{}_lib", cluster_id);
            out.push_str(&format!(
                "        {} [label=\"lib.rs\"];\n",
                node_id
            ));
        } else {
            for file in &module.files {
                out.push_str(&format!(
                    "        {} [label=\"{}\"];\n",
                    sanitize_node_id(&file.id),
                    file.name
                ));
            }

            out.push('\n');

            for edge in &module.file_edges {
                out.push_str(&format!(
                    "        {} -> {};\n",
                    sanitize_node_id(&edge.from),
                    sanitize_node_id(&edge.to)
                ));
            }
        }

        out.push_str("    }\n\n");
    }

    // ── inter-module edges ───────────────────────────────────────────────────
    for edge in &ir.module_edges {
        let from_module = ir.modules.iter().find(|m| m.id == edge.source);
        let to_module = ir.modules.iter().find(|m| m.id == edge.target);

        let from_node = from_module
            .and_then(|m| lib_node(m))
            .unwrap_or_else(|| format!("{}_lib", cluster_id_of(&edge.source)));

        let to_node = to_module
            .and_then(|m| entry_node(m))
            .unwrap_or_else(|| format!("{}_lib", cluster_id_of(&edge.target)));

        let color = edge_color(&edge.source);

        let label_attr = if edge.imported_types.is_empty() {
            String::new()
        } else {
            format!(" label=\"{}\"", edge.imported_types.join(", "))
        };

        out.push_str(&format!(
            "    {} -> {} [{} color=\"{}\" ltail={} lhead={}];\n",
            sanitize_node_id(&from_node),
            sanitize_node_id(&to_node),
            if label_attr.is_empty() {
                String::new()
            } else {
                format!("{} ", label_attr.trim_start())
            },
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
fn lib_node(module: &crate::ir::Module) -> Option<String> {
    if module.files.is_empty() {
        return None;
    }
    // find a file that is never a `from` in file_edges — it is the sink
    let froms: std::collections::HashSet<&str> =
        module.file_edges.iter().map(|e| e.from.as_str()).collect();
    let sink = module
        .files
        .iter()
        .find(|f| !froms.contains(f.id.as_str()));
    sink.map(|f| sanitize_node_id(&f.id))
}

/// The entry node is the first file — the one that is never a `to`.
fn entry_node(module: &crate::ir::Module) -> Option<String> {
    if module.files.is_empty() {
        return None;
    }
    let tos: std::collections::HashSet<&str> =
        module.file_edges.iter().map(|e| e.to.as_str()).collect();
    let source = module
        .files
        .iter()
        .find(|f| !tos.contains(f.id.as_str()));
    source.map(|f| sanitize_node_id(&f.id))
}

fn cluster_id_of(module_id: &str) -> String {
    format!("cluster_{}", slugify(module_id))
}

fn sanitize_node_id(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

fn slugify(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}
