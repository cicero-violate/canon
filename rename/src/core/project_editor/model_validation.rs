use anyhow::{anyhow, Result};
use database::graph_log::GraphSnapshot;
use std::collections::HashSet;

pub(crate) fn validate_model0(snapshot: &mut GraphSnapshot) -> Result<()> {
    validate_identity_integrity(snapshot)?;
    validate_referential_integrity(snapshot)?;
    validate_ordering(snapshot)?;
    validate_csr(snapshot)?;
    validate_visibility(snapshot)?;
    Ok(())
}

fn validate_identity_integrity(snapshot: &GraphSnapshot) -> Result<()> {
    let mut node_ids = HashSet::with_capacity(snapshot.nodes.len());
    for node in &snapshot.nodes {
        if !node_ids.insert(node.id.clone()) {
            return Err(anyhow!("duplicate node id in snapshot"));
        }
    }
    let mut edge_ids = HashSet::with_capacity(snapshot.edges.len());
    for edge in &snapshot.edges {
        if !edge_ids.insert(edge.id.clone()) {
            return Err(anyhow!("duplicate edge id in snapshot"));
        }
    }
    Ok(())
}

fn validate_referential_integrity(snapshot: &GraphSnapshot) -> Result<()> {
    let node_ids: HashSet<_> = snapshot.nodes.iter().map(|n| n.id.clone()).collect();
    for edge in &snapshot.edges {
        if !node_ids.contains(&edge.from) {
            return Err(anyhow!("edge references missing from-node"));
        }
        if !node_ids.contains(&edge.to) {
            return Err(anyhow!("edge references missing to-node"));
        }
    }
    Ok(())
}

fn validate_ordering(snapshot: &GraphSnapshot) -> Result<()> {
    let mut prev: Option<&[u8; 16]> = None;
    for node in &snapshot.nodes {
        let id = &node.id.0;
        if let Some(prev) = prev {
            if prev > id {
                return Err(anyhow!("node ordering is not stable (ids not sorted)"));
            }
        }
        prev = Some(id);
    }
    let mut prev: Option<&[u8; 16]> = None;
    for edge in &snapshot.edges {
        let id = &edge.id.0;
        if let Some(prev) = prev {
            if prev > id {
                return Err(anyhow!("edge ordering is not stable (ids not sorted)"));
            }
        }
        prev = Some(id);
    }
    Ok(())
}

fn validate_csr(snapshot: &mut GraphSnapshot) -> Result<()> {
    let node_len = snapshot.nodes.len();
    let edge_len = snapshot.edges.len();
    let csr = snapshot.csr();
    if csr.row_ptr.len() != node_len + 1 {
        return Err(anyhow!("csr row_ptr length mismatch"));
    }
    if csr.col_idx.len() != edge_len {
        return Err(anyhow!("csr col_idx length mismatch"));
    }
    let mut prev = None;
    for &v in &csr.row_ptr {
        if let Some(prev) = prev {
            if v < prev {
                return Err(anyhow!("csr row_ptr is decreasing"));
            }
        }
        prev = Some(v);
    }
    let node_count = node_len as i32;
    for &idx in &csr.col_idx {
        if idx < 0 || idx >= node_count {
            return Err(anyhow!("csr col_idx out of bounds"));
        }
    }
    Ok(())
}

fn validate_visibility(snapshot: &GraphSnapshot) -> Result<()> {
    for node in &snapshot.nodes {
        if !emits_rust_syntax(node) {
            continue;
        }
        let Some(value) = node.metadata.get("visibility") else {
            return Err(anyhow!("missing visibility metadata"));
        };
        if value.is_empty() {
            return Err(anyhow!("empty visibility metadata"));
        }
        if value == "public" || value == "crate" || value == "private" {
            continue;
        }
        if let Some(rest) = value.strip_prefix("restricted:") {
            if rest.is_empty() {
                return Err(anyhow!("restricted visibility missing path"));
            }
            continue;
        }
        return Err(anyhow!("unknown visibility value: {value}"));
    }
    Ok(())
}

fn emits_rust_syntax(node: &database::graph_log::WireNode) -> bool {
    let node_kind = node.metadata.get("node_kind").map(|s| s.as_str()).unwrap_or("");
    matches!(
        node_kind,
        "module"
            | "struct"
            | "enum"
            | "union"
            | "trait"
            | "impl"
            | "function"
            | "const"
            | "static"
            | "type_alias"
    )
}
