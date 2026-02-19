//! Linux state capture that records facts into the state graph.

use crate::state::capability::shell::{probe_fact, LinuxFact};
use crate::state::builder::{KernelGraphBuilder, NodePayload};
use crate::state::graph::GraphSnapshot;
use std::collections::BTreeMap;

/// Captures the provided facts and stores true facts into a GraphSnapshot.
pub fn capture_linux_state(facts: &[LinuxFact]) -> GraphSnapshot {
    let mut builder = KernelGraphBuilder::new();
    for fact in facts {
        if probe_fact(fact) {
            let (key, label, metadata) = fact_to_node(fact);
            let mut payload = NodePayload::new(key, label);
            for (k, v) in metadata {
                payload = payload.with_metadata(k, v);
            }
            let _ = builder.add_node(payload);
        }
    }
    builder.finalize()
}

fn fact_to_node(fact: &LinuxFact) -> (String, String, BTreeMap<String, String>) {
    let mut metadata = BTreeMap::new();
    match fact {
        LinuxFact::Exists(path) => {
            metadata.insert("path".into(), path.display().to_string());
            (
                format!("linux::exists::{}", path.display()),
                "exists".into(),
                metadata,
            )
        }
        LinuxFact::File(path) => {
            metadata.insert("path".into(), path.display().to_string());
            (
                format!("linux::file::{}", path.display()),
                "file".into(),
                metadata,
            )
        }
        LinuxFact::Dir(path) => {
            metadata.insert("path".into(), path.display().to_string());
            (
                format!("linux::dir::{}", path.display()),
                "dir".into(),
                metadata,
            )
        }
        LinuxFact::ProcessRunning(name) => {
            metadata.insert("process".into(), name.clone());
            (
                format!("linux::proc::{name}"),
                "process_running".into(),
                metadata,
            )
        }
        LinuxFact::BinaryInstalled(name) => {
            metadata.insert("binary".into(), name.clone());
            (
                format!("linux::bin::{name}"),
                "binary_installed".into(),
                metadata,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_generates_nodes_for_true_facts() {
        let facts = vec![LinuxFact::BinaryInstalled("sh".into())];
        let snapshot = capture_linux_state(&facts);
        assert!(!snapshot.nodes().is_empty());
    }
}
