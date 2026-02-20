//! Minimal API surface for database.
//! Shows every major capability in one file.
//! Intended as a reference for agents and new contributors.

use database::{
    delta::{Delta, Source},
    epoch::Epoch,
    graph_log::{GraphDelta, WireEdge, WireEdgeId, WireNode, WireNodeId},
    primitives::{DeltaID, PageID},
    MemoryEngine, MemoryEngineConfig, MemoryTransition,
};
use std::collections::BTreeMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join("database_minimal_api");
    std::fs::create_dir_all(&dir)?;
    let tlog_path = dir.join("state.tlog");
    if tlog_path.exists() {
        std::fs::remove_file(&tlog_path)?;
    }
    let graph_log_path = tlog_path.with_extension("graph.log");
    if graph_log_path.exists() {
        std::fs::remove_file(&graph_log_path)?;
    }

    let engine = MemoryEngine::new(MemoryEngineConfig { tlog_path })?;

    // ── 1. Genesis state ─────────────────────────────────────────────────────
    let root0 = engine.genesis();
    println!("[1] genesis root:      {:?}", &root0[..4]);

    // ── 2. Commit a page delta ────────────────────────────────────────────────
    let payload = vec![1u8; 4096];
    let mask = vec![true; 4096];
    let delta = Delta::new_dense(DeltaID(1), PageID(0), Epoch(0), payload, mask, Source("minimal_api".into()))?;
    let (root1, proof) = engine.step(root0, delta)?;
    println!("[2] after page delta:  {:?}", &root1[..4]);
    println!("    commit proof hash: {:?}", &proof.state_hash[..4]);

    // ── 3. Merkle root is deterministic ──────────────────────────────────────
    assert_eq!(engine.current_root_hash(), root1);
    println!("[3] root stable:       ok");

    // ── 4. Persist graph nodes ────────────────────────────────────────────────
    let node_a = WireNode { id: WireNodeId([1u8; 16]), key: "crate::foo::bar".into(), label: "bar".into(), metadata: BTreeMap::new() };
    let node_b = WireNode { id: WireNodeId([2u8; 16]), key: "crate::foo::baz".into(), label: "baz".into(), metadata: BTreeMap::new() };
    engine.commit_graph_delta(GraphDelta::AddNode(node_a))?;
    engine.commit_graph_delta(GraphDelta::AddNode(node_b))?;
    println!("[4] graph nodes committed");

    // ── 5. Persist a graph edge ───────────────────────────────────────────────
    let edge = WireEdge { id: WireEdgeId([3u8; 16]), from: WireNodeId([1u8; 16]), to: WireNodeId([2u8; 16]), kind: "call".into(), metadata: BTreeMap::new() };
    engine.commit_graph_delta(GraphDelta::AddEdge(edge))?;
    println!("[5] graph edge committed");

    // ── 6. How many graph deltas on disk ──────────────────────────────────────
    let count = engine.graph_delta_count();
    println!("[6] graph delta count: {}", count);
    assert_eq!(count, 3);

    // ── 7. Materialize full graph from log ────────────────────────────────────
    let mut snapshot = engine.materialized_graph()?;
    println!("[7] nodes: {}  edges: {}", snapshot.nodes.len(), snapshot.edges.len());
    assert_eq!(snapshot.nodes.len(), 2);
    assert_eq!(snapshot.edges.len(), 1);
    let levels = snapshot.bfs_gpu(WireNodeId([1u8; 16]));
    println!("[7b] bfs from node_a: {:?}", levels);

    // ── 8. Point-in-time replay — only first 2 deltas (nodes only) ────────────
    let snapshot_at_2 = engine.graph_snapshot_at(2)?;
    println!("[8] snapshot@2 nodes: {}  edges: {}", snapshot_at_2.nodes.len(), snapshot_at_2.edges.len());
    assert_eq!(snapshot_at_2.nodes.len(), 2);
    assert_eq!(snapshot_at_2.edges.len(), 0);

    // ── 9. Checkpoint page state to disk ─────────────────────────────────────
    let ckpt = dir.join("state.checkpoint");
    engine.checkpoint(&ckpt)?;
    println!("[9] checkpoint written: {}", ckpt.display());

    println!("\nAll checks passed.");
    Ok(())
}
