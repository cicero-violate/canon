use database::graph_log::{GraphSnapshot, WireEdge, WireEdgeId, WireNode, WireNodeId};
use std::collections::BTreeMap;

fn node(id: u8, key: &str) -> WireNode {
    WireNode {
        id: WireNodeId([id; 16]),
        key: key.to_string(),
        label: key.to_string(),
        metadata: BTreeMap::new(),
    }
}

fn edge(id: u8, from: u8, to: u8, kind: &str) -> WireEdge {
    WireEdge {
        id: WireEdgeId([id; 16]),
        from: WireNodeId([from; 16]),
        to: WireNodeId([to; 16]),
        kind: kind.to_string(),
        metadata: BTreeMap::new(),
    }
}

fn main() {
    // Build a tiny directed graph:
    // 0 -> 1, 0 -> 2, 1 -> 2, 2 -> 3
    let nodes = vec![
        node(0, "n0"),
        node(1, "n1"),
        node(2, "n2"),
        node(3, "n3"),
    ];
    let edges = vec![
        edge(10, 0, 1, "edge"),
        edge(11, 0, 2, "edge"),
        edge(12, 1, 2, "edge"),
        edge(13, 2, 3, "edge"),
    ];

    let mut snapshot = GraphSnapshot::new(nodes, edges);

    let csr = snapshot.csr();
    println!("CSR row_ptr: {:?}", csr.row_ptr);
    println!("CSR col_idx: {:?}", csr.col_idx);

    // GPU BFS requires the `cuda` feature.
    // Start from node 0.
    let levels = snapshot.bfs_gpu(WireNodeId([0; 16]));
    println!("BFS levels from n0: {:?}", levels);
}
