//! Minimal API surface for graph_gpu.
//! Shows CSR construction and BFS traversal.
//! Intended as a reference for agents and new contributors.

use graph_gpu::{
    csr::{CsrGraph, EdgeKind, InputEdge},
    traversal::bfs,
    unified::cuda_available,
};

fn main() {
    println!("[0] CUDA available: {}", cuda_available());

    // ── 1. Define nodes (external u64 IDs) ───────────────────────────────────
    // Represents: main -> foo -> bar
    //                  -> baz
    let node_ids: Vec<u64> = vec![
        0x01, // main
        0x02, // foo
        0x03, // bar
        0x04, // baz
        0x05, // unreachable
    ];

    // ── 2. Define edges ───────────────────────────────────────────────────────
    let edges = vec![
        InputEdge { from: 0x01, to: 0x02, kind: EdgeKind::Call },
        InputEdge { from: 0x02, to: 0x03, kind: EdgeKind::Call },
        InputEdge { from: 0x02, to: 0x04, kind: EdgeKind::Call },
        InputEdge { from: 0x03, to: 0x04, kind: EdgeKind::Reference },
    ];

    // ── 3. Build CSR in unified memory ────────────────────────────────────────
    let csr = CsrGraph::build(&node_ids, &edges);
    println!("[1] CSR built: {} nodes, {} edges, on_gpu: {}",
        csr.n_nodes, csr.n_edges, csr.row_offsets.on_gpu());

    // ── 4. Neighbor inspection (CPU) ──────────────────────────────────────────
    let foo_idx = csr.node_index(0x02).unwrap();
    let neighbors = csr.neighbors(foo_idx);
    let kinds     = csr.neighbor_kinds(foo_idx);
    println!("[2] foo neighbors: {:?}  kinds: {:?}", neighbors, kinds);
    assert_eq!(neighbors.len(), 2);

    // ── 5. BFS from main — all Call edges ─────────────────────────────────────
    let main_idx = csr.node_index(0x01).unwrap();
    let result   = bfs(&csr, main_idx, Some(EdgeKind::Call));
    println!("[3] BFS from main (Call only):");
    println!("    max_dist:  {}", result.max_dist);
    println!("    reachable: {:?}", result.reachable());
    // unreachable (0x05) should not appear
    let unreachable_idx = csr.node_index(0x05).unwrap();
    assert_eq!(result.dist[unreachable_idx as usize], -1);
    println!("    0x05 unreachable: ok");

    // ── 6. BFS — no edge filter (all edge kinds) ──────────────────────────────
    let result_all = bfs(&csr, main_idx, None);
    println!("[4] BFS from main (all edges):");
    println!("    reachable: {:?}", result_all.reachable());
    // bar -> baz via Reference, so baz reachable either way
    assert!(result_all.dist[csr.node_index(0x04).unwrap() as usize] >= 0);

    // ── 7. BFS from isolated node ─────────────────────────────────────────────
    let result_iso = bfs(&csr, unreachable_idx, None);
    assert_eq!(result_iso.reachable().len(), 1); // only itself
    println!("[5] isolated node reachable count: {} (self only)", result_iso.reachable().len());

    // ── 8. Point-in-time: rebuild CSR from a subset ───────────────────────────
    // Simulate: only first 3 nodes, 1 edge
    let csr2 = CsrGraph::build(&node_ids[..3], &edges[..1]);
    println!("[6] subset CSR: {} nodes, {} edges", csr2.n_nodes, csr2.n_edges);
    assert_eq!(csr2.n_nodes, 3);
    assert_eq!(csr2.n_edges, 1);

    println!("\nAll checks passed.");
}
