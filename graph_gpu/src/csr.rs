//! CSR (Compressed Sparse Row) graph representation in unified memory.
//!
//! Layout:
//!   row_offsets : [u32; N+1]  — row_offsets[i]..row_offsets[i+1] = neighbors of i
//!   col_indices : [u32; E]    — neighbor node indices
//!   node_ids    : [u64; N]    — stable external node IDs (WireNodeId bytes as u64)
//!   edge_kinds  : [u8;  E]    — edge kind per col_index entry
use crate::unified::UnifiedVec;
use std::collections::HashMap;

/// Edge kind encoding (matches state/graph.rs EdgeKind).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Contains = 0,
    Call = 1,
    ControlFlow = 2,
    Reference = 3,
}

impl EdgeKind {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Call,
            2 => Self::ControlFlow,
            3 => Self::Reference,
            _ => Self::Contains,
        }
    }
}

/// Input edge for the CSR builder.
#[derive(Debug, Clone)]
pub struct InputEdge {
    pub from: u64, // external node id
    pub to: u64,
    pub kind: EdgeKind,
}

/// CSR graph in unified memory — zero-copy CPU/GPU.
pub struct CsrGraph {
    /// row_offsets[i]..row_offsets[i+1] = neighbors of node i
    pub row_offsets: UnifiedVec<u32>,
    /// Neighbor node indices (column indices)
    pub col_indices: UnifiedVec<u32>,
    /// Edge kind per col_index
    pub edge_kinds: UnifiedVec<u8>,
    /// External node IDs indexed by internal index
    pub node_ids: UnifiedVec<u64>,
    /// Number of nodes
    pub n_nodes: usize,
    /// Number of edges
    pub n_edges: usize,
}

impl CsrGraph {
    /// Build a CSR graph from a node id list and edge list.
    /// Nodes are assigned internal indices 0..N in the order given.
    pub fn build(node_ids: &[u64], edges: &[InputEdge]) -> Self {
        let n = node_ids.len();
        let e = edges.len();

        // Map external id -> internal index
        let id_map: HashMap<u64, u32> = node_ids.iter().enumerate().map(|(i, &id)| (id, i as u32)).collect();

        // Count out-degree per node
        let mut degree = vec![0u32; n];
        let mut valid_edges: Vec<(u32, u32, u8)> = Vec::with_capacity(e);
        for edge in edges {
            if let (Some(&from), Some(&to)) = (id_map.get(&edge.from), id_map.get(&edge.to)) {
                degree[from as usize] += 1;
                valid_edges.push((from, to, edge.kind as u8));
            }
        }

        // Prefix sum -> row_offsets
        let mut row_offsets = UnifiedVec::<u32>::with_capacity(n + 1);
        let mut offset = 0u32;
        for i in 0..n {
            row_offsets.push(offset);
            offset += degree[i];
        }
        row_offsets.push(offset); // sentinel

        let e_valid = valid_edges.len();
        let mut col_indices = UnifiedVec::<u32>::with_capacity(e_valid);
        let mut edge_kinds = UnifiedVec::<u8>::with_capacity(e_valid);

        // Temporary write cursors
        let mut cursor = row_offsets.as_slice()[..n].to_vec();

        // Allocate full slices before writing
        // Fill with placeholder first so len == capacity
        col_indices.fill(0u32);
        edge_kinds.fill(0u8);

        for (from, to, kind) in &valid_edges {
            let pos = cursor[*from as usize] as usize;
            col_indices.as_mut_slice()[pos] = *to;
            edge_kinds.as_mut_slice()[pos] = *kind;
            cursor[*from as usize] += 1;
        }

        // node_ids array
        let mut nid_buf = UnifiedVec::<u64>::with_capacity(n);
        for &id in node_ids {
            nid_buf.push(id);
        }

        Self { row_offsets, col_indices, edge_kinds, node_ids: nid_buf, n_nodes: n, n_edges: e_valid }
    }

    /// Internal index for an external node id, if present.
    pub fn node_index(&self, external_id: u64) -> Option<u32> {
        self.node_ids.as_slice().iter().position(|&id| id == external_id).map(|i| i as u32)
    }

    /// Neighbors of internal node index `i` (as internal indices).
    pub fn neighbors(&self, i: u32) -> &[u32] {
        let offsets = self.row_offsets.as_slice();
        let start = offsets[i as usize] as usize;
        let end = offsets[i as usize + 1] as usize;
        &self.col_indices.as_slice()[start..end]
    }

    /// Edge kinds for neighbors of node `i`.
    pub fn neighbor_kinds(&self, i: u32) -> &[u8] {
        let offsets = self.row_offsets.as_slice();
        let start = offsets[i as usize] as usize;
        let end = offsets[i as usize + 1] as usize;
        &self.edge_kinds.as_slice()[start..end]
    }
}
