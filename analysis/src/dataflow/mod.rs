//! Reaching definitions and use/def chains.
//!
//! Delegates to algorithms::control_flow::dataflow, with optional GPU offload.

pub use algorithms::control_flow::dataflow::{BlockId, DataflowFacts, DataflowResult, DefId};

use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::collections::HashSet;

pub fn reaching_definitions(blocks: &[BlockId], pred: &HashMap<BlockId, Vec<BlockId>>, facts: &DataflowFacts) -> DataflowResult {
    if should_use_gpu(blocks, pred, facts) {
        #[cfg(feature = "cuda")]
        {
            if let Some(res) = reaching_definitions_gpu(blocks, pred, facts) {
                return res;
            }
        }
    }
    algorithms::control_flow::dataflow::reaching_definitions(blocks, pred, facts)
}

fn should_use_gpu(blocks: &[BlockId], pred: &HashMap<BlockId, Vec<BlockId>>, facts: &DataflowFacts) -> bool {
    let edges: usize = pred.values().map(|v| v.len()).sum();
    let defs: usize = facts.r#gen.values().map(|s| s.len()).sum();
    blocks.len().saturating_mul(edges + defs) > 1_000_000
}

#[cfg(feature = "cuda")]
fn reaching_definitions_gpu(blocks: &[BlockId], pred: &HashMap<BlockId, Vec<BlockId>>, facts: &DataflowFacts) -> Option<DataflowResult> {
    use algorithms::control_flow::gpu::reaching_definitions_gpu as gpu;
    use algorithms::graph::csr::Csr;

    let max_block = blocks.iter().copied().max().unwrap_or(0);
    if max_block + 1 != blocks.len() {
        return None;
    }
    let mut seen = vec![false; blocks.len()];
    for &b in blocks {
        if b >= seen.len() {
            return None;
        }
        seen[b] = true;
    }
    if seen.iter().any(|v| !*v) {
        return None;
    }

    // Build def index.
    let mut all_defs: Vec<String> = facts.r#gen.values().flat_map(|s| s.iter().cloned()).collect();
    all_defs.sort();
    all_defs.dedup();
    let def_count = all_defs.len();
    let mut def_index: HashMap<String, usize> = HashMap::new();
    for (i, d) in all_defs.iter().enumerate() {
        def_index.insert(d.clone(), i);
    }
    let words = (def_count + 63) / 64;

    let mut r#gen = vec![0u64; blocks.len() * words];
    let mut kill = vec![0u64; blocks.len() * words];
    for (&b, defs) in &facts.r#gen {
        for d in defs {
            if let Some(&i) = def_index.get(d) {
                r#gen[b * words + (i >> 6)] |= 1u64 << (i & 63);
            }
        }
    }
    for (&b, defs) in &facts.kill {
        for d in defs {
            if let Some(&i) = def_index.get(d) {
                kill[b * words + (i >> 6)] |= 1u64 << (i & 63);
            }
        }
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); blocks.len()];
    for (&b, preds) in pred {
        for &p in preds {
            if p < blocks.len() {
                adj[b].push(p);
            }
        }
    }
    let pred_csr = Csr::from_adj(&adj);
    let out_bits = gpu(&pred_csr, blocks.len(), def_count, &r#gen, &kill);

    let mut out: HashMap<BlockId, HashSet<DefId>> = HashMap::new();
    let mut r#in: HashMap<BlockId, HashSet<DefId>> = HashMap::new();
    for b in blocks {
        out.insert(*b, HashSet::new());
        r#in.insert(*b, HashSet::new());
    }
    for b in 0..blocks.len() {
        let mut set = HashSet::new();
        for i in 0..def_count {
            let word = out_bits[b * words + (i >> 6)];
            if (word >> (i & 63)) & 1 == 1 {
                set.insert(all_defs[i].clone());
            }
        }
        out.insert(b, set);
    }
    Some(DataflowResult { r#in, out })
}
