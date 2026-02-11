//! Kernel fusion analyzer.
//!
//! Detects producer → consumer GPU pipelines that are safe to fuse
//! (single-use edges, pure math kernels).

use std::collections::{HashMap, HashSet};

use crate::ir::{CanonicalIr, GpuFunctionId};

/// Producer/consumer pair that can be fused into a single GPU kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionCandidate {
    pub producer_gpu: GpuFunctionId,
    pub consumer_gpu: GpuFunctionId,
    pub producer_function: String,
    pub consumer_function: String,
}

/// Scan the call graph for eligible fusion pairs.
pub fn analyze_fusion_candidates(ir: &CanonicalIr) -> Vec<FusionCandidate> {
    let mut function_to_gpu: HashMap<&str, &crate::ir::GpuFunction> = HashMap::new();
    for gpu in &ir.gpu_functions {
        if gpu.properties.pure
            && gpu.properties.no_io
            && gpu.properties.no_alloc
            && gpu.properties.no_branch
        {
            function_to_gpu.insert(gpu.function.as_str(), gpu);
        }
    }
    if function_to_gpu.is_empty() || ir.call_edges.is_empty() {
        return Vec::new();
    }

    let mut outgoing: HashMap<&str, HashSet<&str>> = HashMap::new();
    let mut incoming: HashMap<&str, HashSet<&str>> = HashMap::new();
    for edge in &ir.call_edges {
        outgoing
            .entry(edge.caller.as_str())
            .or_default()
            .insert(edge.callee.as_str());
        incoming
            .entry(edge.callee.as_str())
            .or_default()
            .insert(edge.caller.as_str());
    }

    let mut candidates = Vec::new();
    for edge in &ir.call_edges {
        let producer_fn = edge.caller.as_str();
        let consumer_fn = edge.callee.as_str();
        let Some(producer_gpu) = function_to_gpu.get(producer_fn) else {
            continue;
        };
        let Some(consumer_gpu) = function_to_gpu.get(consumer_fn) else {
            continue;
        };

        // Require exclusive edge between producer and consumer.
        if outgoing
            .get(producer_fn)
            .map(|set| set.len() == 1)
            .unwrap_or(false)
            && incoming
                .get(consumer_fn)
                .map(|set| set.len() == 1)
                .unwrap_or(false)
        {
            candidates.push(FusionCandidate {
                producer_gpu: producer_gpu.id.clone(),
                consumer_gpu: consumer_gpu.id.clone(),
                producer_function: producer_gpu.function.clone(),
                consumer_function: consumer_gpu.function.clone(),
            });
        }
    }

    candidates
}
