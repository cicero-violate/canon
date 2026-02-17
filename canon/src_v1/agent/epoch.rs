//! Epoch-boundary meta-tick auto-fire.
//!
//! Isolated here so runtime::tick_executor does not need to import agent types,
//! breaking the tick_executor ↔ agent cycle.

use crate::agent::capability::CapabilityGraph;
use crate::agent::meta::run_meta_tick;
use crate::agent::reward::{NodeOutcome, RewardLedger};
use crate::ir::CanonicalIr;

/// Checks whether the completed tick pushed an epoch boundary, and if so
/// auto-fires run_meta_tick() against a default empty CapabilityGraph.
///
/// The interval N is derived from the most recent PolicyParameters entry
/// via meta_tick_interval(). If no policy exists the interval defaults to 10.
///
/// The mutated graph is discarded here (stateless fire-and-forget); callers
/// that need the graph should use run_meta_tick() directly with a live graph.
pub fn maybe_fire_meta_tick(ir: &CanonicalIr, epoch_count: usize, tick_reward: f64, tick_id: &str) {
    if epoch_count == 0 {
        return;
    }

    let interval = ir
        .policy_parameters
        .last()
        .map(|p| p.meta_tick_interval())
        .unwrap_or(10) as usize;

    if epoch_count % interval != 0 {
        return;
    }

    let mut ledger = RewardLedger::new(0.1, 0.5);
    for record in &ir.reward_deltas {
        ledger.record(
            record.tick.clone(),
            NodeOutcome::Accepted {
                reward: record.reward,
            },
        );
    }

    let default_graph = CapabilityGraph::default();
    match run_meta_tick(&default_graph, &ledger) {
        Ok(meta_result) => {
            eprintln!(
                "[meta-tick] epoch={epoch_count} interval={interval} tick={tick_id} reward={tick_reward:.4} applied={} rejected={}",
                meta_result.applied.len(),
                meta_result.rejected.len(),
            );
        }
        Err(e) => {
            eprintln!("[meta-tick] epoch={epoch_count} error={e}");
        }
    }
}
