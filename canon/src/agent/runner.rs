//! Top-level agent runner — drives the full Tier 7 loop.
//!
//! Loop per tick:
//!   1. AgentCallDispatcher::dispatch_order() → ordered node list
//!   2. For each node: dispatcher.dispatch() → AgentCallInput
//!                     call_llm()             → AgentCallOutput
//!                     dispatcher.record_output()
//!   3. run_pipeline(ir, layout, proposal, stage_outputs) → PipelineResult
//!   4. record_pipeline_outcome(ledger, node_id, result)
//!   5. Every meta_tick_interval ticks: run_meta_tick(graph, ledger) → Φ(G)
//!   6. Every policy_update_interval ticks: ledger.update_policy()
//!
//! No LLM calls inside this file — delegated entirely to llm_provider::call_llm().
//! Async — requires a tokio runtime. WsBridge is passed in from main.

use std::path::{Path, PathBuf};
use std::fs;

use crate::ir::CanonicalIr;
use crate::layout::LayoutGraph;
use crate::io_utils::{load_capability_graph, save_capability_graph};

use super::call::AgentCallOutput;
use super::capability::CapabilityGraph;
use super::dispatcher::AgentCallDispatcher;
use super::llm_provider::{LlmProviderError, call_llm};
use super::ws_server::WsBridge;
use super::meta::run_meta_tick;
use super::pipeline::{PipelineError, run_pipeline, record_pipeline_outcome};
use super::refactor::{RefactorKind, RefactorProposal, RefactorTarget};
use super::reward::RewardLedger;

/// Configuration for one agent run session.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// How many pipeline ticks to run before stopping (0 = run forever).
    pub max_ticks: u64,
    /// Fire meta-tick every N ticks (0 = never).
    pub meta_tick_interval: u64,
    /// Update policy every N ticks (0 = never).
    pub policy_update_interval: u64,
    /// EMA alpha for reward ledger.
    pub ledger_alpha: f64,
    /// Base trust threshold for dispatcher.
    pub base_trust_threshold: f64,
    /// Path to save the capability graph after each meta-tick.
    pub graph_out: PathBuf,
    /// Path to save the ledger after each tick.
    pub ledger_out: PathBuf,
    /// Path to save the mutated IR after each successful pipeline run.
    pub ir_out: PathBuf,
}

impl RunnerConfig {
    pub fn new(graph_out: PathBuf, ledger_out: PathBuf, ir_out: PathBuf) -> Self {
        Self {
            max_ticks: 0,
            meta_tick_interval: 10,
            policy_update_interval: 5,
            ledger_alpha: 0.1,
            base_trust_threshold: 0.5,
            graph_out,
            ledger_out,
            ir_out,
        }
    }
}

/// Errors from the agent runner.
#[derive(Debug)]
pub enum RunnerError {
    /// Dispatcher could not produce a call order.
    Dispatch(crate::agent::call::AgentCallError),
    /// LLM provider failed for a node.
    Llm { node_id: String, error: LlmProviderError },
    /// No stage outputs were collected (empty graph).
    NoOutputs,
    /// I/O error persisting graph or ledger.
    Io(String),
}

impl std::fmt::Display for RunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunnerError::Dispatch(e) => write!(f, "dispatch error: {e}"),
            RunnerError::Llm { node_id, error } => {
                write!(f, "llm error on node {node_id}: {error}")
            }
            RunnerError::NoOutputs => write!(f, "no stage outputs collected — graph may be empty"),
            RunnerError::Io(e) => write!(f, "i/o error: {e}"),
        }
    }
}

impl std::error::Error for RunnerError {}

/// Statistics for one completed tick.
#[derive(Debug)]
pub struct TickStats {
    pub tick_number: u64,
    pub nodes_called: usize,
    pub llm_errors: usize,
    pub pipeline_reward: Option<f64>,
    pub pipeline_error: Option<String>,
    pub meta_tick_fired: bool,
    pub policy_updated: bool,
}

/// Runs the Tier 7 agent loop.
///
/// `ir` and `layout` are the starting state — they are updated in-place
/// after each successful pipeline run.
/// `graph` is the CapabilityGraph — mutated by meta-ticks.
/// `proposal_seed` is used to construct the RefactorProposal for each tick;
/// the runner increments the id each tick.
pub async fn run_agent(
    ir: &mut CanonicalIr,
    layout: &mut LayoutGraph,
    graph: &mut CapabilityGraph,
    proposal_seed: RefactorProposal,
    config: &RunnerConfig,
    bridge: &WsBridge,
) -> Result<Vec<TickStats>, RunnerError> {
    let mut ledger = RewardLedger::new(config.ledger_alpha, config.base_trust_threshold);
    let mut stats: Vec<TickStats> = Vec::new();
    let mut tick_number: u64 = 0;

    // Wait for extension then open a tab before the first tick.
    eprintln!("[runner] waiting for extension to connect...");
    bridge.wait_for_connection().await;
    eprintln!("[runner] extension connected — opening ChatGPT tab");
    bridge.open_tab().await.map_err(|_| RunnerError::Io("failed to open tab".into()))?;
    eprintln!("[runner] waiting for tab to be ready...");
    bridge.wait_for_tab().await;
    // Give ChatGPT's React app time to fully mount after tab ready.
    tokio::time::sleep(std::time::Duration::from_secs(4)).await;
    eprintln!("[runner] tab ready — starting tick loop");

    loop {
        tick_number += 1;
        if config.max_ticks > 0 && tick_number > config.max_ticks {
            break;
        }

        eprintln!("[runner] tick {tick_number} — dispatching {} nodes", graph.nodes.len());

        let mut tick_stats = TickStats {
            tick_number,
            nodes_called: 0,
            llm_errors: 0,
            pipeline_reward: None,
            pipeline_error: None,
            meta_tick_fired: false,
            policy_updated: false,
        };

        // --- Step 1: dispatch all nodes in topological order ---
        let mut dispatcher = AgentCallDispatcher::new(graph, ir)
            .with_trust_threshold(config.base_trust_threshold);

        let order = dispatcher
            .dispatch_order()
            .map_err(RunnerError::Dispatch)?;

        let mut stage_outputs: Vec<AgentCallOutput> = Vec::new();

        for node_id in &order {
            // Rebuild dispatcher each iteration so record_output is visible.
            let input = match dispatcher.dispatch(node_id) {
                Ok(inp) => inp,
                Err(e) => {
                    eprintln!("[runner] dispatch skip {node_id}: {e}");
                    continue;
                }
            };

            eprintln!("[runner]   → calling node {node_id} (stage={:?})", input.stage);

            match call_llm(bridge, &input).await {
                Ok(output) => {
                    eprintln!("[runner]   ✓ node {node_id} responded");
                    tick_stats.nodes_called += 1;
                    stage_outputs.push(output.clone());
                    dispatcher.record_output(output);
                }
                Err(e) => {
                    eprintln!("[runner]   ✗ node {node_id} llm error: {e}");
                    tick_stats.llm_errors += 1;
                    // Continue — partial outputs may still be enough for pipeline.
                }
            }
        }

        if stage_outputs.is_empty() {
            eprintln!("[runner] tick {tick_number} — no outputs, skipping pipeline");
            stats.push(tick_stats);
            continue;
        }

        // --- Step 2: construct proposal for this tick ---
        let mut proposal = proposal_seed.clone();
        proposal.id = format!("{}-tick-{}", proposal_seed.id, tick_number);

        // --- Step 3: run pipeline ---
        let pipeline_result = run_pipeline(ir, layout, proposal, &stage_outputs);

        // --- Step 4: record outcome in ledger ---
        // Use the first node id as the primary ledger key for this tick.
        let primary_node = order.first().map(|s| s.as_str()).unwrap_or("unknown");
        let _threshold =
            record_pipeline_outcome(&mut ledger, primary_node, pipeline_result.as_ref());

        match pipeline_result {
            Ok(result) => {
                eprintln!(
                    "[runner] tick {tick_number} — pipeline OK reward={:.4} admission={}",
                    result.reward, result.admission_id
                );
                tick_stats.pipeline_reward = Some(result.reward);
                // Update live IR and layout.
                *ir = result.ir;
                *layout = result.layout;
                // Persist IR.
                persist_ir(ir, &config.ir_out)?;
            }
            Err(ref e) => {
                eprintln!("[runner] tick {tick_number} — pipeline error: {e}");
                tick_stats.pipeline_error = Some(e.to_string());
            }
        }

        // --- Step 5: meta-tick ---
        if config.meta_tick_interval > 0 && tick_number % config.meta_tick_interval == 0 {
            tick_stats.meta_tick_fired = true;
            match run_meta_tick(graph, &ledger) {
                Ok(result) => {
                    eprintln!(
                        "[runner] meta-tick fired — applied={} rejected={} H={:.4}→{:.4}",
                        result.applied.len(),
                        result.rejected.len(),
                        result.entropy_before,
                        result.entropy_after,
                    );
                    *graph = result.graph;
                    save_capability_graph(graph, &config.graph_out)
                        .map_err(|e| RunnerError::Io(e.to_string()))?;
                }
                Err(crate::agent::meta::MetaTickError::NothingToDo) => {
                    eprintln!("[runner] meta-tick — nothing to do");
                }
                Err(e) => {
                    eprintln!("[runner] meta-tick error: {e}");
                }
            }
        }

        // --- Step 6: policy update ---
        if config.policy_update_interval > 0
            && tick_number % config.policy_update_interval == 0
        {
            if let Some(current_policy) = ir.policy_parameters.last() {
                match ledger.update_policy(current_policy) {
                    Ok(new_policy) => {
                        eprintln!(
                            "[runner] policy updated — lr={:.4} baseline={:.4}",
                            new_policy.learning_rate, new_policy.reward_baseline
                        );
                        ir.policy_parameters.push(new_policy);
                        tick_stats.policy_updated = true;
                    }
                    Err(e) => {
                        eprintln!("[runner] policy update error: {e:?}");
                    }
                }
            }
        }

        // Persist ledger every tick.
        persist_ledger(&ledger, &config.ledger_out)?;

        stats.push(tick_stats);
    }

    Ok(stats)
}

// ---------------------------------------------------------------------------
// Persistence helpers
// ---------------------------------------------------------------------------

fn persist_ir(ir: &CanonicalIr, path: &Path) -> Result<(), RunnerError> {
    let json = serde_json::to_string_pretty(ir)
        .map_err(|e| RunnerError::Io(e.to_string()))?;
    fs::write(path, json).map_err(|e| RunnerError::Io(e.to_string()))
}

fn persist_ledger(ledger: &RewardLedger, path: &Path) -> Result<(), RunnerError> {
    let json = serde_json::to_string_pretty(ledger)
        .map_err(|e| RunnerError::Io(e.to_string()))?;
    fs::write(path, json).map_err(|e| RunnerError::Io(e.to_string()))
}
