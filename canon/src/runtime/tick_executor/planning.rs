//! Planning, finalization, and judgment registration.

use crate::ir::world_model::PredictionHead;
use crate::ir::{DeltaId, ExecutionRecord, Judgment, JudgmentDecision, JudgmentPredicate, Plan, Proposal, ProposalGoal, ProposalKind, ProposalStatus, RewardRecord, Word};
use crate::runtime::rollout::{RolloutEngine, RolloutResult};
use crate::runtime::value::Value;
use crate::CanonicalIr;
use std::collections::BTreeMap;

use super::types::{default_predicted_snapshot, PlanContext, PredictionContext, TickExecutionResult};

pub(super) fn plan_tick(ir: &mut CanonicalIr, tick_id: &str, skip_planning: bool) -> PlanContext {
    if skip_planning {
        return PlanContext::default();
    }

    let mut planned_utility = 0.0;
    let mut planning_depth = 0;
    let mut predicted_deltas = Vec::new();
    let mut prediction_context = PredictionContext::default();

    if let Some((rollout, utility)) = search_best_depth(ir, tick_id, 3, BTreeMap::<String, Value>::new()) {
        planned_utility = utility;
        planning_depth = rollout.depth_executed;
        predicted_deltas = rollout.predicted_deltas.clone();
        let horizon = rollout.depth_executed;
        let snapshot = rollout.predicted_state.unwrap_or_else(|| default_predicted_snapshot(tick_id, horizon));
        prediction_context = PredictionContext { predicted_deltas: predicted_deltas.clone(), predicted_snapshot: Some(snapshot.clone()) };
        ir.world_model.push_prediction_head(PredictionHead {
            tick: tick_id.to_string(),
            horizon,
            estimated_reward: rollout.total_reward,
            predicted_deltas: rollout.predicted_deltas,
            predicted_state: snapshot,
        });
    }

    PlanContext { planned_utility, planning_depth, predicted_deltas, prediction_context }
}

/// Evaluate multiple depths and return best (rollout, utility)
fn search_best_depth(ir: &CanonicalIr, tick_id: &str, max_depth: u32, inputs: BTreeMap<String, Value>) -> Option<(RolloutResult, f64)> {
    let engine = RolloutEngine::new(ir);
    let mut best = None;

    for depth in 1..=max_depth {
        if let Ok(result) = engine.rollout(tick_id, depth, inputs.clone()) {
            let utility = result.total_reward + world_model_bonus(ir, tick_id, result.total_reward);
            if best.as_ref().map_or(true, |(_, best_util)| utility > *best_util) {
                best = Some((result, utility));
            }
        }
    }

    best
}

fn world_model_bonus(ir: &CanonicalIr, tick_id: &str, reward: f64) -> f64 {
    ir.world_model.prediction_head.iter().rev().find(|head| head.tick == tick_id).map(|head| (head.estimated_reward - reward) * 0.1).unwrap_or(0.0)
}
pub(super) fn finalize_execution(ir: &mut CanonicalIr, tick_id: &str, result: &TickExecutionResult, planned_utility: f64, planning_depth: u32, predicted_deltas: Vec<DeltaId>) {
    let exec_id = register_plan_and_execution(ir, tick_id, result, planned_utility, planning_depth, predicted_deltas);
    let actual = result.reward;
    let delta = actual - planned_utility;
    let previous_reward = ir.reward_deltas.last().map(|record| record.reward);
    let reward_delta = previous_reward.map(|prev| result.reward - prev);
    ir.reward_deltas.push(RewardRecord { id: format!("reward-{}", result.tick_id), tick: result.tick_id.clone(), execution: exec_id, delta: reward_delta, reward: result.reward });
    println!("[planner] tick={} depth={} planned={} actual={} delta={}", tick_id, planning_depth, planned_utility, actual, delta);
}

fn register_plan_and_execution(ir: &mut CanonicalIr, tick_id: &str, result: &TickExecutionResult, planned_utility: f64, planning_depth: u32, predicted_deltas: Vec<DeltaId>) -> String {
    let plan_id = format!("plan.{tick_id}");
    let plan_key = plan_id.clone();
    let exec_id = format!("execution.{tick_id}");
    let exec_key = exec_id.clone();
    let judgment_id = ensure_planning_judgment(ir);
    ir.plans.retain(|plan| plan.id != plan_key);
    ir.plans.push(Plan {
        id: plan_id.clone(),
        judgment: judgment_id,
        steps: result.execution_order.clone(),
        expected_deltas: predicted_deltas,
        search_depth: planning_depth,
        utility_estimate: planned_utility,
    });
    let actual_deltas = result.emitted_deltas.iter().map(|delta| delta.delta_id.clone()).collect::<Vec<_>>();
    ir.executions.retain(|exec| exec.id != exec_key);
    ir.executions.push(ExecutionRecord {
        id: exec_id.clone(),
        tick: tick_id.to_owned(),
        plan: plan_id,
        outcome_deltas: actual_deltas,
        errors: Vec::new(),
        events: Vec::new(),
        reward: Some(result.reward),
        planned_utility: Some(planned_utility),
        planning_depth: Some(planning_depth),
    });
    exec_id
}

fn ensure_planning_judgment(ir: &mut CanonicalIr) -> String {
    if let Some(judgment) = ir.judgments.iter().find(|item| item.decision == JudgmentDecision::Accept) {
        return judgment.id.clone();
    }
    let proposal_id = "proposal.planner".to_owned();
    if ir.proposals.iter().all(|proposal| proposal.id != proposal_id) {
        ir.proposals.push(Proposal {
            id: proposal_id.clone(),
            kind: ProposalKind::Structural,
            goal: ProposalGoal { id: Word::new("planner").expect("word must be canonical"), description: "Auto-generated planner goal".to_owned() },
            nodes: Vec::new(),
            apis: Vec::new(),
            edges: Vec::new(),
            status: ProposalStatus::Accepted,
        });
    }
    let predicate_id = "predicate.planner".to_owned();
    if ir.judgment_predicates.iter().all(|predicate| predicate.id != predicate_id) {
        ir.judgment_predicates.push(JudgmentPredicate { id: predicate_id.clone(), description: "Auto-generated planner predicate".to_owned() });
    }
    let judgment_id = "judgment.planner".to_owned();
    ir.judgments.push(Judgment { id: judgment_id.clone(), proposal: proposal_id, predicate: predicate_id, decision: JudgmentDecision::Accept, rationale: "planner fixture".to_owned() });
    judgment_id
}
