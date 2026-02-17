use super::error::{Violation, ViolationDetail};
use super::helpers::Indexes;
use super::rules::CanonRule;
use crate::ir::*;

pub fn check<'a>(ir: &'a CanonicalIr, idx: &Indexes<'a>, violations: &mut Vec<Violation>) {
    check_epochs(ir, idx, violations);
    check_ticks(ir, idx, violations);
    check_plans(ir, idx, violations);
    check_executions(ir, idx, violations);
    check_gpu(ir, idx, violations);
    check_reward_monotonicity(ir, violations);
    check_reward_collapse(ir, violations);
}

fn check_epochs(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for epoch in &ir.tick_epochs {
        for tick in &epoch.ticks {
            if idx.ticks.get(tick.as_str()).is_none() {
                violations.push(Violation::structured(
                    CanonRule::TickEpochs,
                    epoch.id.clone(),
                    ViolationDetail::EpochMissingTick {
                        epoch: epoch.id.clone(),
                        tick: tick.clone(),
                    },
                ));
            }
        }
        if let Some(parent) = &epoch.parent_epoch {
            if parent == &epoch.id {
                violations.push(Violation::structured(
                    CanonRule::TickEpochs,
                    epoch.id.clone(),
                    ViolationDetail::EpochSelfParent {
                        epoch: epoch.id.clone(),
                    },
                ));
            } else if idx.epochs.get(parent.as_str()).is_none() {
                violations.push(Violation::structured(
                    CanonRule::TickEpochs,
                    epoch.id.clone(),
                    ViolationDetail::EpochMissingParent {
                        epoch: epoch.id.clone(),
                        parent: parent.clone(),
                    },
                ));
            } else {
                let mut cursor = parent.as_str();
                let mut seen = std::collections::HashSet::new();
                seen.insert(epoch.id.as_str());
                while let Some(pe) = idx.epochs.get(cursor) {
                    if !seen.insert(pe.id.as_str()) {
                        violations.push(Violation::structured(
                            CanonRule::TickEpochs,
                            epoch.id.clone(),
                            ViolationDetail::EpochCycle {
                                epoch: epoch.id.clone(),
                            },
                        ));
                        break;
                    }
                    if let Some(next) = &pe.parent_epoch {
                        cursor = next.as_str();
                    } else {
                        break;
                    }
                }
            }
        }
    }
}

fn check_ticks(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for tick in &ir.ticks {
        if idx.tick_graphs.get(tick.graph.as_str()).is_none() {
            violations.push(Violation::structured(
                CanonRule::TickRoot,
                tick.id.clone(),
                ViolationDetail::TickMissingGraph {
                    tick: tick.id.clone(),
                    graph: tick.graph.clone(),
                },
            ));
        }
        for d in &tick.input_state {
            if idx.deltas.get(d.as_str()).is_none() {
                violations.push(Violation::structured(
                    CanonRule::TickRoot,
                    tick.id.clone(),
                    ViolationDetail::TickMissingDelta {
                        tick: tick.id.clone(),
                        delta: d.clone(),
                    },
                ));
            }
        }
        for d in &tick.output_deltas {
            if idx.deltas.get(d.as_str()).is_none() {
                violations.push(Violation::structured(
                    CanonRule::TickRoot,
                    tick.id.clone(),
                    ViolationDetail::TickMissingDelta {
                        tick: tick.id.clone(),
                        delta: d.clone(),
                    },
                ));
            }
        }
    }
}

fn check_plans(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for plan in &ir.plans {
        let Some(j) = idx.judgments.get(plan.judgment.as_str()) else {
            violations.push(Violation::structured(
                CanonRule::PlanArtifacts,
                plan.id.clone(),
                ViolationDetail::PlanMissingJudgment {
                    plan: plan.id.clone(),
                    judgment: plan.judgment.clone(),
                },
            ));
            continue;
        };
        if j.decision != JudgmentDecision::Accept {
            violations.push(Violation::structured(
                CanonRule::PlanArtifacts,
                plan.id.clone(),
                ViolationDetail::PlanNotAccepted {
                    plan: plan.id.clone(),
                },
            ));
        }
        for step in &plan.steps {
            if idx.functions.get(step.as_str()).is_none() {
                violations.push(Violation::structured(
                    CanonRule::PlanArtifacts,
                    plan.id.clone(),
                    ViolationDetail::PlanMissingFunction {
                        plan: plan.id.clone(),
                        function: step.clone(),
                    },
                ));
            }
        }
        for d in &plan.expected_deltas {
            if idx.deltas.get(d.as_str()).is_none() {
                violations.push(Violation::structured(
                    CanonRule::PlanArtifacts,
                    plan.id.clone(),
                    ViolationDetail::PlanMissingDelta {
                        plan: plan.id.clone(),
                        delta: d.clone(),
                    },
                ));
            }
        }
    }
}

fn check_executions(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for exec in &ir.executions {
        if idx.ticks.get(exec.tick.as_str()).is_none() {
            violations.push(Violation::structured(
                CanonRule::ExecutionBoundary,
                exec.id.clone(),
                ViolationDetail::ExecutionMissingTick {
                    execution: exec.id.clone(),
                    tick: exec.tick.clone(),
                },
            ));
        }
        if idx.plans.get(exec.plan.as_str()).is_none() {
            violations.push(Violation::structured(
                CanonRule::ExecutionBoundary,
                exec.id.clone(),
                ViolationDetail::ExecutionMissingPlan {
                    execution: exec.id.clone(),
                    plan: exec.plan.clone(),
                },
            ));
        }
        for d in &exec.outcome_deltas {
            if idx.deltas.get(d.as_str()).is_none() {
                violations.push(Violation::structured(
                    CanonRule::ExecutionBoundary,
                    exec.id.clone(),
                    ViolationDetail::ExecutionMissingDelta {
                        execution: exec.id.clone(),
                        delta: d.clone(),
                    },
                ));
            }
        }
    }
}

fn check_reward_monotonicity(ir: &CanonicalIr, violations: &mut Vec<Violation>) {
    // Enforce r_t >= r_{t-1} - epsilon across the append-only reward log.
    // Slack epsilon is fixed at 0.0 for Layer F; will be parameterised in Layer L.
    const EPSILON: f64 = 0.0;
    let records = &ir.reward_deltas;
    for i in 1..records.len() {
        let prev = records[i - 1].reward;
        let curr = records[i].reward;
        if curr < prev - EPSILON {
            violations.push(Violation::structured(
                CanonRule::RewardCollapseDetected,
                records[i].id.clone(),
                ViolationDetail::RewardDrop {
                    subject: records[i].id.clone(),
                },
            ));
        }
    }
}

fn check_gpu(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for gpu in &ir.gpu_functions {
        if idx.functions.get(gpu.function.as_str()).is_none() {
            violations.push(Violation::structured(
                CanonRule::GpuLawfulMath,
                gpu.id.clone(),
                ViolationDetail::GpuMissingFunction {
                    gpu: gpu.id.clone(),
                    function: gpu.function.clone(),
                },
            ));
        }
        if gpu.inputs.is_empty() || gpu.outputs.is_empty() {
            violations.push(Violation::structured(
                CanonRule::GpuLawfulMath,
                gpu.id.clone(),
                ViolationDetail::GpuMissingPorts {
                    gpu: gpu.id.clone(),
                },
            ));
        }
        for port in gpu.inputs.iter().chain(gpu.outputs.iter()) {
            if port.lanes == 0 {
                violations.push(Violation::structured(
                    CanonRule::GpuLawfulMath,
                    gpu.id.clone(),
                    ViolationDetail::GpuInvalidLanes {
                        gpu: gpu.id.clone(),
                        port: port.name.to_string(),
                    },
                ));
            }
        }
        if !gpu.properties.pure
            || !gpu.properties.no_alloc
            || !gpu.properties.no_branch
            || !gpu.properties.no_io
        {
            violations.push(Violation::structured(
                CanonRule::GpuLawfulMath,
                gpu.id.clone(),
                ViolationDetail::GpuContractViolation {
                    gpu: gpu.id.clone(),
                },
            ));
        }
    }
}

fn check_reward_collapse(ir: &CanonicalIr, violations: &mut Vec<Violation>) {
    const EPSILON: f64 = 0.05;
    let mut prev = None;
    for epoch in &ir.tick_epochs {
        if let Some(snapshot) = &epoch.policy_snapshot {
            if let Some(last_reward) = prev {
                let delta = snapshot.reward_at_snapshot - last_reward;
                if delta < -EPSILON {
                    violations.push(Violation::structured(
                        CanonRule::RewardMonotonicity,
                        epoch.id.clone(),
                        ViolationDetail::RewardDrop {
                            subject: epoch.id.clone(),
                        },
                    ));
                }
            }
            prev = Some(snapshot.reward_at_snapshot);
        }
    }
}
