use super::error::Violation;
use super::helpers::Indexes;
use super::rules::CanonRule;
use crate::ir::*;

pub fn check<'a>(ir: &'a CanonicalIr, idx: &Indexes<'a>, violations: &mut Vec<Violation>) {
    check_epochs(ir, idx, violations);
    check_ticks(ir, idx, violations);
    check_plans(ir, idx, violations);
    check_executions(ir, idx, violations);
    check_gpu(ir, idx, violations);
}

fn check_epochs(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for epoch in &ir.tick_epochs {
        for tick in &epoch.ticks {
            if idx.ticks.get(tick.as_str()).is_none() {
                violations.push(Violation::new(
                    CanonRule::TickEpochs,
                    format!("epoch `{}` references unknown tick `{tick}`", epoch.id),
                ));
            }
        }
        if let Some(parent) = &epoch.parent_epoch {
            if parent == &epoch.id {
                violations.push(Violation::new(
                    CanonRule::TickEpochs,
                    format!("epoch `{}` may not reference itself as parent", epoch.id),
                ));
            } else if idx.epochs.get(parent.as_str()).is_none() {
                violations.push(Violation::new(
                    CanonRule::TickEpochs,
                    format!("epoch `{}` references missing parent `{parent}`", epoch.id),
                ));
            } else {
                let mut cursor = parent.as_str();
                let mut seen = std::collections::HashSet::new();
                seen.insert(epoch.id.as_str());
                while let Some(pe) = idx.epochs.get(cursor) {
                    if !seen.insert(pe.id.as_str()) {
                        violations.push(Violation::new(
                            CanonRule::TickEpochs,
                            format!("epoch hierarchy containing `{}` forms a cycle", epoch.id),
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
            violations.push(Violation::new(
                CanonRule::TickRoot,
                format!(
                    "tick `{}` references missing graph `{}`",
                    tick.id, tick.graph
                ),
            ));
        }
        for d in &tick.input_state {
            if idx.deltas.get(d.as_str()).is_none() {
                violations.push(Violation::new(
                    CanonRule::TickRoot,
                    format!("tick `{}` input delta `{d}` does not exist", tick.id),
                ));
            }
        }
        for d in &tick.output_deltas {
            if idx.deltas.get(d.as_str()).is_none() {
                violations.push(Violation::new(
                    CanonRule::TickRoot,
                    format!("tick `{}` output delta `{d}` does not exist", tick.id),
                ));
            }
        }
    }
}

fn check_plans(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for plan in &ir.plans {
        let Some(j) = idx.judgments.get(plan.judgment.as_str()) else {
            violations.push(Violation::new(
                CanonRule::PlanArtifacts,
                format!(
                    "plan `{}` references missing judgment `{}`",
                    plan.id, plan.judgment
                ),
            ));
            continue;
        };
        if j.decision != JudgmentDecision::Accept {
            violations.push(Violation::new(
                CanonRule::PlanArtifacts,
                format!("plan `{}` must point to an accepted judgment", plan.id),
            ));
        }
        for step in &plan.steps {
            if idx.functions.get(step.as_str()).is_none() {
                violations.push(Violation::new(
                    CanonRule::PlanArtifacts,
                    format!("plan `{}` references unknown function `{step}`", plan.id),
                ));
            }
        }
        for d in &plan.expected_deltas {
            if idx.deltas.get(d.as_str()).is_none() {
                violations.push(Violation::new(
                    CanonRule::PlanArtifacts,
                    format!("plan `{}` expects unknown delta `{d}`", plan.id),
                ));
            }
        }
    }
}

fn check_executions(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for exec in &ir.executions {
        if idx.ticks.get(exec.tick.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::ExecutionBoundary,
                format!(
                    "execution `{}` references missing tick `{}`",
                    exec.id, exec.tick
                ),
            ));
        }
        if idx.plans.get(exec.plan.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::ExecutionBoundary,
                format!(
                    "execution `{}` references missing plan `{}`",
                    exec.id, exec.plan
                ),
            ));
        }
        for d in &exec.outcome_deltas {
            if idx.deltas.get(d.as_str()).is_none() {
                violations.push(Violation::new(
                    CanonRule::ExecutionBoundary,
                    format!("execution `{}` captured unknown delta `{d}`", exec.id),
                ));
            }
        }
    }
}

fn check_gpu(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for gpu in &ir.gpu_functions {
        if idx.functions.get(gpu.function.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::GpuLawfulMath,
                format!(
                    "gpu kernel `{}` references missing function `{}`",
                    gpu.id, gpu.function
                ),
            ));
        }
        if gpu.inputs.is_empty() || gpu.outputs.is_empty() {
            violations.push(Violation::new(
                CanonRule::GpuLawfulMath,
                format!("gpu kernel `{}` must enumerate inputs and outputs", gpu.id),
            ));
        }
        for port in gpu.inputs.iter().chain(gpu.outputs.iter()) {
            if port.lanes == 0 {
                violations.push(Violation::new(
                    CanonRule::GpuLawfulMath,
                    format!(
                        "gpu kernel `{}` port `{}` must specify lanes > 0",
                        gpu.id, port.name
                    ),
                ));
            }
        }
        if !gpu.properties.pure
            || !gpu.properties.no_alloc
            || !gpu.properties.no_branch
            || !gpu.properties.no_io
        {
            violations.push(Violation::new(
                CanonRule::GpuLawfulMath,
                format!("gpu kernel `{}` violates the math-only contract", gpu.id),
            ));
        }
    }
}
