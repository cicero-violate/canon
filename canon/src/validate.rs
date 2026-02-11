use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::ir::*;
use crate::proposal::resolve_proposal_nodes;
use petgraph::algo::is_cyclic_directed;
use petgraph::graphmap::DiGraphMap;

pub fn validate_ir(ir: &CanonicalIr) -> Result<(), ValidationErrors> {
    let mut violations = Vec::new();

    if ir.project.language != Language::Rust {
        violations.push(Violation::new(
            CanonRule::ProjectEnvelope,
            "project language must currently be Rust",
        ));
    }
    if ir.project.version.trim().is_empty() {
        violations.push(Violation::new(
            CanonRule::ProjectEnvelope,
            "project version must be declared",
        ));
    }
    if ir.version_contract.current != ir.meta.version {
        violations.push(Violation::new(
            CanonRule::VersionEvolution,
            format!(
                "version contract `{}` must match meta version `{}`",
                ir.version_contract.current, ir.meta.version
            ),
        ));
    }
    let mut compatible_versions = HashSet::new();
    for version in &ir.version_contract.compatible_with {
        if !compatible_versions.insert(version) {
            violations.push(Violation::new(
                CanonRule::VersionEvolution,
                format!("version `{version}` listed multiple times in compatibility"),
            ));
        }
    }
    if ir.version_contract.migration_proofs.is_empty() {
        violations.push(Violation::new(
            CanonRule::VersionEvolution,
            "version contract must provide at least one migration proof",
        ));
    }

    let modules = index_by_id(
        &ir.modules,
        |m| m.id.as_str(),
        CanonRule::ExplicitArtifacts,
        "module",
        &mut violations,
    );
    let structs = index_by_id(
        &ir.structs,
        |s| s.id.as_str(),
        CanonRule::ExplicitArtifacts,
        "struct",
        &mut violations,
    );
    let traits = index_by_id(
        &ir.traits,
        |t| t.id.as_str(),
        CanonRule::ExplicitArtifacts,
        "trait",
        &mut violations,
    );
    let predicates = index_by_id(
        &ir.judgment_predicates,
        |p| p.id.as_str(),
        CanonRule::JudgmentDecisions,
        "judgment predicate",
        &mut violations,
    );
    let impls = index_by_id(
        &ir.impl_blocks,
        |i| i.id.as_str(),
        CanonRule::ImplBinding,
        "impl",
        &mut violations,
    );
    let functions = index_by_id(
        &ir.functions,
        |f| f.id.as_str(),
        CanonRule::ExecutionOnlyInImpl,
        "function",
        &mut violations,
    );
    let tick_graphs = index_by_id(
        &ir.tick_graphs,
        |g| g.id.as_str(),
        CanonRule::TickGraphAcyclic,
        "tick graph",
        &mut violations,
    );
    let ticks = index_by_id(
        &ir.ticks,
        |t| t.id.as_str(),
        CanonRule::TickRoot,
        "tick",
        &mut violations,
    );
    let proposals = index_by_id(
        &ir.proposals,
        |p| p.id.as_str(),
        CanonRule::ProposalDeclarative,
        "proposal",
        &mut violations,
    );
    let judgments_map = index_by_id(
        &ir.judgments,
        |j| j.id.as_str(),
        CanonRule::JudgmentDecisions,
        "judgment",
        &mut violations,
    );
    let _learning_map = index_by_id(
        &ir.learning,
        |l| l.id.as_str(),
        CanonRule::LearningDeclarations,
        "learning",
        &mut violations,
    );
    let deltas = index_by_id(
        &ir.deltas,
        |d| d.id.as_str(),
        CanonRule::EffectsAreDeltas,
        "delta",
        &mut violations,
    );
    let proofs = index_by_id(
        &ir.proofs,
        |p| p.id.as_str(),
        CanonRule::DeltaProofs,
        "proof",
        &mut violations,
    );
    let epochs = index_by_id(
        &ir.tick_epochs,
        |e| e.id.as_str(),
        CanonRule::TickEpochs,
        "tick epoch",
        &mut violations,
    );
    let plans = index_by_id(
        &ir.plans,
        |p| p.id.as_str(),
        CanonRule::PlanArtifacts,
        "plan",
        &mut violations,
    );
    let _execution_records = index_by_id(
        &ir.executions,
        |e| e.id.as_str(),
        CanonRule::ExecutionBoundary,
        "execution record",
        &mut violations,
    );
    let admissions = index_by_id(
        &ir.admissions,
        |a| a.id.as_str(),
        CanonRule::AdmissionBridge,
        "admission",
        &mut violations,
    );
    let _applied_records = index_by_id(
        &ir.applied_deltas,
        |a| a.id.as_str(),
        CanonRule::AdmissionBridge,
        "applied delta",
        &mut violations,
    );
    for proof_id in &ir.version_contract.migration_proofs {
        match proofs.get(proof_id.as_str()) {
            Some(proof) if proof.scope == ProofScope::Law => {}
            Some(_) => violations.push(Violation::new(
                CanonRule::VersionEvolution,
                format!("version migration proof `{}` must have law scope", proof_id),
            )),
            None => violations.push(Violation::new(
                CanonRule::VersionEvolution,
                format!("version migration proof `{}` was not found", proof_id),
            )),
        }
    }
    let _gpu_function_map = index_by_id(
        &ir.gpu_functions,
        |g| g.id.as_str(),
        CanonRule::GpuLawfulMath,
        "gpu function",
        &mut violations,
    );

    let mut dependency_names = HashSet::new();
    for dependency in &ir.dependencies {
        if dependency.version.trim().is_empty() {
            violations.push(Violation::new(
                CanonRule::ExternalDependencies,
                format!("dependency `{}` must provide a version", dependency.name),
            ));
        }
        if !dependency_names.insert(dependency.name.as_str()) {
            violations.push(Violation::new(
                CanonRule::ExternalDependencies,
                format!("dependency `{}` declared multiple times", dependency.name),
            ));
        }
    }

    for epoch in &ir.tick_epochs {
        for tick in &epoch.ticks {
            if !ticks.contains_key(tick.as_str()) {
                violations.push(Violation::new(
                    CanonRule::TickEpochs,
                    format!("epoch `{}` references unknown tick `{}`", epoch.id, tick),
                ));
            }
        }
        if let Some(parent) = &epoch.parent_epoch {
            if parent == &epoch.id {
                violations.push(Violation::new(
                    CanonRule::TickEpochs,
                    format!("epoch `{}` may not reference itself as parent", epoch.id),
                ));
            } else if !epochs.contains_key(parent.as_str()) {
                violations.push(Violation::new(
                    CanonRule::TickEpochs,
                    format!(
                        "epoch `{}` references missing parent `{}`",
                        epoch.id, parent
                    ),
                ));
            } else {
                let mut cursor = parent.as_str();
                let mut seen = HashSet::new();
                seen.insert(epoch.id.as_str());
                while let Some(parent_epoch) = epochs.get(cursor) {
                    if !seen.insert(parent_epoch.id.as_str()) {
                        violations.push(Violation::new(
                            CanonRule::TickEpochs,
                            format!("epoch hierarchy containing `{}` forms a cycle", epoch.id),
                        ));
                        break;
                    }
                    if let Some(next) = &parent_epoch.parent_epoch {
                        cursor = next.as_str();
                    } else {
                        break;
                    }
                }
            }
        }
    }

    for plan in &ir.plans {
        let Some(judgment) = judgments_map.get(plan.judgment.as_str()) else {
            violations.push(Violation::new(
                CanonRule::PlanArtifacts,
                format!(
                    "plan `{}` references missing judgment `{}`",
                    plan.id, plan.judgment
                ),
            ));
            continue;
        };
        if judgment.decision != JudgmentDecision::Accept {
            violations.push(Violation::new(
                CanonRule::PlanArtifacts,
                format!(
                    "plan `{}` must point to an accepted judgment `{}`",
                    plan.id, plan.judgment
                ),
            ));
        }
        for step in &plan.steps {
            if !functions.contains_key(step.as_str()) {
                violations.push(Violation::new(
                    CanonRule::PlanArtifacts,
                    format!("plan `{}` references unknown function `{}`", plan.id, step),
                ));
            }
        }
        for delta_id in &plan.expected_deltas {
            if !deltas.contains_key(delta_id.as_str()) {
                violations.push(Violation::new(
                    CanonRule::PlanArtifacts,
                    format!("plan `{}` expects unknown delta `{}`", plan.id, delta_id),
                ));
            }
        }
    }

    for execution in &ir.executions {
        if !ticks.contains_key(execution.tick.as_str()) {
            violations.push(Violation::new(
                CanonRule::ExecutionBoundary,
                format!(
                    "execution `{}` references missing tick `{}`",
                    execution.id, execution.tick
                ),
            ));
        }
        if !plans.contains_key(execution.plan.as_str()) {
            violations.push(Violation::new(
                CanonRule::ExecutionBoundary,
                format!(
                    "execution `{}` references missing plan `{}`",
                    execution.id, execution.plan
                ),
            ));
        }
        for delta_id in &execution.outcome_deltas {
            if !deltas.contains_key(delta_id.as_str()) {
                violations.push(Violation::new(
                    CanonRule::ExecutionBoundary,
                    format!(
                        "execution `{}` captured unknown delta `{}`",
                        execution.id, delta_id
                    ),
                ));
            }
        }
    }

    for admission in &ir.admissions {
        if !ticks.contains_key(admission.tick.as_str()) {
            violations.push(Violation::new(
                CanonRule::AdmissionBridge,
                format!(
                    "admission `{}` references missing tick `{}`",
                    admission.id, admission.tick
                ),
            ));
        }
        let Some(judgment) = judgments_map.get(admission.judgment.as_str()) else {
            violations.push(Violation::new(
                CanonRule::AdmissionBridge,
                format!(
                    "admission `{}` references missing judgment `{}`",
                    admission.id, admission.judgment
                ),
            ));
            continue;
        };
        if judgment.decision != JudgmentDecision::Accept {
            violations.push(Violation::new(
                CanonRule::AdmissionBridge,
                format!(
                    "admission `{}` must reference accepted judgment `{}`",
                    admission.id, admission.judgment
                ),
            ));
        }
        for delta_id in &admission.delta_ids {
            if !deltas.contains_key(delta_id.as_str()) {
                violations.push(Violation::new(
                    CanonRule::AdmissionBridge,
                    format!(
                        "admission `{}` lists unknown delta `{}`",
                        admission.id, delta_id
                    ),
                ));
            }
        }
    }

    let mut previous_order = None;
    for record in &ir.applied_deltas {
        if !admissions.contains_key(record.admission.as_str()) {
            violations.push(Violation::new(
                CanonRule::AdmissionBridge,
                format!(
                    "applied delta `{}` references missing admission `{}`",
                    record.id, record.admission
                ),
            ));
        }
        if !deltas.contains_key(record.delta.as_str()) {
            violations.push(Violation::new(
                CanonRule::AdmissionBridge,
                format!(
                    "applied delta `{}` references unknown delta `{}`",
                    record.id, record.delta
                ),
            ));
        }
        if let Some(prev) = previous_order {
            if record.order < prev {
                violations.push(Violation::new(
                    CanonRule::AdmissionBridge,
                    format!(
                        "applied deltas must be non-decreasing order but `{}` < `{}`",
                        record.order, prev
                    ),
                ));
            }
        }
        previous_order = Some(record.order);
    }

    for structure in &ir.structs {
        if modules.get(structure.module.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::ExplicitArtifacts,
                format!(
                    "struct `{}` references unknown module `{}`",
                    structure.name, structure.module
                ),
            ));
        }
    }

    let mut trait_functions: HashMap<&str, (&Trait, &TraitFunction)> = HashMap::new();
    for tr in &ir.traits {
        if modules.get(tr.module.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::ExplicitArtifacts,
                format!(
                    "trait `{}` references unknown module `{}`",
                    tr.name, tr.module
                ),
            ));
        }
        for func in &tr.functions {
            if trait_functions
                .insert(func.id.as_str(), (tr, func))
                .is_some()
            {
                violations.push(Violation::new(
                    CanonRule::TraitVerbs,
                    format!(
                        "trait function `{}` must be unique across all traits",
                        func.id
                    ),
                ));
            }
        }
    }

    let module_adjacency = validate_module_graph(ir, &modules, &mut violations);

    for block in &ir.impl_blocks {
        let Some(structure) = structs.get(block.struct_id.as_str()) else {
            violations.push(Violation::new(
                CanonRule::ImplBinding,
                format!(
                    "impl `{}` references missing struct `{}`",
                    block.id, block.struct_id
                ),
            ));
            continue;
        };

        let Some(trait_def) = traits.get(block.trait_id.as_str()) else {
            violations.push(Violation::new(
                CanonRule::ImplBinding,
                format!(
                    "impl `{}` references missing trait `{}`",
                    block.id, block.trait_id
                ),
            ));
            continue;
        };

        if modules.get(block.module.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::ImplBinding,
                format!(
                    "impl `{}` references unknown module `{}`",
                    block.id, block.module
                ),
            ));
        }

        if block.module != structure.module {
            violations.push(Violation::new(
                CanonRule::ImplBinding,
                format!(
                    "impl `{}` must live in the same module `{}` as struct `{}`",
                    block.id, structure.module, structure.name
                ),
            ));
        }

        if trait_def.module != block.module {
            violations.push(Violation::new(
                CanonRule::ImplBinding,
                format!(
                    "impl `{}` binding to trait `{}` must share module `{}`",
                    block.id, trait_def.name, block.module
                ),
            ));
        }
    }

    let mut binding_lookup: HashMap<&str, (&ImplBlock, &ImplFunctionBinding)> = HashMap::new();
    for block in &ir.impl_blocks {
        for binding in &block.functions {
            if binding_lookup
                .insert(binding.function.as_str(), (block, binding))
                .is_some()
            {
                violations.push(Violation::new(
                    CanonRule::ExecutionOnlyInImpl,
                    format!(
                        "function `{}` is bound more than once in impl `{}`",
                        binding.function, block.id
                    ),
                ));
            }

            if trait_functions
                .get(binding.trait_fn.as_str())
                .map(|(t, _)| t.id.as_str())
                != Some(block.trait_id.as_str())
            {
                violations.push(Violation::new(
                    CanonRule::ImplBinding,
                    format!(
                        "impl `{}` cannot bind trait function `{}` from another trait",
                        block.id, binding.trait_fn
                    ),
                ));
            }
        }
    }

    for function in &ir.functions {
        let Some(block) = impls.get(function.impl_id.as_str()) else {
            violations.push(Violation::new(
                CanonRule::ExecutionOnlyInImpl,
                format!(
                    "function `{}` references missing impl `{}`",
                    function.name, function.impl_id
                ),
            ));
            continue;
        };

        if function.module != block.module {
            violations.push(Violation::new(
                CanonRule::ExecutionOnlyInImpl,
                format!(
                    "function `{}` must live in module `{}` declared by its impl `{}`",
                    function.name, block.module, block.id
                ),
            ));
        }

        if let Some((trait_def, _)) = trait_functions.get(function.trait_function.as_str()) {
            if trait_def.id != block.trait_id {
                violations.push(Violation::new(
                    CanonRule::ImplBinding,
                    format!(
                        "function `{}` implements trait function `{}` but impl `{}` is for trait `{}`",
                        function.name, function.trait_function, block.id, block.trait_id
                    ),
                ));
            }
        } else {
            violations.push(Violation::new(
                CanonRule::ImplBinding,
                format!(
                    "function `{}` references unknown trait function `{}`",
                    function.name, function.trait_function
                ),
            ));
        }

        if !function.contract.total
            || !function.contract.deterministic
            || !function.contract.explicit_inputs
            || !function.contract.explicit_outputs
            || !function.contract.effects_are_deltas
        {
            violations.push(Violation::new(
                CanonRule::FunctionContracts,
                format!(
                    "function `{}` must assert totality, determinism, explicit IO, and delta effects",
                    function.name
                ),
            ));
        }
    }

    let mut permission_cache: HashMap<&str, HashSet<&str>> = HashMap::new();
    let mut call_graph: DiGraphMap<&str, ()> = DiGraphMap::new();
    for id in functions.keys() {
        call_graph.add_node(*id);
    }
    let mut call_pairs: HashSet<(FunctionId, FunctionId)> = HashSet::new();

    for edge in &ir.call_edges {
        let Some(caller) = functions.get(edge.caller.as_str()) else {
            violations.push(Violation::new(
                CanonRule::CallGraphPublicApis,
                format!(
                    "call edge `{}` references missing caller `{}`",
                    edge.id, edge.caller
                ),
            ));
            continue;
        };
        let Some(callee) = functions.get(edge.callee.as_str()) else {
            violations.push(Violation::new(
                CanonRule::CallGraphPublicApis,
                format!(
                    "call edge `{}` references missing callee `{}`",
                    edge.id, edge.callee
                ),
            ));
            continue;
        };

        call_pairs.insert((edge.caller.clone(), edge.callee.clone()));
        call_graph.add_edge(edge.caller.as_str(), edge.callee.as_str(), ());

        if callee.visibility != Visibility::Public {
            violations.push(Violation::new(
                CanonRule::CallGraphPublicApis,
                format!(
                    "call edge `{}` targets private function `{}`",
                    edge.id, callee.name
                ),
            ));
        }

        let caller_module = caller.module.as_str();
        let callee_module = callee.module.as_str();
        if !module_has_permission(
            caller_module,
            callee_module,
            &module_adjacency,
            &mut permission_cache,
        ) {
            violations.push(Violation::new(
                CanonRule::CallGraphRespectsDag,
                format!(
                    "module `{}` lacks import permission to call `{}` in module `{}`",
                    caller_module, callee.name, callee_module
                ),
            ));
        }
    }

    if is_cyclic_directed(&call_graph) {
        violations.push(Violation::new(
            CanonRule::CallGraphAcyclic,
            "call graph contains recursion which violates Canon rules 29 and 75",
        ));
    }

    for graph in &ir.tick_graphs {
        let mut tick_graph: DiGraphMap<&str, ()> = DiGraphMap::new();
        let mut node_set: HashSet<&str> = HashSet::new();
        for node in &graph.nodes {
            if functions.get(node.as_str()).is_none() {
                violations.push(Violation::new(
                    CanonRule::TickGraphEdgesDeclared,
                    format!(
                        "tick graph `{}` references unknown function `{}`",
                        graph.name, node
                    ),
                ));
                continue;
            }
            node_set.insert(node.as_str());
            tick_graph.add_node(node.as_str());
        }

        for edge in &graph.edges {
            if !node_set.contains(edge.from.as_str()) || !node_set.contains(edge.to.as_str()) {
                violations.push(Violation::new(
                    CanonRule::TickGraphEdgesDeclared,
                    format!(
                        "tick graph `{}` edge {} -> {} must reference declared nodes",
                        graph.name, edge.from, edge.to
                    ),
                ));
                continue;
            }
            if !call_pairs.contains(&(edge.from.clone(), edge.to.clone())) {
                violations.push(Violation::new(
                    CanonRule::TickGraphEdgesDeclared,
                    format!(
                        "tick graph `{}` edge {} -> {} must exist in call graph",
                        graph.name, edge.from, edge.to
                    ),
                ));
            }
            tick_graph.add_edge(edge.from.as_str(), edge.to.as_str(), ());
        }

        if is_cyclic_directed(&tick_graph) {
            violations.push(Violation::new(
                CanonRule::TickGraphAcyclic,
                format!("tick graph `{}` must be acyclic", graph.name),
            ));
        }
    }

    for policy in &ir.loop_policies {
        if !tick_graphs.contains_key(policy.graph.as_str()) {
            violations.push(Violation::new(
                CanonRule::LoopContinuationJudgment,
                format!(
                    "loop policy `{}` references missing tick graph `{}`",
                    policy.id, policy.graph
                ),
            ));
        }
        if !predicates.contains_key(policy.continuation.as_str()) {
            violations.push(Violation::new(
                CanonRule::LoopContinuationJudgment,
                format!(
                    "loop policy `{}` must reference a judgment predicate but `{}` was not found",
                    policy.id, policy.continuation
                ),
            ));
        }
    }

    for tick in &ir.ticks {
        if !tick_graphs.contains_key(tick.graph.as_str()) {
            violations.push(Violation::new(
                CanonRule::TickRoot,
                format!(
                    "tick `{}` references missing graph `{}`",
                    tick.id, tick.graph
                ),
            ));
        }
        for delta in &tick.input_state {
            if deltas.get(delta.as_str()).is_none() {
                violations.push(Violation::new(
                    CanonRule::TickRoot,
                    format!("tick `{}` input delta `{}` does not exist", tick.id, delta),
                ));
            }
        }
        for delta in &tick.output_deltas {
            if deltas.get(delta.as_str()).is_none() {
                violations.push(Violation::new(
                    CanonRule::TickRoot,
                    format!("tick `{}` output delta `{}` does not exist", tick.id, delta),
                ));
            }
        }
    }

    for gpu in &ir.gpu_functions {
        if functions.get(gpu.function.as_str()).is_none() {
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
                format!(
                    "gpu kernel `{}` must enumerate its inputs and outputs",
                    gpu.id
                ),
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

    for proposal in &ir.proposals {
        if proposal.nodes.is_empty() || proposal.apis.is_empty() || proposal.edges.is_empty() {
            violations.push(Violation::new(
                CanonRule::ProposalDeclarative,
                format!(
                    "proposal `{}` must enumerate nodes, APIs, and edges",
                    proposal.id
                ),
            ));
        }
        if proposal.goal.description.trim().is_empty() {
            violations.push(Violation::new(
                CanonRule::ProposalDeclarative,
                format!(
                    "proposal `{}` must include a textual goal description",
                    proposal.id
                ),
            ));
        }

        let resolved_nodes = match resolve_proposal_nodes(proposal) {
            Ok(nodes) => nodes,
            Err(err) => {
                violations.push(Violation::new(
                    CanonRule::ProposalDeclarative,
                    format!("proposal `{}` is invalid: {err}", proposal.id),
                ));
                continue;
            }
        };
        let proposed_module_ids: HashSet<&str> = resolved_nodes
            .modules
            .iter()
            .map(|m| m.id.as_str())
            .collect();
        let proposed_trait_ids: HashSet<&str> = resolved_nodes
            .traits
            .iter()
            .map(|t| t.id.as_str())
            .collect();

        for structure in &resolved_nodes.structs {
            if modules.get(structure.module.as_str()).is_none()
                && !proposed_module_ids.contains(structure.module.as_str())
            {
                violations.push(Violation::new(
                    CanonRule::ProposalDeclarative,
                    format!(
                        "proposal `{}` references unknown module `{}` for struct `{}`",
                        proposal.id, structure.module, structure.id
                    ),
                ));
            }
        }
        for trait_spec in &resolved_nodes.traits {
            if modules.get(trait_spec.module.as_str()).is_none()
                && !proposed_module_ids.contains(trait_spec.module.as_str())
            {
                violations.push(Violation::new(
                    CanonRule::ProposalDeclarative,
                    format!(
                        "proposal `{}` references unknown module `{}` for trait `{}`",
                        proposal.id, trait_spec.module, trait_spec.id
                    ),
                ));
            }
        }

        for edge in &proposal.edges {
            if modules.get(edge.from.as_str()).is_none()
                && !proposed_module_ids.contains(edge.from.as_str())
            {
                violations.push(Violation::new(
                    CanonRule::ProposalDeclarative,
                    format!(
                        "proposal `{}` edge {} -> {} references unknown module `{}`",
                        proposal.id, edge.from, edge.to, edge.from
                    ),
                ));
            }
            if modules.get(edge.to.as_str()).is_none()
                && !proposed_module_ids.contains(edge.to.as_str())
            {
                violations.push(Violation::new(
                    CanonRule::ProposalDeclarative,
                    format!(
                        "proposal `{}` edge {} -> {} references unknown module `{}`",
                        proposal.id, edge.from, edge.to, edge.to
                    ),
                ));
            }
        }
        for api in &proposal.apis {
            if traits.get(api.trait_id.as_str()).is_none()
                && !proposed_trait_ids.contains(api.trait_id.as_str())
            {
                violations.push(Violation::new(
                    CanonRule::ProposalDeclarative,
                    format!(
                        "proposal `{}` references unknown trait `{}`",
                        proposal.id, api.trait_id
                    ),
                ));
            }
        }
    }

    for judgment in &ir.judgments {
        if proposals.get(judgment.proposal.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::JudgmentDecisions,
                format!(
                    "judgment `{}` references missing proposal `{}`",
                    judgment.id, judgment.proposal
                ),
            ));
        }
        if predicates.get(judgment.predicate.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::JudgmentDecisions,
                format!(
                    "judgment `{}` references missing predicate `{}`",
                    judgment.id, judgment.predicate
                ),
            ));
        }
    }

    for item in &ir.learning {
        if proposals.get(item.proposal.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::LearningDeclarations,
                format!(
                    "learning `{}` references missing proposal `{}`",
                    item.id, item.proposal
                ),
            ));
        }
        if item.new_rules.is_empty() {
            violations.push(Violation::new(
                CanonRule::LearningDeclarations,
                format!("learning `{}` must enumerate proposed rules", item.id),
            ));
        }
        if item.proof_object_hash.is_none() {
            violations.push(Violation::new(
                CanonRule::LearningDeclarations,
                format!("learning `{}` must include proof_object_hash", item.id),
            ));
        }
    }

    for delta in &ir.deltas {
        if !delta.append_only {
            violations.push(Violation::new(
                CanonRule::DeltaAppendOnly,
                format!("delta `{}` must be append-only", delta.id),
            ));
        }
        if !pipeline_stage_allows(delta.stage, delta.kind) {
            violations.push(Violation::new(
                CanonRule::DeltaPipeline,
                format!(
                    "delta `{}` of kind `{:?}` is not legal in stage `{:?}`",
                    delta.id, delta.kind, delta.stage
                ),
            ));
        }
        if proofs.get(delta.proof.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::DeltaProofs,
                format!(
                    "delta `{}` requires proof `{}` but it was not found",
                    delta.id, delta.proof
                ),
            ));
            continue;
        }
        let scope = proofs
            .get(delta.proof.as_str())
            .map(|p| p.scope)
            .expect("checked above");
        if !proof_scope_allows(delta.kind, scope) {
            violations.push(Violation::new(
                CanonRule::ProofScope,
                format!(
                    "delta `{}` of kind `{:?}` cannot carry proof scope `{:?}`",
                    delta.id, delta.kind, scope
                ),
            ));
        }
        if let Some(function_id) = &delta.related_function {
            if functions.get(function_id.as_str()).is_none() {
                violations.push(Violation::new(
                    CanonRule::EffectsAreDeltas,
                    format!(
                        "delta `{}` references missing function `{}`",
                        delta.id, function_id
                    ),
                ));
            }
        }
    }

    for structure in &ir.structs {
        for history in &structure.history {
            if deltas.get(history.delta.as_str()).is_none() {
                violations.push(Violation::new(
                    CanonRule::EffectsAreDeltas,
                    format!(
                        "struct `{}` references missing delta `{}` in history",
                        structure.name, history.delta
                    ),
                ));
            }
        }
    }

    for function in &ir.functions {
        for delta in &function.deltas {
            if deltas.get(delta.delta.as_str()).is_none() {
                violations.push(Violation::new(
                    CanonRule::EffectsAreDeltas,
                    format!(
                        "function `{}` references missing delta `{}`",
                        function.name, delta.delta
                    ),
                ));
            }
        }
        if function.outputs.is_empty() {
            violations.push(Violation::new(
                CanonRule::FunctionContracts,
                format!(
                    "function `{}` must explicitly enumerate outputs",
                    function.name
                ),
            ));
        }
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(ValidationErrors::new(violations))
    }
}

fn validate_module_graph<'a>(
    ir: &'a CanonicalIr,
    modules: &HashMap<&'a str, &'a Module>,
    violations: &mut Vec<Violation>,
) -> HashMap<&'a str, Vec<&'a str>> {
    let mut graph: DiGraphMap<&str, ()> = DiGraphMap::new();
    for id in modules.keys() {
        graph.add_node(*id);
    }

    let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();
    for edge in &ir.module_edges {
        let from = edge.source.as_str();
        let to = edge.target.as_str();
        if modules.get(from).is_none() || modules.get(to).is_none() {
            violations.push(Violation::new(
                CanonRule::ExplicitArtifacts,
                format!(
                    "module edge `{}` -> `{}` references unknown modules",
                    edge.source, edge.target
                ),
            ));
            continue;
        }
        if from == to {
            violations.push(Violation::new(
                CanonRule::ModuleSelfImport,
                format!("module `{}` may not import itself", edge.source),
            ));
            continue;
        }
        graph.add_edge(from, to, ());
        adjacency.entry(from).or_default().push(to);
    }

    if is_cyclic_directed(&graph) {
        violations.push(Violation::new(
            CanonRule::ModuleDag,
            "module import permissions must form a strict DAG",
        ));
    }

    adjacency
}

fn module_has_permission<'a>(
    from: &'a str,
    to: &'a str,
    adjacency: &HashMap<&'a str, Vec<&'a str>>,
    cache: &mut HashMap<&'a str, HashSet<&'a str>>,
) -> bool {
    if from == to {
        return true;
    }

    if let Some(known) = cache.get(from) {
        if known.contains(to) {
            return true;
        }
    }

    let mut stack = vec![from];
    let mut seen = HashSet::new();
    while let Some(node) = stack.pop() {
        if node == to {
            cache.entry(from).or_default().insert(to);
            return true;
        }
        if !seen.insert(node) {
            continue;
        }
        if let Some(neighbors) = adjacency.get(node) {
            for neighbor in neighbors {
                stack.push(*neighbor);
            }
        }
    }
    false
}

fn index_by_id<'a, T, F>(
    items: &'a [T],
    id_fn: F,
    rule: CanonRule,
    kind: &str,
    violations: &mut Vec<Violation>,
) -> HashMap<&'a str, &'a T>
where
    F: Fn(&'a T) -> &'a str,
{
    let mut map = HashMap::new();
    for item in items {
        let id = id_fn(item);
        if map.insert(id, item).is_some() {
            violations.push(Violation::new(
                rule,
                format!("duplicate {kind} id `{}` is forbidden", id),
            ));
        }
    }
    map
}

fn pipeline_stage_allows(stage: PipelineStage, kind: DeltaKind) -> bool {
    match stage {
        PipelineStage::Observe => true,
        PipelineStage::Learn => matches!(
            kind,
            DeltaKind::State | DeltaKind::Structure | DeltaKind::History
        ),
        PipelineStage::Decide => matches!(kind, DeltaKind::Structure | DeltaKind::History),
        PipelineStage::Plan => matches!(kind, DeltaKind::Structure | DeltaKind::History),
        PipelineStage::Act => matches!(kind, DeltaKind::State | DeltaKind::Io | DeltaKind::History),
    }
}

fn proof_scope_allows(kind: DeltaKind, scope: ProofScope) -> bool {
    match kind {
        DeltaKind::State => matches!(scope, ProofScope::Execution),
        DeltaKind::Io => matches!(scope, ProofScope::Execution),
        DeltaKind::Structure => matches!(scope, ProofScope::Structure | ProofScope::Law),
        DeltaKind::History => matches!(scope, ProofScope::Structure | ProofScope::Execution),
    }
}

#[derive(Debug, Clone)]
pub struct Violation {
    rule: CanonRule,
    detail: String,
}

impl Violation {
    pub fn new(rule: CanonRule, detail: impl Into<String>) -> Self {
        Self {
            rule,
            detail: detail.into(),
        }
    }

    pub fn rule(&self) -> CanonRule {
        self.rule
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonRule {
    ExplicitArtifacts,
    ModuleDag,
    ModuleSelfImport,
    ImplBinding,
    ExecutionOnlyInImpl,
    FunctionContracts,
    EffectsAreDeltas,
    DeltaProofs,
    DeltaAppendOnly,
    CallGraphPublicApis,
    CallGraphRespectsDag,
    CallGraphAcyclic,
    TickGraphAcyclic,
    TickGraphEdgesDeclared,
    LoopContinuationJudgment,
    GpuLawfulMath,
    ProposalDeclarative,
    JudgmentDecisions,
    LearningDeclarations,
    TraitVerbs,
    TickRoot,
    ProjectEnvelope,
    ExternalDependencies,
    DeltaPipeline,
    ProofScope,
    VersionEvolution,
    TickEpochs,
    PlanArtifacts,
    ExecutionBoundary,
    AdmissionBridge,
}

impl CanonRule {
    pub fn code(self) -> &'static str {
        match self {
            CanonRule::ExplicitArtifacts => "Rule 5",
            CanonRule::ModuleDag => "Rule 13",
            CanonRule::ModuleSelfImport => "Rule 14",
            CanonRule::ImplBinding => "Rule 26",
            CanonRule::ExecutionOnlyInImpl => "Rule 27",
            CanonRule::FunctionContracts => "Rules 31-34",
            CanonRule::EffectsAreDeltas => "Rules 34-38",
            CanonRule::DeltaProofs => "Rules 67-69",
            CanonRule::DeltaAppendOnly => "Rule 38",
            CanonRule::CallGraphPublicApis => "Rule 42",
            CanonRule::CallGraphRespectsDag => "Rule 43",
            CanonRule::CallGraphAcyclic => "Rules 49 & 75",
            CanonRule::TickGraphAcyclic => "Rules 48-50",
            CanonRule::TickGraphEdgesDeclared => "Rule 41",
            CanonRule::LoopContinuationJudgment => "Rule 52",
            CanonRule::GpuLawfulMath => "Rules 81-89",
            CanonRule::ProposalDeclarative => "Rules 57-59",
            CanonRule::JudgmentDecisions => "Rules 53-55",
            CanonRule::LearningDeclarations => "Rules 61-62",
            CanonRule::TraitVerbs => "Rule 23",
            CanonRule::TickRoot => "Rules 46-51",
            CanonRule::ProjectEnvelope => "Rules 5-8",
            CanonRule::ExternalDependencies => "Rule 5",
            CanonRule::DeltaPipeline => "Rules 50-79",
            CanonRule::ProofScope => "Rules 63-69",
            CanonRule::VersionEvolution => "Rule 99",
            CanonRule::TickEpochs => "Rules 46-51",
            CanonRule::PlanArtifacts => "Rules 57-60",
            CanonRule::ExecutionBoundary => "Rules 46-55",
            CanonRule::AdmissionBridge => "Rules 53-69",
        }
    }

    pub fn text(self) -> &'static str {
        match self {
            CanonRule::ExplicitArtifacts => {
                "All referenced artifacts must exist and remain unique."
            }
            CanonRule::ModuleDag => "Modules must form a strict acyclic DAG.",
            CanonRule::ModuleSelfImport => "Modules may not import themselves.",
            CanonRule::ImplBinding => "Impl blocks must bind nouns to verbs lawfully.",
            CanonRule::ExecutionOnlyInImpl => "Execution may only occur inside impl blocks.",
            CanonRule::FunctionContracts => {
                "Functions must be total with explicit IO and delta effects."
            }
            CanonRule::EffectsAreDeltas => "All effects must surface as declared deltas.",
            CanonRule::DeltaProofs => "Every delta must carry an attached proof obligation.",
            CanonRule::DeltaAppendOnly => "Deltas must be append-only artifacts.",
            CanonRule::CallGraphPublicApis => "Call edges may target public APIs only.",
            CanonRule::CallGraphRespectsDag => {
                "Call graphs must respect module import permissions."
            }
            CanonRule::CallGraphAcyclic => "No recursion or cycles are allowed in execution.",
            CanonRule::TickGraphAcyclic => "Each tick graph must be acyclic.",
            CanonRule::TickGraphEdgesDeclared => "Tick graphs must align with the call graph.",
            CanonRule::LoopContinuationJudgment => "Loop continuation is a judgment decision.",
            CanonRule::GpuLawfulMath => "GPU kernels may contain lawful math only.",
            CanonRule::ProposalDeclarative => {
                "Proposals must enumerate goals, nodes, APIs, and edges."
            }
            CanonRule::JudgmentDecisions => "Judgment references must be explicit and structural.",
            CanonRule::LearningDeclarations => "Learning proposals must be explicit artifacts.",
            CanonRule::TraitVerbs => "Traits declare verbs/capabilities in a unique namespace.",
            CanonRule::TickRoot => "Each IR instance must declare explicit tick roots.",
            CanonRule::ProjectEnvelope => "Projects must declare their package envelope.",
            CanonRule::ExternalDependencies => "External dependencies must be explicit and unique.",
            CanonRule::DeltaPipeline => "Pipeline stages may emit only lawful delta kinds.",
            CanonRule::ProofScope => "Delta proofs must match their semantic scope.",
            CanonRule::VersionEvolution => "Version upgrades require explicit law-scoped proofs.",
            CanonRule::TickEpochs => "Ticks must belong to an acyclic epoch hierarchy.",
            CanonRule::PlanArtifacts => "Plans must reference accepted judgments and lawful steps.",
            CanonRule::ExecutionBoundary => {
                "Execution results must reference known plans, ticks, and deltas."
            }
            CanonRule::AdmissionBridge => {
                "Admissions bridge judgments to applied deltas deterministically."
            }
        }
    }
}

#[derive(Debug)]
pub struct ValidationErrors {
    violations: Vec<Violation>,
}

impl ValidationErrors {
    pub fn new(violations: Vec<Violation>) -> Self {
        Self { violations }
    }

    pub fn violations(&self) -> &[Violation] {
        &self.violations
    }
}

impl fmt::Display for ValidationErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Canon validation failed with {} violation(s):",
            self.violations.len()
        )?;
        for violation in &self.violations {
            writeln!(
                f,
                "- {} ({}) → {}",
                violation.rule().code(),
                violation.rule().text(),
                violation.detail()
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationErrors {}
