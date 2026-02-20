use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{
    ir::proposal::{derive_word_from_identifier, sanitize_identifier, ModuleSpec, ResolvedProposalNodes, StructSpec, TraitSpec},
    ir::{Delta, DeltaId, DeltaKind, DeltaPayload, FunctionSignature, PipelineStage, Proposal, ProposedEdge, TraitFunction, TypeKind, TypeRef, ValuePort, Visibility, Word},
};

use super::{AcceptProposalError, ProposalAcceptanceInput};

pub(super) fn emit_deltas(
    input: &ProposalAcceptanceInput, proposal: &Proposal, resolved: &ResolvedProposalNodes, trait_function_map: &BTreeMap<String, Vec<TraitFunction>>, known_delta_ids: &mut HashSet<String>,
) -> Result<(Vec<Delta>, Vec<DeltaId>), AcceptProposalError> {
    let mut deltas = Vec::new();
    let mut delta_ids = Vec::new();

    for module in sorted_modules(&resolved.modules) {
        let payload =
            DeltaPayload::AddModule { module_id: module.id.clone(), name: module.name.clone(), visibility: Visibility::Public, description: format!("Accepted via proposal `{}`.", proposal.id) };
        let identifier = delta_identifier(&proposal.id, "module", &module.id);
        let delta = build_delta(&identifier, input, &payload, None, &module.id)?;
        register_delta(known_delta_ids, &delta)?;
        delta_ids.push(delta.id.clone());
        deltas.push(delta);
    }

    for structure in sorted_structs(&resolved.structs) {
        let payload = DeltaPayload::AddStruct { module: structure.module.clone(), struct_id: structure.id.clone(), name: structure.name.clone() };
        let identifier = delta_identifier(&proposal.id, "struct", &structure.id);
        let delta = build_delta(&identifier, input, &payload, None, &structure.id)?;
        register_delta(known_delta_ids, &delta)?;
        delta_ids.push(delta.id.clone());
        deltas.push(delta);
    }

    for trait_spec in sorted_traits(&resolved.traits) {
        let payload = DeltaPayload::AddTrait { module: trait_spec.module.clone(), trait_id: trait_spec.id.clone(), name: trait_spec.name.clone() };
        let identifier = delta_identifier(&proposal.id, "trait", &trait_spec.id);
        let delta = build_delta(&identifier, input, &payload, None, &trait_spec.id)?;
        register_delta(known_delta_ids, &delta)?;
        delta_ids.push(delta.id.clone());
        deltas.push(delta);
    }

    for (trait_id, functions) in trait_function_map {
        for function in functions {
            let payload = DeltaPayload::AddTraitFunction { trait_id: trait_id.clone(), function: function.clone() };
            let identifier = delta_identifier(&proposal.id, "trait_fn", &function.id);
            let delta = build_delta(&identifier, input, &payload, None, &function.id)?;
            register_delta(known_delta_ids, &delta)?;
            delta_ids.push(delta.id.clone());
            deltas.push(delta);
        }
    }

    let trait_targets: Vec<_> = trait_function_map.keys().cloned().collect();
    let trait_modules: HashMap<String, String> = resolved.traits.iter().map(|spec| (spec.id.clone(), spec.module.clone())).collect();
    for structure in sorted_structs(&resolved.structs) {
        for trait_id in trait_targets.iter() {
            if let Some(module) = trait_modules.get(trait_id) {
                if module != &structure.module {
                    continue;
                }
            }
            let impl_id = format!("impl.{}.{}", sanitize_identifier(&structure.id), sanitize_identifier(trait_id),);
            let payload = DeltaPayload::AddImpl { module: structure.module.clone(), impl_id: impl_id.clone(), struct_id: structure.id.clone(), trait_id: trait_id.clone() };
            let identifier = delta_identifier(&proposal.id, "impl", &impl_id);
            let delta = build_delta(&identifier, input, &payload, None, &impl_id)?;
            register_delta(known_delta_ids, &delta)?;
            delta_ids.push(delta.id.clone());
            deltas.push(delta);

            if let Some(functions) = trait_function_map.get(trait_id) {
                for function in functions {
                    let signature = build_function_signature(function, &structure.name)?;
                    let function_id = format!("fn.{}.{}", sanitize_identifier(&impl_id), sanitize_identifier(&function.id),);
                    let payload = DeltaPayload::AddFunction { function_id: function_id.clone(), impl_id: impl_id.clone(), signature };
                    let identifier = delta_identifier(&proposal.id, "fn", &function_id);
                    let delta = build_delta(&identifier, input, &payload, Some(function_id.clone()), &function_id)?;
                    register_delta(known_delta_ids, &delta)?;
                    delta_ids.push(delta.id.clone());
                    deltas.push(delta);
                }
            }
        }
    }

    for edge in sorted_edges(&proposal.edges) {
        let payload = DeltaPayload::AddModuleEdge { from: edge.from.clone(), to: edge.to.clone(), rationale: edge.rationale.clone() };
        let artifact = format!("{}->{}", edge.from, edge.to);
        let identifier = delta_identifier(&proposal.id, "module_edge", &artifact);
        let delta = build_delta(&identifier, input, &payload, None, &artifact)?;
        register_delta(known_delta_ids, &delta)?;
        delta_ids.push(delta.id.clone());
        deltas.push(delta);
    }

    Ok((deltas, delta_ids))
}

pub(super) fn build_trait_function_map(proposal: &Proposal) -> Result<BTreeMap<String, Vec<TraitFunction>>, AcceptProposalError> {
    let mut map: BTreeMap<String, Vec<TraitFunction>> = BTreeMap::new();
    for api in &proposal.apis {
        if api.functions.is_empty() {
            return Err(AcceptProposalError::EmptyApi { trait_id: api.trait_id.clone() });
        }
        let entry = map.entry(api.trait_id.clone()).or_default();
        for fn_id in &api.functions {
            let name = derive_word_from_identifier(fn_id)?;
            let output = default_trait_output(&name)?;
            entry.push(TraitFunction { id: fn_id.clone(), name: name.clone(), inputs: vec![], outputs: vec![output], default_body: None });
        }
    }
    for functions in map.values_mut() {
        functions.sort_by(|a, b| a.id.cmp(&b.id));
        functions.dedup_by(|a, b| a.id == b.id);
    }
    Ok(map)
}

fn default_trait_output(name: &Word) -> Result<ValuePort, AcceptProposalError> {
    let output_name = Word::new(format!("{}Output", name.as_str())).map_err(AcceptProposalError::Word)?;
    let ty_name = Word::new(format!("{}Result", name.as_str())).map_err(AcceptProposalError::Word)?;
    Ok(ValuePort { name: output_name, ty: TypeRef { name: ty_name, kind: TypeKind::External, params: vec![], ref_kind: crate::ir::RefKind::None, lifetime: None } })
}

fn build_delta(identifier: &str, input: &ProposalAcceptanceInput, payload: &DeltaPayload, related_function: Option<String>, artifact_label: &str) -> Result<Delta, AcceptProposalError> {
    Ok(Delta {
        id: identifier.to_owned(),
        kind: DeltaKind::Structure,
        stage: PipelineStage::Decide,
        append_only: true,
        proof: input.proof_id.clone(),
        description: format!("Proposal `{}` impacts `{}`.", input.proposal_id, artifact_label),
        related_function,
        payload: Some(payload.clone()),
        proof_object_hash: None,
    })
}

fn register_delta(known_ids: &mut HashSet<String>, delta: &Delta) -> Result<(), AcceptProposalError> {
    if known_ids.insert(delta.id.clone()) {
        Ok(())
    } else {
        Err(AcceptProposalError::DuplicateDelta(delta.id.clone()))
    }
}

fn delta_identifier(proposal_id: &str, kind: &str, artifact: &str) -> String {
    format!("delta.{}.{}.{}", sanitize_identifier(proposal_id), kind, sanitize_identifier(artifact))
}

fn build_function_signature(trait_function: &TraitFunction, struct_name: &Word) -> Result<FunctionSignature, AcceptProposalError> {
    let signature_name = format!("{}{}", trait_function.name, struct_name);
    let name = Word::new(signature_name)?;
    Ok(FunctionSignature {
        name,
        receiver: crate::ir::Receiver::None,
        is_async: false,
        is_unsafe: false,
        lifetime_params: Vec::new(),
        generics: Vec::new(),
        where_clauses: Vec::new(),
        doc: None,
        inputs: trait_function.inputs.clone(),
        outputs: trait_function.outputs.clone(),
        visibility: Visibility::Private,
        trait_function: trait_function.id.clone(),
    })
}

fn sorted_modules(mods: &[ModuleSpec]) -> Vec<ModuleSpec> {
    let mut modules = mods.to_vec();
    modules.sort_by(|a, b| a.id.cmp(&b.id));
    modules
}

fn sorted_structs(structs: &[StructSpec]) -> Vec<StructSpec> {
    let mut list = structs.to_vec();
    list.sort_by(|a, b| a.id.cmp(&b.id));
    list
}

fn sorted_traits(traits: &[TraitSpec]) -> Vec<TraitSpec> {
    let mut list = traits.to_vec();
    list.sort_by(|a, b| a.id.cmp(&b.id));
    list
}

fn sorted_edges(edges: &[ProposedEdge]) -> Vec<ProposedEdge> {
    let mut list = edges.to_vec();
    list.sort_by(|a, b| a.from.cmp(&b.from).then_with(|| a.to.cmp(&b.to)));
    list
}
