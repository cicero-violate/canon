use std::collections::{BTreeMap, HashMap, HashSet};

use thiserror::Error;

use crate::{
    evolution::{EvolutionError, apply_deltas},
    ir::{
        CanonicalIr, Delta, DeltaAdmission, DeltaId, DeltaKind, DeltaPayload, FunctionSignature,
        Judgment, JudgmentDecision, JudgmentPredicate, PipelineStage, Proof, ProofArtifact,
        ProofScope, Proposal, ProposalStatus, Tick, TraitFunction, TypeKind, TypeRef, ValuePort,
        Visibility, Word, WordError,
    },
    proof::smt_bridge::attach_function_proofs,
    proposal::{
        DslProposalArtifacts, DslProposalError, ModuleSpec, ProposalResolutionError,
        ResolvedProposalNodes, StructSpec, TraitSpec, create_proposal_from_dsl,
        derive_word_from_identifier, resolve_proposal_nodes, sanitize_identifier,
    },
};

#[derive(Debug, Clone)]
pub struct ProposalAcceptanceInput {
    pub proposal_id: String,
    pub proof_id: String,
    pub predicate_id: String,
    pub judgment_id: String,
    pub admission_id: String,
    pub tick_id: String,
    pub rationale: String,
}

#[derive(Debug, Clone)]
pub struct ProposalAcceptance {
    pub ir: CanonicalIr,
    pub delta_ids: Vec<DeltaId>,
    pub judgment_id: String,
    pub admission_id: String,
}

const DSL_PROOF_ID: &str = "proof.dsl.bootstrap";
const DSL_PREDICATE_ID: &str = "predicate.dsl.autoaccept";
const DSL_TICK_ID: &str = "tick.bootstrap";

#[derive(Debug, Error)]
pub enum AutoAcceptDslError {
    #[error(transparent)]
    Proposal(#[from] DslProposalError),
    #[error(transparent)]
    Accept(#[from] AcceptProposalError),
    #[error("unable to infer tick graph for bootstrap tick")]
    MissingTickGraph,
}

pub fn auto_accept_dsl_proposal(
    ir: &CanonicalIr,
    dsl_source: &str,
) -> Result<ProposalAcceptance, AutoAcceptDslError> {
    let DslProposalArtifacts {
        proposal,
        goal_slug,
    } = create_proposal_from_dsl(dsl_source)?;
    let mut working = ir.clone();
    ensure_dsl_proof(&mut working);
    ensure_dsl_predicate(&mut working);
    ensure_dsl_tick(&mut working).map_err(|_| AutoAcceptDslError::MissingTickGraph)?;
    working.proposals.push(proposal);

    let proposal_id = format!("proposal.dsl.{goal_slug}");
    let judgment = format!("judgment.dsl.{goal_slug}");
    let admission = format!("admission.dsl.{goal_slug}");
    let acceptance = accept_proposal(
        &working,
        ProposalAcceptanceInput {
            proposal_id,
            proof_id: DSL_PROOF_ID.to_string(),
            predicate_id: DSL_PREDICATE_ID.to_string(),
            judgment_id: judgment,
            admission_id: admission,
            tick_id: DSL_TICK_ID.to_string(),
            rationale: "Auto-accepted DSL proposal.".to_string(),
        },
    )?;
    Ok(acceptance)
}

#[derive(Debug, Error)]
pub enum AcceptProposalError {
    #[error("proposal `{0}` does not exist")]
    UnknownProposal(String),
    #[error("proposal `{proposal}` must be submitted; found status `{status:?}`")]
    InvalidProposalStatus {
        proposal: String,
        status: ProposalStatus,
    },
    #[error("proposal `{0}` must enumerate nodes, APIs, and edges")]
    IncompleteProposal(String),
    #[error("proof `{0}` is not registered")]
    UnknownProof(String),
    #[error("judgment predicate `{0}` is not registered")]
    UnknownPredicate(String),
    #[error("tick `{0}` is not registered")]
    UnknownTick(String),
    #[error("judgment `{0}` already exists")]
    DuplicateJudgment(String),
    #[error("admission `{0}` already exists")]
    DuplicateAdmission(String),
    #[error("delta `{0}` already exists")]
    DuplicateDelta(String),
    #[error("proposal `{0}` did not emit any structural deltas")]
    NoDeltas(String),
    #[error("module `{0}` referenced by proposal is unknown")]
    UnknownModule(String),
    #[error("trait `{0}` referenced by proposal is unknown")]
    UnknownTrait(String),
    #[error("proposal referenced trait `{trait_id}` without declaring any functions")]
    EmptyApi { trait_id: String },
    #[error("artifact `{kind}` with id `{id}` already exists in Canon")]
    ArtifactExists { kind: &'static str, id: String },
    #[error(transparent)]
    Resolution(#[from] ProposalResolutionError),
    #[error(transparent)]
    Evolution(#[from] EvolutionError),
    #[error("word error: {0}")]
    Word(#[from] WordError),
    #[error(transparent)]
    Proof(#[from] crate::proof::smt_bridge::SmtError),
}

pub fn accept_proposal(
    ir: &CanonicalIr,
    input: ProposalAcceptanceInput,
) -> Result<ProposalAcceptance, AcceptProposalError> {
    let mut working = ir.clone();
    let proposal_index = working
        .proposals
        .iter()
        .position(|proposal| proposal.id == input.proposal_id)
        .ok_or_else(|| AcceptProposalError::UnknownProposal(input.proposal_id.clone()))?;

    let proposal = working.proposals.get(proposal_index).expect("index");
    enforce_proposal_ready(proposal)?;
    let resolved = resolve_proposal_nodes(proposal)?;
    let trait_function_map = build_trait_function_map(proposal)?;
    enforce_references(ir, proposal, &resolved, &trait_function_map)?;
    ensure_proof_exists(&working, &input.proof_id)?;
    ensure_predicate_exists(&working, &input.predicate_id)?;
    ensure_tick_exists(&working, &input.tick_id)?;
    ensure_unique_judgment(&working, &input.judgment_id)?;
    ensure_unique_admission(&working, &input.admission_id)?;

    let mut known_delta_ids: HashSet<String> =
        working.deltas.iter().map(|d| d.id.clone()).collect();
    let (mut deltas, delta_ids) = emit_deltas(
        &input,
        proposal,
        &resolved,
        &trait_function_map,
        &mut known_delta_ids,
    )?;
    if deltas.is_empty() {
        return Err(AcceptProposalError::NoDeltas(proposal.id.clone()));
    }

    working.deltas.append(&mut deltas);
    working.proposals[proposal_index].status = ProposalStatus::Accepted;
    working.judgments.push(Judgment {
        id: input.judgment_id.clone(),
        proposal: input.proposal_id.clone(),
        predicate: input.predicate_id.clone(),
        decision: JudgmentDecision::Accept,
        rationale: input.rationale.clone(),
    });
    working.admissions.push(DeltaAdmission {
        id: input.admission_id.clone(),
        judgment: input.judgment_id.clone(),
        tick: input.tick_id.clone(),
        delta_ids: delta_ids.clone(),
    });

    let mut evolved = apply_deltas(&working, &[input.admission_id.clone()])?;
    attach_function_proofs(&mut evolved)?;
    Ok(ProposalAcceptance {
        ir: evolved,
        delta_ids,
        judgment_id: input.judgment_id,
        admission_id: input.admission_id,
    })
}

fn enforce_proposal_ready(proposal: &Proposal) -> Result<(), AcceptProposalError> {
    if proposal.status != ProposalStatus::Submitted {
        return Err(AcceptProposalError::InvalidProposalStatus {
            proposal: proposal.id.clone(),
            status: proposal.status,
        });
    }
    if proposal.nodes.is_empty() || proposal.apis.is_empty() || proposal.edges.is_empty() {
        return Err(AcceptProposalError::IncompleteProposal(proposal.id.clone()));
    }
    Ok(())
}

fn ensure_proof_exists(ir: &CanonicalIr, proof_id: &str) -> Result<(), AcceptProposalError> {
    if ir.proofs.iter().any(|proof| proof.id == proof_id) {
        Ok(())
    } else {
        Err(AcceptProposalError::UnknownProof(proof_id.to_owned()))
    }
}

fn ensure_predicate_exists(
    ir: &CanonicalIr,
    predicate_id: &str,
) -> Result<(), AcceptProposalError> {
    if ir.judgment_predicates.iter().any(|p| p.id == predicate_id) {
        Ok(())
    } else {
        Err(AcceptProposalError::UnknownPredicate(
            predicate_id.to_owned(),
        ))
    }
}

fn ensure_tick_exists(ir: &CanonicalIr, tick_id: &str) -> Result<(), AcceptProposalError> {
    if ir.ticks.iter().any(|tick| tick.id == tick_id) {
        Ok(())
    } else {
        Err(AcceptProposalError::UnknownTick(tick_id.to_owned()))
    }
}

fn ensure_unique_judgment(ir: &CanonicalIr, judgment_id: &str) -> Result<(), AcceptProposalError> {
    if ir.judgments.iter().any(|j| j.id == judgment_id) {
        Err(AcceptProposalError::DuplicateJudgment(
            judgment_id.to_owned(),
        ))
    } else {
        Ok(())
    }
}

fn ensure_unique_admission(
    ir: &CanonicalIr,
    admission_id: &str,
) -> Result<(), AcceptProposalError> {
    if ir.admissions.iter().any(|a| a.id == admission_id) {
        Err(AcceptProposalError::DuplicateAdmission(
            admission_id.to_owned(),
        ))
    } else {
        Ok(())
    }
}

fn enforce_references(
    ir: &CanonicalIr,
    proposal: &Proposal,
    resolved: &ResolvedProposalNodes,
    trait_functions: &BTreeMap<String, Vec<TraitFunction>>,
) -> Result<(), AcceptProposalError> {
    let existing_modules: HashSet<&str> = ir.modules.iter().map(|m| m.id.as_str()).collect();
    for module in &resolved.modules {
        if existing_modules.contains(module.id.as_str()) {
            return Err(AcceptProposalError::ArtifactExists {
                kind: "module",
                id: module.id.clone(),
            });
        }
    }
    let known_modules: HashSet<&str> = existing_modules
        .into_iter()
        .chain(resolved.modules.iter().map(|m| m.id.as_str()))
        .collect();

    let existing_traits: HashSet<&str> = ir.traits.iter().map(|t| t.id.as_str()).collect();
    for trait_spec in &resolved.traits {
        if existing_traits.contains(trait_spec.id.as_str()) {
            return Err(AcceptProposalError::ArtifactExists {
                kind: "trait",
                id: trait_spec.id.clone(),
            });
        }
    }
    for trait_spec in &resolved.traits {
        if !known_modules.contains(trait_spec.module.as_str()) {
            return Err(AcceptProposalError::UnknownModule(
                trait_spec.module.clone(),
            ));
        }
    }
    let known_traits: HashSet<&str> = existing_traits
        .into_iter()
        .chain(resolved.traits.iter().map(|t| t.id.as_str()))
        .collect();

    for structure in &resolved.structs {
        if !known_modules.contains(structure.module.as_str()) {
            return Err(AcceptProposalError::UnknownModule(structure.module.clone()));
        }
        if ir.structs.iter().any(|s| s.id == structure.id) {
            return Err(AcceptProposalError::ArtifactExists {
                kind: "struct",
                id: structure.id.clone(),
            });
        }
    }

    for edge in &proposal.edges {
        if !known_modules.contains(edge.from.as_str()) {
            return Err(AcceptProposalError::UnknownModule(edge.from.clone()));
        }
        if !known_modules.contains(edge.to.as_str()) {
            return Err(AcceptProposalError::UnknownModule(edge.to.clone()));
        }
    }

    for trait_id in trait_functions.keys() {
        if !known_traits.contains(trait_id.as_str()) {
            return Err(AcceptProposalError::UnknownTrait(trait_id.clone()));
        }
    }

    let existing_impls: HashSet<(String, String)> = ir
        .impl_blocks
        .iter()
        .map(|block| (block.struct_id.clone(), block.trait_id.clone()))
        .collect();
    for structure in &resolved.structs {
        for trait_id in trait_functions.keys() {
            if existing_impls.contains(&(structure.id.clone(), trait_id.clone())) {
                return Err(AcceptProposalError::ArtifactExists {
                    kind: "impl",
                    id: format!("{}->{}", structure.id, trait_id),
                });
            }
        }
    }

    Ok(())
}

fn emit_deltas(
    input: &ProposalAcceptanceInput,
    proposal: &Proposal,
    resolved: &ResolvedProposalNodes,
    trait_function_map: &BTreeMap<String, Vec<TraitFunction>>,
    known_delta_ids: &mut HashSet<String>,
) -> Result<(Vec<Delta>, Vec<DeltaId>), AcceptProposalError> {
    let mut deltas = Vec::new();
    let mut delta_ids = Vec::new();

    for module in sorted_modules(&resolved.modules) {
        let payload = DeltaPayload::AddModule {
            module_id: module.id.clone(),
            name: module.name.clone(),
            visibility: Visibility::Public,
            description: format!("Accepted via proposal `{}`.", proposal.id),
        };
        let identifier = delta_identifier(&proposal.id, "module", &module.id);
        let delta = build_delta(&identifier, input, &payload, None, &module.id)?;
        register_delta(known_delta_ids, &delta)?;
        delta_ids.push(delta.id.clone());
        deltas.push(delta);
    }

    for structure in sorted_structs(&resolved.structs) {
        let payload = DeltaPayload::AddStruct {
            module: structure.module.clone(),
            struct_id: structure.id.clone(),
            name: structure.name.clone(),
        };
        let identifier = delta_identifier(&proposal.id, "struct", &structure.id);
        let delta = build_delta(&identifier, input, &payload, None, &structure.id)?;
        register_delta(known_delta_ids, &delta)?;
        delta_ids.push(delta.id.clone());
        deltas.push(delta);
    }

    for trait_spec in sorted_traits(&resolved.traits) {
        let payload = DeltaPayload::AddTrait {
            module: trait_spec.module.clone(),
            trait_id: trait_spec.id.clone(),
            name: trait_spec.name.clone(),
        };
        let identifier = delta_identifier(&proposal.id, "trait", &trait_spec.id);
        let delta = build_delta(&identifier, input, &payload, None, &trait_spec.id)?;
        register_delta(known_delta_ids, &delta)?;
        delta_ids.push(delta.id.clone());
        deltas.push(delta);
    }

    for (trait_id, functions) in trait_function_map {
        for function in functions {
            let payload = DeltaPayload::AddTraitFunction {
                trait_id: trait_id.clone(),
                function: function.clone(),
            };
            let identifier = delta_identifier(&proposal.id, "trait_fn", &function.id);
            let delta = build_delta(&identifier, input, &payload, None, &function.id)?;
            register_delta(known_delta_ids, &delta)?;
            delta_ids.push(delta.id.clone());
            deltas.push(delta);
        }
    }

    let trait_targets: Vec<_> = trait_function_map.keys().cloned().collect();
    let trait_modules: HashMap<String, String> = resolved
        .traits
        .iter()
        .map(|spec| (spec.id.clone(), spec.module.clone()))
        .collect();
    for structure in sorted_structs(&resolved.structs) {
        for trait_id in trait_targets.iter() {
            if let Some(module) = trait_modules.get(trait_id) {
                if module != &structure.module {
                    continue;
                }
            }
            let impl_id = format!(
                "impl.{}.{}",
                sanitize_identifier(&structure.id),
                sanitize_identifier(trait_id),
            );
            let payload = DeltaPayload::AddImpl {
                module: structure.module.clone(),
                impl_id: impl_id.clone(),
                struct_id: structure.id.clone(),
                trait_id: trait_id.clone(),
            };
            let identifier = delta_identifier(&proposal.id, "impl", &impl_id);
            let delta = build_delta(&identifier, input, &payload, None, &impl_id)?;
            register_delta(known_delta_ids, &delta)?;
            delta_ids.push(delta.id.clone());
            deltas.push(delta);

            if let Some(functions) = trait_function_map.get(trait_id) {
                for function in functions {
                    let signature = build_function_signature(function, &structure.name)?;
                    let function_id = format!(
                        "fn.{}.{}",
                        sanitize_identifier(&impl_id),
                        sanitize_identifier(&function.id),
                    );
                    let payload = DeltaPayload::AddFunction {
                        function_id: function_id.clone(),
                        impl_id: impl_id.clone(),
                        signature,
                    };
                    let identifier = delta_identifier(&proposal.id, "fn", &function_id);
                    let delta = build_delta(
                        &identifier,
                        input,
                        &payload,
                        Some(function_id.clone()),
                        &function_id,
                    )?;
                    register_delta(known_delta_ids, &delta)?;
                    delta_ids.push(delta.id.clone());
                    deltas.push(delta);
                }
            }
        }
    }

    for edge in sorted_edges(&proposal.edges) {
        let payload = DeltaPayload::AddModuleEdge {
            from: edge.from.clone(),
            to: edge.to.clone(),
            rationale: edge.rationale.clone(),
        };
        let artifact = format!("{}->{}", edge.from, edge.to);
        let identifier = delta_identifier(&proposal.id, "module_edge", &artifact);
        let delta = build_delta(&identifier, input, &payload, None, &artifact)?;
        register_delta(known_delta_ids, &delta)?;
        delta_ids.push(delta.id.clone());
        deltas.push(delta);
    }

    Ok((deltas, delta_ids))
}

fn build_trait_function_map(
    proposal: &Proposal,
) -> Result<BTreeMap<String, Vec<TraitFunction>>, AcceptProposalError> {
    let mut map: BTreeMap<String, Vec<TraitFunction>> = BTreeMap::new();
    for api in &proposal.apis {
        if api.functions.is_empty() {
            return Err(AcceptProposalError::EmptyApi {
                trait_id: api.trait_id.clone(),
            });
        }
        let entry = map.entry(api.trait_id.clone()).or_default();
        for fn_id in &api.functions {
            let name = derive_word_from_identifier(fn_id)?;
            let output = default_trait_output(&name)?;
            entry.push(TraitFunction {
                id: fn_id.clone(),
                name: name.clone(),
                inputs: vec![],
                outputs: vec![output],
            });
        }
    }
    for functions in map.values_mut() {
        functions.sort_by(|a, b| a.id.cmp(&b.id));
        functions.dedup_by(|a, b| a.id == b.id);
    }
    Ok(map)
}

fn default_trait_output(name: &Word) -> Result<ValuePort, AcceptProposalError> {
    let output_name =
        Word::new(format!("{}Output", name.as_str())).map_err(AcceptProposalError::Word)?;
    let ty_name =
        Word::new(format!("{}Result", name.as_str())).map_err(AcceptProposalError::Word)?;
    Ok(ValuePort {
        name: output_name,
        ty: TypeRef {
            name: ty_name,
            kind: TypeKind::External,
        },
    })
}

fn build_delta(
    identifier: &str,
    input: &ProposalAcceptanceInput,
    payload: &DeltaPayload,
    related_function: Option<String>,
    artifact_label: &str,
) -> Result<Delta, AcceptProposalError> {
    Ok(Delta {
        id: identifier.to_owned(),
        kind: DeltaKind::Structure,
        stage: PipelineStage::Decide,
        append_only: true,
        proof: input.proof_id.clone(),
        description: format!(
            "Proposal `{}` impacts `{}`.",
            input.proposal_id, artifact_label
        ),
        related_function,
        payload: Some(payload.clone()),
        proof_object_hash: None,
    })
}

fn register_delta(
    known_ids: &mut HashSet<String>,
    delta: &Delta,
) -> Result<(), AcceptProposalError> {
    if known_ids.insert(delta.id.clone()) {
        Ok(())
    } else {
        Err(AcceptProposalError::DuplicateDelta(delta.id.clone()))
    }
}

fn delta_identifier(proposal_id: &str, kind: &str, artifact: &str) -> String {
    format!(
        "delta.{}.{}.{}",
        sanitize_identifier(proposal_id),
        kind,
        sanitize_identifier(artifact)
    )
}

fn build_function_signature(
    trait_function: &TraitFunction,
    struct_name: &Word,
) -> Result<FunctionSignature, AcceptProposalError> {
    let signature_name = format!("{}{}", trait_function.name, struct_name);
    let name = Word::new(signature_name)?;
    Ok(FunctionSignature {
        name,
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

fn sorted_edges(edges: &[crate::ir::ProposedEdge]) -> Vec<crate::ir::ProposedEdge> {
    let mut list = edges.to_vec();
    list.sort_by(|a, b| a.from.cmp(&b.from).then_with(|| a.to.cmp(&b.to)));
    list
}

fn ensure_dsl_proof(ir: &mut CanonicalIr) {
    if ir.proofs.iter().any(|proof| proof.id == DSL_PROOF_ID) {
        return;
    }
    ir.proofs.push(Proof {
        id: DSL_PROOF_ID.to_string(),
        invariant: "DSL submissions are structurally lawful.".to_string(),
        scope: ProofScope::Structure,
        evidence: ProofArtifact {
            uri: "canon://dsl/bootstrap".to_string(),
            hash: "dsl-bootstrap".to_string(),
        },
        proof_object_hash: None,
    });
}

fn ensure_dsl_predicate(ir: &mut CanonicalIr) {
    if ir
        .judgment_predicates
        .iter()
        .any(|predicate| predicate.id == DSL_PREDICATE_ID)
    {
        return;
    }
    ir.judgment_predicates.push(JudgmentPredicate {
        id: DSL_PREDICATE_ID.to_string(),
        description: "Auto-accept DSL proposals.".to_string(),
    });
}

fn ensure_dsl_tick(ir: &mut CanonicalIr) -> Result<(), ()> {
    if ir.ticks.iter().any(|tick| tick.id == DSL_TICK_ID) {
        return Ok(());
    }
    let graph_id = ir
        .tick_graphs
        .first()
        .map(|graph| graph.id.clone())
        .ok_or(())?;
    ir.ticks.push(Tick {
        id: DSL_TICK_ID.to_string(),
        graph: graph_id,
        input_state: vec![],
        output_deltas: vec![],
    });
    Ok(())
}
