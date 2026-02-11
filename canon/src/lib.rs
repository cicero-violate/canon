pub mod decision;
pub mod evolution;
pub mod gpu;
pub mod ir;
pub mod materialize;
pub mod memory;
pub mod observe;
pub mod patch_protocol;
pub mod proof;
pub mod proof_object;
mod proposal;
pub mod runtime;
pub mod schema;
pub mod validate;

pub use decision::{
    AcceptProposalError, AutoAcceptDslError, ProposalAcceptance, ProposalAcceptanceInput,
    accept_proposal, auto_accept_dsl_proposal,
};
pub use evolution::{EvolutionError, apply_deltas};
pub use gpu::{
    codegen::{GpuProgram, flatten_ports, generate_shader},
    dispatch::{GpuExecutor, GpuExecutorError},
};
pub use ir::{CanonicalIr, PipelineStage};
pub use materialize::{FileEntry, FileTree, materialize, write_file_tree};
pub use observe::execution_events_to_observe_deltas;
pub use patch_protocol::{
    ApprovedPatchRegistry, PatchApplier, PatchDecision, PatchError, PatchGate, PatchMetadata,
    PatchProposal, PatchQueue, VerifiedPatch,
};
pub use proof::smt_bridge::{
    SmtCertificate, SmtError, attach_function_proofs, verify_function_postconditions,
};
pub use proof_object::{
    ProofArtifact as CanonProofArtifact, ProofObject, ProofResult, evaluate_proof_object,
};
pub use proposal::{DslProposalArtifacts, DslProposalError, create_proposal_from_dsl};
pub use schema::generate_schema;
pub use validate::{CanonRule, ValidationErrors, Violation, validate_ir};
