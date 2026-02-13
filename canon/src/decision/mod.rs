pub mod accept;
mod auto_dot;
mod auto_dsl;
mod auto_fn_ast;
mod bootstrap;

pub use accept::{
    AcceptProposalError, ProposalAcceptance, ProposalAcceptanceInput, accept_proposal,
};
pub use auto_dot::{AutoAcceptDotError, auto_accept_dot_proposal};
pub use auto_dsl::{AutoAcceptDslError, auto_accept_dsl_proposal};
pub use auto_fn_ast::{AutoAcceptFnAstError, auto_accept_fn_ast};

pub(crate) const DSL_PROOF_ID: &str = "proof.dsl.bootstrap";
pub(crate) const DSL_PREDICATE_ID: &str = "predicate.dsl.autoaccept";
pub(crate) const DSL_TICK_ID: &str = "tick.bootstrap";
