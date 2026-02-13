use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{
    ids::{ModuleId, ProposalId, TraitFunctionId, TraitId},
    word::Word,
};

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProposalKind {
    #[default]
    Structural,
    FunctionBody,
    SchemaEvolution,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct Proposal {
    pub id: ProposalId,
    #[serde(default)]
    pub kind: ProposalKind,
    pub goal: ProposalGoal,
    pub nodes: Vec<ProposedNode>,
    pub apis: Vec<ProposedApi>,
    pub edges: Vec<ProposedEdge>,
    pub status: ProposalStatus,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ProposalGoal {
    pub id: Word,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ProposedNode {
    pub id: Option<String>,
    pub name: Word,
    pub module: Option<ModuleId>,
    pub kind: ProposedNodeKind,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ProposedApi {
    pub trait_id: TraitId,
    pub functions: Vec<TraitFunctionId>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ProposedEdge {
    pub from: ModuleId,
    pub to: ModuleId,
    pub rationale: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProposedNodeKind {
    Module,
    Struct,
    Trait,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStatus {
    Draft,
    Submitted,
    Accepted,
    Rejected,
}
