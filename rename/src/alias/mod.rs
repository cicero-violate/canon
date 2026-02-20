//! semantic: domain=tooling
//! Alias and re-export modeling for comprehensive rename tracking

mod graph;
mod helpers;
mod types;
mod visibility;

pub use graph::AliasGraph;
pub use types::{
    AliasEdge, EdgeKind, ExposurePath, ImportNode, LeakedSymbol, ResolutionChain, ResolutionStep,
    StepKind, UseKind, VisibilityLeakAnalysis, VisibilityScope,
};
