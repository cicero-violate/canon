//! Unified capture model
//! AST ↔ HIR ↔ MIR linkage layer

use serde::{Deserialize, Serialize};

/// Stable cross-layer key.
/// - AST: path-based synthetic id
/// - HIR: rustc_hir::def_id::DefId (as string)
/// - MIR: rustc_middle::ty::InstanceDef / DefId
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnifiedId {
    pub crate_name: String,
    pub def_path: String, // rustc def_path_str
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AstNode {
    pub id: UnifiedId,
    pub span: (u32, u32),
    pub kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HirNode {
    pub id: UnifiedId,
    pub hir_def_id: String,
    pub kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MirNode {
    pub id: UnifiedId,
    pub basic_blocks: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UnifiedModel {
    pub ast: Vec<AstNode>,
    pub hir: Vec<HirNode>,
    pub mir: Vec<MirNode>,
}

impl UnifiedModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Connect layers by UnifiedId equality
    pub fn verify_links(&self) -> bool {
        use std::collections::HashSet;

        let ast_ids: HashSet<_> = self.ast.iter().map(|n| &n.id).collect();
        let hir_ids: HashSet<_> = self.hir.iter().map(|n| &n.id).collect();
        let mir_ids: HashSet<_> = self.mir.iter().map(|n| &n.id).collect();

        hir_ids.is_subset(&ast_ids) && mir_ids.is_subset(&hir_ids)
    }
}

