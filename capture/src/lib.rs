
#![feature(rustc_private)]
//! capture crate
//! Single public API: capture::capture(&root)

use anyhow::Result;
use std::path::Path;

pub mod ast_capture;
pub mod unified;

use model::model_ir::Model;
use unified::UnifiedModel;

/// Pipeline entry:
/// Source₀ -> Model₀ (AST-based)
pub fn capture(root: &Path) -> Result<Model> {
    let mut model = ast_capture::capture_ast(root)?;

    // Merge HIR/MIR layer if feature enabled
    #[cfg(feature = "hir")]
    {
        // Use unified rustc backend entrypoint
        if let Ok(()) = capture_rustc::run(root) {
            // If capture_rustc is extended later to return a Model,
            // merge it here. For now, run() performs HIR/MIR capture side-effects.
        }
    }

    Ok(model)
}

/// Unified pipeline entry:
/// AST + (future HIR/MIR) -> UnifiedModel
pub fn capture_unified(root: &Path) -> Result<UnifiedModel> {
    // Step 1: AST capture
    let model0 = ast_capture::capture_ast(root)?;

    let mut unified = UnifiedModel::new();

    // Populate AST layer into unified model
    for m in model0.modules {
        unified.ast.push(unified::AstNode {
            id: unified::UnifiedId {
                crate_name: model0.crate_.name.clone(),
                def_path: m.id,
            },
            span: (0, 0),
            kind: "module".into(),
        });
    }

    // HIR + MIR layers will be added via rustc backend
    #[cfg(feature = "hir")]
    {
        let _ = capture_rustc::run(root);
    }

    Ok(unified)
}
