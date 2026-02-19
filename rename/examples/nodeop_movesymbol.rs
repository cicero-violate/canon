#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

use rename::rename::core::project_editor::ProjectEditor;
use rename::rename::structured::NodeOp;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dry_run = std::env::args().any(|arg| arg == "--dry-run");
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/rename");
    let mut editor = ProjectEditor::load_with_rustc(project_path)?;

    let moves = [
        (
            "crate::rustc_integration::frontends::rustc::context::FrontendMetadata",
            "crate::rustc_integration::frontends::rustc::frontend_context",
        ),
        (
            "crate::rustc_integration::frontends::rustc::collector::RustcFrontend",
            "crate::rustc_integration::frontends::rustc::frontend_driver",
        ),
        (
            "crate::rustc_integration::frontends::rustc::hir_bodies::encode_hir_body_json",
            "crate::rustc_integration::frontends::rustc::hir_dump",
        ),
        (
            "crate::rustc_integration::frontends::rustc::items::capture_adt",
            "crate::rustc_integration::frontends::rustc::item_capture",
        ),
        (
            "crate::rustc_integration::frontends::rustc::metadata::apply_common_metadata",
            "crate::rustc_integration::frontends::rustc::metadata_capture",
        ),
        (
            "crate::rustc_integration::frontends::rustc::mir::capture_function",
            "crate::rustc_integration::frontends::rustc::mir_capture",
        ),
        (
            "crate::rustc_integration::frontends::rustc::nodes::ensure_node",
            "crate::rustc_integration::frontends::rustc::node_builder",
        ),
        (
            "crate::rustc_integration::frontends::rustc::traits::capture_trait",
            "crate::rustc_integration::frontends::rustc::trait_capture",
        ),
        (
            "crate::rustc_integration::frontends::rustc::types::capture_function_types",
            "crate::rustc_integration::frontends::rustc::type_capture",
        ),
        (
            "crate::rustc_integration::frontends::rustc::crate_meta::capture_crate_metadata",
            "crate::rustc_integration::frontends::rustc::crate_metadata",
        ),
    ];

    for (symbol_id, new_module_path) in moves {
        let handle = editor
            .registry
            .handles
            .get(symbol_id)
            .cloned()
            .ok_or_else(|| format!("symbol handle not found: {symbol_id}"))?;

        editor.queue(
            symbol_id,
            NodeOp::MoveSymbol {
                handle,
                new_module_path: new_module_path.to_string(),
                new_crate: None,
            },
        )?;
    }

    let conflicts = editor.validate()?;
    println!("conflicts: {conflicts:?}");

    let report = editor.apply()?;
    println!("touched: {:?}", report.touched_files);
    println!("file moves: {:?}", report.file_moves);

    let preview = editor.preview()?;
    println!("preview:\n{preview}");

    if dry_run {
        println!("dry-run: skipping commit()");
    } else {
        let written = editor.commit()?;
        println!("written: {:?}", written);
    }

    Ok(())
}
