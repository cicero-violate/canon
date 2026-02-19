#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

use rename::rename::core::project_editor::ProjectEditor;
use rename::rename::structured::FieldMutation;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/rename");
    let mut editor = ProjectEditor::load_with_rustc(project_path)?;

    let renames = [
        (
            "crate::rustc_integration::frontends::rustc::metadata_capture::serialize_generics",
            "serialize_generic_params",
        ),
        (
            "crate::rustc_integration::frontends::rustc::metadata_capture::serialize_predicates",
            "serialize_where_predicates",
        ),
        (
            "crate::rustc_integration::frontends::rustc::metadata_capture::format_param_kind",
            "format_generic_param_kind",
        ),
        (
            "crate::rustc_integration::frontends::rustc::metadata_capture::param_has_default",
            "generic_param_has_default",
        ),
        (
            "crate::rustc_integration::frontends::rustc::metadata_capture::source_file_stats",
            "compute_source_file_stats",
        ),
        (
            "crate::rustc_integration::frontends::rustc::crate_metadata::serialize_dependencies",
            "serialize_crate_dependencies",
        ),
        (
            "crate::rustc_integration::frontends::rustc::mir_capture::serialize_statements",
            "serialize_statement_kinds",
        ),
        (
            "crate::rustc_integration::frontends::rustc::mir_capture::serialize_mir_dump",
            "serialize_mir_body",
        ),
        (
            "crate::rustc_integration::frontends::rustc::mir_capture::call_span",
            "callsite_span",
        ),
        (
            "crate::rustc_integration::frontends::rustc::item_capture::serialize_struct_fields",
            "serialize_struct_field_list",
        ),
    ];


    for (symbol_id, new_name) in renames {
        editor.queue_by_id(symbol_id, FieldMutation::RenameIdent(new_name.to_string()))?;
    }

    let conflicts = editor.validate()?;
    println!("conflicts: {conflicts:?}");

    let report = editor.apply()?;
    println!("touched: {:?}", report.touched_files);

    let preview = editor.preview()?;
    println!("preview: {preview}");

    let written = editor.commit()?;
    println!("written: {:?}", written);

    Ok(())
}
