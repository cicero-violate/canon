#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

use rename::apply_rename_with_map;
use std::collections::HashMap;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Target project root (this crate)
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/rename");

    // IMPORTANT:
    // Always use fully-qualified paths to avoid global identifier corruption.
    // This prevents doc strings, comments, and unrelated modules from breaking.
    let mut map = HashMap::new();

    // Normalize unclear / overloaded names

    map.insert("crate::rename::scope::LexicalScope".to_string(), "ScopeFrame".to_string());

    map.insert("crate::rename::scope::LexicalBinder".to_string(), "ScopeBinder".to_string());

    map.insert("crate::rename::structured::StructuredEditConfig".to_string(), "StructuredEditOptions".to_string());

    map.insert("crate::rename::structured::PassOrchestrator".to_string(), "StructuredPassRunner".to_string());

    map.insert("crate::rename::structured::UseTreePass".to_string(), "UsePathRewritePass".to_string());

    map.insert("crate::rename::structured::StructuredPass".to_string(), "StructuredRewritePass".to_string());

    map.insert("crate::rename::core::StructuredEditTracker".to_string(), "StructuredRewriteTracker".to_string());

    map.insert("crate::rename::core::SymbolEdit".to_string(), "SymbolRewriteEdit".to_string());

    map.insert("crate::rename::core::rename::SpanKey".to_string(), "SpanRangeKey".to_string());

    map.insert("crate::rename::core::rename::SpanRenamer".to_string(), "SpanRangeRenamer".to_string());

    map.insert("crate::rustc_integration::frontends::rustc::types::capture_types_from_function".to_string(), "capture_function_types".to_string());

    map.insert("crate::rustc_integration::frontends::rustc::types::collect_types_from_ty".to_string(), "collect_type_dependencies".to_string());

    map.insert("crate::rustc_integration::frontends::rustc::hir_bodies::serialize_hir_body".to_string(), "encode_hir_body_json".to_string());

    map.insert("crate::rustc_integration::frontends::rustc::collector::extract_cfg_flags".to_string(), "parse_cfg_flags".to_string());

    map.insert("crate::rustc_integration::frontends::rustc::collector::determine_sysroot".to_string(), "resolve_sysroot".to_string());

    // dry_run = true
    // out_path = None
    apply_rename_with_map(project_path, &map, true, None)?;

    // dry_run = false
    // out_path = None
    apply_rename_with_map(project_path, &map, false, None)?;

    println!("Scoped renames applied successfully.");
    Ok(())
}
