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

    map.insert(
        "crate::rename::scope::Scope".to_string(),
        "LexicalScope".to_string(),
    );

    map.insert(
        "crate::rename::scope::ScopedBinder".to_string(),
        "LexicalBinder".to_string(),
    );

    map.insert(
        "crate::rename::core::SymbolTable".to_string(),
        "SymbolIndex".to_string(),
    );

    map.insert(
        "crate::rename::core::OccurrenceEntry".to_string(),
        "SymbolOccurrence".to_string(),
    );

    map.insert(
        "crate::rename::core::OccurrenceVisitor".to_string(),
        "SymbolOccurrenceVisitor".to_string(),
    );

    map.insert(
        "crate::rename::core::TypeContext".to_string(),
        "LocalTypeContext".to_string(),
    );

    map.insert(
        "crate::rename::core::ModuleChange".to_string(),
        "ModuleRenamePlan".to_string(),
    );

    map.insert(
        "crate::rename::core::ModEdit".to_string(),
        "ModuleDeclarationEdit".to_string(),
    );

    map.insert(
        "crate::rename::core::ModEditKind".to_string(),
        "ModuleDeclarationEditKind".to_string(),
    );

    map.insert(
        "crate::rename::core::FlushResult".to_string(),
        "RewriteSummary".to_string(),
    );

    // dry_run = false
    // out_path = None
    apply_rename_with_map(project_path, &map, false, None)?;

    println!("Scoped renames applied successfully.");
    Ok(())
}
