use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::Result;
use syn::visit::Visit;

use crate::alias::VisibilityLeakAnalysis;
use crate::alias::{AliasGraph, VisibilityScope};
use crate::core::collect::{add_file_module_symbol, collect_symbols};
use crate::core::oracle::StructuralEditOracle;
use crate::core::paths::module_path_for_file;
use crate::core::rename::apply_symbol_edits_to_ast;
use crate::core::symbol_id::normalize_symbol_id;
use crate::core::types::{FileRename, SymbolEdit, SymbolIndex, SymbolOccurrence};
use crate::core::use_map::build_use_map;
use crate::module_path::{ModuleMovePlan, ModulePath};
use crate::occurrence::EnhancedOccurrenceVisitor;
use crate::structured::{FieldMutation, NodeOp};
use crate::state::NodeRegistry;

use super::EditConflict;

pub struct PropagationResult {
    pub rewrites: Vec<SymbolEdit>,
    pub conflicts: Vec<EditConflict>,
    pub file_renames: Vec<crate::core::types::FileRename>,
}

pub fn propagate(
    op: &NodeOp,
    symbol_id: &str,
    registry: &NodeRegistry,
    oracle: &dyn StructuralEditOracle,
) -> Result<PropagationResult> {
    match op {
        NodeOp::MutateField { mutation, .. } => match mutation {
            FieldMutation::RenameIdent(new_name) => {
                propagate_rename(symbol_id, new_name, registry, oracle)
            }
            FieldMutation::ChangeVisibility(new_vis) => {
                propagate_visibility(symbol_id, new_vis, registry, oracle)
            }
            FieldMutation::RemoveStructField(field) => {
                propagate_remove_field(symbol_id, field, registry, oracle)
            }
            FieldMutation::RemoveVariant(variant) => {
                propagate_remove_variant(symbol_id, variant, registry, oracle)
            }
            FieldMutation::ReplaceSignature(sig) => {
                propagate_signature(symbol_id, sig, registry, oracle)
            }
            FieldMutation::AddStructField(field) => {
                propagate_add_field(symbol_id, field, registry, oracle)
            }
            FieldMutation::AddVariant(variant) => {
                propagate_add_variant(symbol_id, variant, registry, oracle)
            }
            FieldMutation::AddAttribute(_) | FieldMutation::RemoveAttribute(_) => {
                Ok(PropagationResult {
                    rewrites: Vec::new(),
                    conflicts: Vec::new(),
                    file_renames: Vec::new(),
                })
            }
        },
        NodeOp::DeleteNode { .. } => propagate_delete(symbol_id, oracle),
        NodeOp::ReplaceNode { .. } => propagate_delete(symbol_id, oracle),
        NodeOp::MoveSymbol {
            new_module_path,
            new_crate,
            ..
        } => propagate_move(
            symbol_id,
            new_module_path,
            new_crate.as_deref(),
            registry,
            oracle,
        ),
        NodeOp::InsertBefore { .. } | NodeOp::InsertAfter { .. } | NodeOp::ReorderItems { .. } => {
            Ok(PropagationResult {
                rewrites: Vec::new(),
                conflicts: Vec::new(),
                file_renames: Vec::new(),
            })
        }
    }
}

fn propagate_rename(
    symbol_id: &str,
    new_name: &str,
    registry: &NodeRegistry,
    oracle: &dyn StructuralEditOracle,
) -> Result<PropagationResult> {
    let norm_id = normalize_symbol_id(symbol_id);
    let mut affected_ids: HashSet<String> = oracle
        .impact_of(&norm_id)
        .into_iter()
        .map(|id| normalize_symbol_id(&id))
        .collect();
    affected_ids.insert(norm_id.clone());

    let (_symbol_table, occurrences, _alias_graph) = build_symbol_index_and_occurrences(registry)?;
    let mut rewrites = Vec::new();

    if affected_ids.len() == 1 && oracle.impact_of(&norm_id).is_empty() {
        // Fallback: syn-only occurrences
        for occ in occurrences.iter().filter(|o| o.id == norm_id) {
            rewrites.push(occurrence_to_edit(occ, new_name));
        }
    } else {
        for occ in &occurrences {
            if affected_ids.contains(&occ.id) {
                rewrites.push(occurrence_to_edit(occ, new_name));
            }
        }
    }

    let conflicts = oracle
        .cross_crate_users(&norm_id)
        .into_iter()
        .map(|id| EditConflict {
            symbol_id: normalize_symbol_id(&id),
            reason: "cross-crate rename requires manual update".to_string(),
        })
        .collect();

    Ok(PropagationResult {
        rewrites,
        conflicts,
        file_renames: Vec::new(),
    })
}

fn propagate_delete(
    symbol_id: &str,
    oracle: &dyn StructuralEditOracle,
) -> Result<PropagationResult> {
    let norm_id = normalize_symbol_id(symbol_id);
    let mut conflicts = Vec::new();

    for id in oracle.impact_of(&norm_id) {
        conflicts.push(EditConflict {
            symbol_id: normalize_symbol_id(&id),
            reason: "deleted symbol still used".to_string(),
        });
    }
    for id in oracle.cross_crate_users(&norm_id) {
        conflicts.push(EditConflict {
            symbol_id: normalize_symbol_id(&id),
            reason: "deleted symbol used across crates".to_string(),
        });
    }

    Ok(PropagationResult {
        rewrites: Vec::new(),
        conflicts,
        file_renames: Vec::new(),
    })
}

fn propagate_move(
    symbol_id: &str,
    new_module_path: &str,
    new_crate: Option<&str>,
    registry: &NodeRegistry,
    oracle: &dyn StructuralEditOracle,
) -> Result<PropagationResult> {
    let norm_id = normalize_symbol_id(symbol_id);
    let mut conflicts = Vec::new();
    let mut file_renames = Vec::new();

    if let Some(new_crate) = new_crate {
        conflicts.push(EditConflict {
            symbol_id: norm_id.clone(),
            reason: format!("cross-crate move requires Cargo.toml update ({new_crate})"),
        });
        for id in oracle.impact_of(&norm_id) {
            conflicts.push(EditConflict {
                symbol_id: normalize_symbol_id(&id),
                reason: "cross-crate move requires manual update".to_string(),
            });
        }
        for id in oracle.cross_crate_users(&norm_id) {
            conflicts.push(EditConflict {
                symbol_id: normalize_symbol_id(&id),
                reason: "cross-crate move requires manual update".to_string(),
            });
        }
    } else {
        for id in oracle.cross_crate_users(&norm_id) {
            conflicts.push(EditConflict {
                symbol_id: normalize_symbol_id(&id),
                reason: "move affects external user".to_string(),
            });
        }
    }

    let Some(handle) = registry.handles.get(&norm_id) else {
        return Ok(PropagationResult {
            rewrites: Vec::new(),
            conflicts,
            file_renames,
        });
    };

    let project_root = find_project_root(registry)?;
    let old_module_path = normalize_symbol_id(&module_path_for_file(&project_root, &handle.file));
    let from_path = ModulePath::from_string(&old_module_path);
    let new_module_path = normalize_symbol_id(new_module_path);
    let to_path = ModulePath::from_string(&new_module_path);
    let plan = ModuleMovePlan::new(
        from_path.clone(),
        to_path.clone(),
        handle.file.clone(),
        &project_root,
    )?;

    file_renames.push(FileRename {
        from: handle.file.to_string_lossy().to_string(),
        to: plan.to_file.to_string_lossy().to_string(),
        is_directory_move: plan.create_directory,
        old_module_id: from_path.to_string(),
        new_module_id: to_path.to_string(),
    });

    let (_symbol_table, occurrences, alias_graph) = build_symbol_index_and_occurrences(registry)?;
    let mut rewrites = Vec::new();
    let mut seen = HashSet::new();

    let symbol_name = norm_id.rsplit("::").next().unwrap_or(&norm_id).to_string();
    let new_module_leaf = new_module_path
        .rsplit("::")
        .next()
        .unwrap_or(&new_module_path)
        .to_string();

    for occ in occurrences
        .iter()
        .filter(|occ| occ.id == norm_id && occ.kind == "use")
    {
        push_unique_edit(&mut rewrites, &mut seen, occ, &symbol_name);
    }

    if let Some(old_module) = parent_module_path(&norm_id) {
        if old_module != new_module_path {
            for occ in occurrences
                .iter()
                .filter(|occ| occ.id == old_module && occ.kind == "use_path")
            {
                push_unique_edit(&mut rewrites, &mut seen, occ, &new_module_leaf);
            }
        }
    }

    let mut importer_files: HashSet<String> = HashSet::new();
    for node in alias_graph.get_importers(&norm_id) {
        importer_files.insert(node.file.clone());
    }
    for chain in alias_graph.find_reexport_chains(&norm_id) {
        for node in chain {
            importer_files.insert(node.file.clone());
        }
    }

    for file in importer_files {
        for occ in occurrences
            .iter()
            .filter(|occ| occ.file == file && occ.id == norm_id && occ.kind == "use")
        {
            push_unique_edit(&mut rewrites, &mut seen, occ, &symbol_name);
        }
    }

    Ok(PropagationResult {
        rewrites,
        conflicts,
        file_renames,
    })
}

fn parent_module_path(path: &str) -> Option<String> {
    let mut parts: Vec<&str> = path.split("::").collect();
    if parts.len() <= 1 {
        return None;
    }
    parts.pop();
    Some(parts.join("::"))
}

fn push_unique_edit(
    rewrites: &mut Vec<SymbolEdit>,
    seen: &mut HashSet<String>,
    occ: &SymbolOccurrence,
    new_name: &str,
) {
    let key = format!(
        "{}:{}:{}:{}:{}:{}",
        occ.file,
        occ.span.start.line,
        occ.span.start.column,
        occ.span.end.line,
        occ.span.end.column,
        new_name
    );
    if seen.insert(key) {
        rewrites.push(occurrence_to_edit(occ, new_name));
    }
}

fn propagate_remove_field(
    symbol_id: &str,
    field_name: &str,
    _registry: &NodeRegistry,
    oracle: &dyn StructuralEditOracle,
) -> Result<PropagationResult> {
    let norm_id = normalize_symbol_id(symbol_id);
    let mut conflicts = Vec::new();
    for id in oracle.impact_of(&norm_id) {
        conflicts.push(EditConflict {
            symbol_id: normalize_symbol_id(&id),
            reason: format!("removed field '{field_name}' is still accessed"),
        });
    }
    Ok(PropagationResult {
        rewrites: Vec::new(),
        conflicts,
        file_renames: Vec::new(),
    })
}

fn propagate_remove_variant(
    symbol_id: &str,
    variant_name: &str,
    _registry: &NodeRegistry,
    oracle: &dyn StructuralEditOracle,
) -> Result<PropagationResult> {
    let norm_id = normalize_symbol_id(symbol_id);
    let mut conflicts = Vec::new();
    for id in oracle.impact_of(&norm_id) {
        conflicts.push(EditConflict {
            symbol_id: normalize_symbol_id(&id),
            reason: format!("removed variant '{variant_name}' is still matched"),
        });
    }
    Ok(PropagationResult {
        rewrites: Vec::new(),
        conflicts,
        file_renames: Vec::new(),
    })
}

fn propagate_visibility(
    symbol_id: &str,
    _new_vis: &syn::Visibility,
    registry: &NodeRegistry,
    oracle: &dyn StructuralEditOracle,
) -> Result<PropagationResult> {
    let norm_id = normalize_symbol_id(symbol_id);
    let mut conflicts: Vec<EditConflict> = oracle
        .cross_crate_users(&norm_id)
        .into_iter()
        .map(|id| EditConflict {
            symbol_id: normalize_symbol_id(&id),
            reason: "visibility change affects external user".to_string(),
        })
        .collect();

    let (_symbol_table, _occurrences, alias_graph) = build_symbol_index_and_occurrences(registry)?;
    let visibility_map = build_visibility_map(registry, &alias_graph)?;
    let analysis = alias_graph.analyze_visibility_leaks(&visibility_map);
    conflicts.extend(leak_conflicts(&analysis));

    Ok(PropagationResult {
        rewrites: Vec::new(),
        conflicts,
        file_renames: Vec::new(),
    })
}

fn propagate_signature(
    symbol_id: &str,
    new_sig: &syn::Signature,
    _registry: &NodeRegistry,
    oracle: &dyn StructuralEditOracle,
) -> Result<PropagationResult> {
    let norm_id = normalize_symbol_id(symbol_id);
    let mut conflicts = Vec::new();
    for id in oracle.impact_of(&norm_id) {
        if !oracle.satisfies_bounds(&id, new_sig) {
            conflicts.push(EditConflict {
                symbol_id: normalize_symbol_id(&id),
                reason: "call site incompatible with new signature".to_string(),
            });
        }
    }
    Ok(PropagationResult {
        rewrites: Vec::new(),
        conflicts,
        file_renames: Vec::new(),
    })
}

fn propagate_add_field(
    symbol_id: &str,
    field: &syn::Field,
    _registry: &NodeRegistry,
    oracle: &dyn StructuralEditOracle,
) -> Result<PropagationResult> {
    let norm_id = normalize_symbol_id(symbol_id);
    let name = field
        .ident
        .as_ref()
        .map(|i| i.to_string())
        .unwrap_or_else(|| "<unnamed>".to_string());
    let mut conflicts = Vec::new();
    for id in oracle.impact_of(&norm_id) {
        conflicts.push(EditConflict {
            symbol_id: normalize_symbol_id(&id),
            reason: format!("added field '{name}' requires constructor update"),
        });
    }
    Ok(PropagationResult {
        rewrites: Vec::new(),
        conflicts,
        file_renames: Vec::new(),
    })
}

fn propagate_add_variant(
    symbol_id: &str,
    variant: &syn::Variant,
    _registry: &NodeRegistry,
    oracle: &dyn StructuralEditOracle,
) -> Result<PropagationResult> {
    let norm_id = normalize_symbol_id(symbol_id);
    let name = variant.ident.to_string();
    let mut conflicts = Vec::new();
    for id in oracle.impact_of(&norm_id) {
        conflicts.push(EditConflict {
            symbol_id: normalize_symbol_id(&id),
            reason: format!("added variant '{name}' may require match update"),
        });
    }
    Ok(PropagationResult {
        rewrites: Vec::new(),
        conflicts,
        file_renames: Vec::new(),
    })
}

fn occurrence_to_edit(occ: &SymbolOccurrence, new_name: &str) -> SymbolEdit {
    SymbolEdit {
        id: occ.id.clone(),
        file: occ.file.clone(),
        kind: occ.kind.clone(),
        start: occ.span.start.clone(),
        end: occ.span.end.clone(),
        new_name: new_name.to_string(),
    }
}

fn build_symbol_index_and_occurrences(
    registry: &NodeRegistry,
) -> Result<(SymbolIndex, Vec<SymbolOccurrence>, AliasGraph)> {
    let project_root = find_project_root(registry)?;
    let mut symbols = Vec::new();
    let mut symbol_set: HashSet<String> = HashSet::new();
    let mut symbol_table = SymbolIndex::default();
    let mut alias_graph = AliasGraph::new();

    for (file, ast) in &registry.asts {
        let module_path = normalize_symbol_id(&module_path_for_file(&project_root, file));
        add_file_module_symbol(
            &module_path,
            file,
            &mut symbol_table,
            &mut symbols,
            &mut symbol_set,
        );
        let file_alias_graph = collect_symbols(
            ast,
            &module_path,
            file,
            &mut symbol_table,
            &mut symbols,
            &mut symbol_set,
        );
        for node in file_alias_graph.all_nodes() {
            alias_graph.add_use_node(node.clone());
        }
    }

    alias_graph.build_edges();

    let mut occurrences = Vec::new();
    for (file, ast) in &registry.asts {
        let module_path = normalize_symbol_id(&module_path_for_file(&project_root, file));
        let use_map = build_use_map(ast, &module_path);
        let mut visitor = EnhancedOccurrenceVisitor::new(
            &module_path,
            file,
            &symbol_table,
            &use_map,
            &mut occurrences,
        );
        visitor.visit_file(ast);
    }

    Ok((symbol_table, occurrences, alias_graph))
}

fn build_visibility_map(
    registry: &NodeRegistry,
    _alias_graph: &AliasGraph,
) -> Result<HashMap<String, VisibilityScope>> {
    let project_root = find_project_root(registry)?;
    let mut symbol_table = SymbolIndex::default();
    let mut symbols = Vec::new();
    let mut symbol_set: HashSet<String> = HashSet::new();

    for (file, ast) in &registry.asts {
        let module_path = normalize_symbol_id(&module_path_for_file(&project_root, file));
        add_file_module_symbol(
            &module_path,
            file,
            &mut symbol_table,
            &mut symbols,
            &mut symbol_set,
        );
        let _ = collect_symbols(
            ast,
            &module_path,
            file,
            &mut symbol_table,
            &mut symbols,
            &mut symbol_set,
        );
    }

    let mut visibility_map = HashMap::new();
    for (id, record) in &symbol_table.symbols {
        let vis = if record.kind == "pub use" {
            VisibilityScope::Public
        } else {
            VisibilityScope::Private
        };
        visibility_map.insert(id.clone(), vis);
    }
    Ok(visibility_map)
}

fn leak_conflicts(analysis: &VisibilityLeakAnalysis) -> Vec<EditConflict> {
    analysis
        .leaked_private_symbols
        .iter()
        .map(|leak| EditConflict {
            symbol_id: leak.symbol_id.clone(),
            reason: format!("visibility leak to {}", leak.leaked_to),
        })
        .collect()
}

fn find_project_root(registry: &NodeRegistry) -> Result<PathBuf> {
    let file = registry
        .asts
        .keys()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no ASTs loaded"))?;
    let mut current = file
        .parent()
        .unwrap_or_else(|| Path::new("/"))
        .to_path_buf();
    loop {
        if current.join("Cargo.toml").exists() {
            return Ok(current);
        }
        if !current.pop() {
            break;
        }
    }
    Err(anyhow::anyhow!("Cargo.toml not found for project"))
}

pub fn apply_rewrites(
    registry: &mut NodeRegistry,
    rewrites: &[SymbolEdit],
) -> Result<HashSet<PathBuf>> {
    let mut by_file: HashMap<String, Vec<SymbolEdit>> = HashMap::new();
    for edit in rewrites {
        by_file
            .entry(edit.file.clone())
            .or_default()
            .push(edit.clone());
    }

    let mut touched = HashSet::new();
    for (file, edits) in by_file {
        let path = PathBuf::from(&file);
        let Some(ast) = registry.asts.get_mut(&path) else {
            continue;
        };
        if apply_symbol_edits_to_ast(ast, &edits)? {
            touched.insert(path);
        }
    }
    Ok(touched)
}
