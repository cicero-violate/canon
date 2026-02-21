use std::collections::{HashMap, HashSet};

use super::helpers::extract_module_from_path;
use super::types::{ExposurePath, LeakedSymbol, UseKind, VisibilityLeakAnalysis, VisibilityScope};
use super::AliasGraph;

impl AliasGraph {
    pub fn analyze_visibility_leaks(&self, symbols: &HashMap<String, VisibilityScope>) -> VisibilityLeakAnalysis {
        let mut analysis = VisibilityLeakAnalysis { public_symbols: HashMap::new(), restricted_symbols: HashMap::new(), leaked_private_symbols: Vec::new() };

        for (symbol_id, visibility) in symbols {
            let module = extract_module_from_path(symbol_id);
            let name = symbol_id.rsplit("::").next().unwrap_or(symbol_id);

            match visibility {
                VisibilityScope::Public => {
                    let chains = self.find_reexport_chains(symbol_id);
                    let has_chains = !chains.is_empty();
                    for chain in chains {
                        let reexport_modules: Vec<String> = chain.iter().map(|node| node.module_path.clone()).collect();
                        let exposure = ExposurePath { origin_module: module.clone(), reexport_chain: reexport_modules, visibility: VisibilityScope::Public };
                        analysis.public_symbols.entry(module.clone()).or_insert_with(Vec::new).push((name.to_string(), exposure));
                    }
                    if !has_chains {
                        let exposure = ExposurePath { origin_module: module.clone(), reexport_chain: Vec::new(), visibility: VisibilityScope::Public };
                        analysis.public_symbols.entry(module.clone()).or_insert_with(Vec::new).push((name.to_string(), exposure));
                    }
                }
                VisibilityScope::Private | VisibilityScope::Crate | VisibilityScope::Super | VisibilityScope::Restricted(_) => {
                    analysis.restricted_symbols.entry(module.clone()).or_insert_with(Vec::new).push((name.to_string(), visibility.clone()));
                    self.detect_visibility_leak(symbol_id, visibility, &module, &mut analysis.leaked_private_symbols);
                }
            }
        }

        analysis
    }

    fn detect_visibility_leak(&self, symbol_id: &str, original_visibility: &VisibilityScope, origin_module: &str, leaked: &mut Vec<LeakedSymbol>) {
        let importers = self.get_importers(symbol_id);
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(symbol_id.to_string());

        for importer in importers {
            if matches!(importer.kind, UseKind::ReExport | UseKind::ReExportAliased) && importer.visibility == VisibilityScope::Public {
                if !self.is_visible(origin_module, &importer.module_path, original_visibility) {
                    leaked.push(LeakedSymbol {
                        symbol_id: symbol_id.to_string(),
                        original_visibility: original_visibility.clone(),
                        leaked_to: importer.module_path.clone(),
                        leak_chain: vec![importer.id.clone()],
                    });
                }
                let reexport_path = format!("{}::{}", importer.module_path, importer.local_name);
                self.detect_visibility_leak_recursive(&reexport_path, original_visibility, origin_module, vec![importer.id.clone()], leaked, &mut visited);
            }
        }
    }

    fn detect_visibility_leak_recursive(&self, current_path: &str, original_visibility: &VisibilityScope, origin_module: &str, chain: Vec<String>, leaked: &mut Vec<LeakedSymbol>, visited: &mut HashSet<String>) {
        if !visited.insert(current_path.to_string()) {
            return;
        }
        let importers = self.get_importers(current_path);
        for importer in importers {
            if matches!(importer.kind, UseKind::ReExport | UseKind::ReExportAliased) && importer.visibility == VisibilityScope::Public {
                let mut new_chain = chain.clone();
                new_chain.push(importer.id.clone());
                if !self.is_visible(origin_module, &importer.module_path, original_visibility) {
                    leaked.push(LeakedSymbol {
                        symbol_id: current_path.to_string(),
                        original_visibility: original_visibility.clone(),
                        leaked_to: importer.module_path.clone(),
                        leak_chain: new_chain.clone(),
                    });
                }
                let reexport_path = format!("{}::{}", importer.module_path, importer.local_name);
                self.detect_visibility_leak_recursive(&reexport_path, original_visibility, origin_module, new_chain, leaked, visited);
            }
        }
    }
}
