use std::collections::HashMap;

use super::helpers::extract_module_from_path;
use super::types::{ExposurePath, LeakedSymbol, UseKind, VisibilityLeakAnalysis, VisibilityScope};
use super::AliasGraph;

impl AliasGraph {
    /// Analyze visibility leaks in the graph
    pub fn analyze_visibility_leaks(
        &self,
        symbols: &HashMap<String, VisibilityScope>,
    ) -> VisibilityLeakAnalysis {
        let mut analysis = VisibilityLeakAnalysis {
            public_symbols: HashMap::new(),
            restricted_symbols: HashMap::new(),
            leaked_private_symbols: Vec::new(),
        };

        // Analyze each symbol
        for (symbol_id, visibility) in symbols {
            let module = extract_module_from_path(symbol_id);
            let name = symbol_id.rsplit("::").next().unwrap_or(symbol_id);

            match visibility {
                VisibilityScope::Public => {
                    // Find all re-export chains for this symbol
                    let chains = self.find_reexport_chains(symbol_id);
                    let has_chains = !chains.is_empty();

                    for chain in chains {
                        let reexport_modules: Vec<String> =
                            chain.iter().map(|node| node.module_path.clone()).collect();

                        let exposure = ExposurePath {
                            origin_module: module.clone(),
                            reexport_chain: reexport_modules,
                            visibility: VisibilityScope::Public,
                        };

                        analysis
                            .public_symbols
                            .entry(module.clone())
                            .or_insert_with(Vec::new)
                            .push((name.to_string(), exposure));
                    }

                    // Also add direct exposure if no re-exports
                    if !has_chains {
                        let exposure = ExposurePath {
                            origin_module: module.clone(),
                            reexport_chain: Vec::new(),
                            visibility: VisibilityScope::Public,
                        };

                        analysis
                            .public_symbols
                            .entry(module.clone())
                            .or_insert_with(Vec::new)
                            .push((name.to_string(), exposure));
                    }
                }
                VisibilityScope::Private
                | VisibilityScope::Crate
                | VisibilityScope::Super
                | VisibilityScope::Restricted(_) => {
                    // Track restricted symbols
                    analysis
                        .restricted_symbols
                        .entry(module.clone())
                        .or_insert_with(Vec::new)
                        .push((name.to_string(), visibility.clone()));

                    // Check if this symbol is leaked through public re-exports
                    self.detect_visibility_leak(
                        symbol_id,
                        visibility,
                        &module,
                        &mut analysis.leaked_private_symbols,
                    );
                }
            }
        }

        analysis
    }

    /// Detect if a restricted symbol is leaked through re-exports
    fn detect_visibility_leak(
        &self,
        symbol_id: &str,
        original_visibility: &VisibilityScope,
        origin_module: &str,
        leaked: &mut Vec<LeakedSymbol>,
    ) {
        let importers = self.get_importers(symbol_id);

        for importer in importers {
            // Check if this is a public re-export
            if matches!(importer.kind, UseKind::ReExport | UseKind::ReExportAliased)
                && importer.visibility == VisibilityScope::Public
            {
                // This is a leak if the original visibility doesn't allow it
                if !self.is_visible(origin_module, &importer.module_path, original_visibility) {
                    leaked.push(LeakedSymbol {
                        symbol_id: symbol_id.to_string(),
                        original_visibility: original_visibility.clone(),
                        leaked_to: importer.module_path.clone(),
                        leak_chain: vec![importer.id.clone()],
                    });
                }

                // Recursively check if this re-export is further leaked
                let reexport_path = format!("{}::{}", importer.module_path, importer.local_name);
                self.detect_visibility_leak_recursive(
                    &reexport_path,
                    original_visibility,
                    origin_module,
                    vec![importer.id.clone()],
                    leaked,
                );
            }
        }
    }

    fn detect_visibility_leak_recursive(
        &self,
        current_path: &str,
        original_visibility: &VisibilityScope,
        origin_module: &str,
        chain: Vec<String>,
        leaked: &mut Vec<LeakedSymbol>,
    ) {
        let importers = self.get_importers(current_path);

        for importer in importers {
            if matches!(importer.kind, UseKind::ReExport | UseKind::ReExportAliased)
                && importer.visibility == VisibilityScope::Public
            {
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
                self.detect_visibility_leak_recursive(
                    &reexport_path,
                    original_visibility,
                    origin_module,
                    new_chain,
                    leaked,
                );
            }
        }
    }
}
