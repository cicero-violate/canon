pub mod alias;

pub mod api;

pub mod attributes;

pub mod compiler_capture;

pub mod core;

pub mod fs;

pub mod macros;

pub mod model;

pub mod module_path;

pub mod occurrence;

pub mod pattern;

pub mod resolve;

pub mod scope;

pub mod state;

pub mod structured;

impl From<&Visibility> for VisibilityScope {
    fn from(vis: &Visibility) -> Self {
        match vis {
            Visibility::Public(_) => VisibilityScope::Public,
            Visibility::Restricted(restricted) => {
                if restricted.path.is_ident("crate") {
                    VisibilityScope::Crate
                } else if restricted.path.is_ident("super") {
                    VisibilityScope::Super
                } else if restricted.path.is_ident("self") {
                    VisibilityScope::Private
                } else {
                    VisibilityScope::Restricted(
                        restricted
                            .path
                            .segments
                            .iter()
                            .map(|s| s.ident.to_string())
                            .collect::<Vec<_>>()
                            .join("::"),
                    )
                }
            }
            Visibility::Inherited => VisibilityScope::Private,
        }
    }
}


impl Default for QueryRequest {
    fn default() -> Self {
        Self {
            kinds: Vec::new(),
            module_prefix: None,
            name_contains: None,
        }
    }
}


impl Default for UpsertRequest {
    fn default() -> Self {
        Self {
            edits: Vec::new(),
            node_ops: Vec::new(),
            format: true,
        }
    }
}


impl StructuralEditOracle for GraphSnapshotOracle {
    fn impact_of(&self, symbol_id: &str) -> Vec<String> {
        let symbol_id = normalize_symbol_id(symbol_id);
        let Some(external_id) = self.id_by_key.get(&symbol_id).cloned() else {
            return Vec::new();
        };
        let mut snapshot = self.snapshot.clone();
        let levels = snapshot.bfs_gpu(external_id);
        levels
            .iter()
            .enumerate()
            .filter_map(|(idx, level)| {
                if *level < 0 {
                    return None;
                }
                let key = self.key_by_index.get(idx)?;
                if key == &symbol_id { None } else { Some(key.clone()) }
            })
            .collect()
    }
    fn satisfies_bounds(&self, id: &str, new_sig: &Signature) -> bool {
        let id = normalize_symbol_id(id);
        if let Some(sig) = self.signature_by_key.get(&id) {
            let new_sig = quote::quote!(# new_sig).to_string();
            return sig == &new_sig;
        }
        true
    }
    fn is_macro_generated(&self, symbol_id: &str) -> bool {
        let symbol_id = normalize_symbol_id(symbol_id);
        self.macro_generated.contains(&symbol_id)
    }
    fn cross_crate_users(&self, symbol_id: &str) -> Vec<String> {
        let symbol_id = normalize_symbol_id(symbol_id);
        let Some(symbol_crate) = self.crate_by_key.get(&symbol_id) else {
            return Vec::new();
        };
        let Some(external_id) = self.id_by_key.get(&symbol_id).cloned() else {
            return Vec::new();
        };
        let mut snapshot = self.snapshot.clone();
        let levels = snapshot.bfs_gpu(external_id);
        levels
            .iter()
            .enumerate()
            .filter_map(|(idx, level)| {
                if *level < 0 {
                    return None;
                }
                let key = self.key_by_index.get(idx)?;
                if key == &symbol_id {
                    return None;
                }
                let other_crate = self.crate_by_key.get(key)?;
                if other_crate != symbol_crate { Some(key.clone()) } else { None }
            })
            .collect()
    }
}


impl StructuralEditOracle for NullOracle {
    fn impact_of(&self, _symbol_id: &str) -> Vec<String> {
        Vec::new()
    }
    fn satisfies_bounds(&self, _id: &str, _new_sig: &Signature) -> bool {
        true
    }
    fn is_macro_generated(&self, _symbol_id: &str) -> bool {
        false
    }
    fn cross_crate_users(&self, _symbol_id: &str) -> Vec<String> {
        Vec::new()
    }
}


impl VisitMut for CanonicalRewriteVisitor {
    fn visit_path_mut(&mut self, node: &mut syn::Path) {
        self.rewrite_path(node);
        syn::visit_mut::visit_path_mut(self, node);
    }
    fn visit_item_use_mut(&mut self, node: &mut syn::ItemUse) {
        self.rewrite_use_tree(&mut node.tree, &Vec::new());
        syn::visit_mut::visit_item_use_mut(self, node);
    }
    fn visit_macro_mut(&mut self, node: &mut syn::Macro) {
        self.rewrite_path(&mut node.path);
        syn::visit_mut::visit_macro_mut(self, node);
    }
}


impl VisitMut for SpanRangeRenamer {
    fn visit_ident_mut(&mut self, ident: &mut syn::Ident) {
        let key = SpanRangeKey::from_span(ident.span());
        if let Some(new_name) = self.map.get(&key) {
            if ident.to_string() != *new_name {
                *ident = syn::Ident::new(new_name, ident.span());
                self.changed = true;
            }
        }
        syn::visit_mut::visit_ident_mut(self, ident);
    }
    fn visit_macro_mut(&mut self, mac: &mut syn::Macro) {
        mac.tokens = rewrite_token_stream(
            mac.tokens.clone(),
            &self.map,
            &mut self.changed,
        );
        syn::visit_mut::visit_macro_mut(self, mac);
    }
}


impl std::error::Error for GraphDeltaError {}


impl VisitMut for AttributeRewriteVisitor {
    fn visit_attribute_mut(&mut self, attr: &mut syn::Attribute) {
        self.process_attribute(attr);
    }
}


impl Default for StructuredPassRunner {
    fn default() -> Self {
        Self::new()
    }
}


impl<'a> VisitMut for UseAstRewriter<'a> {
    fn visit_item_use_mut(&mut self, node: &mut syn::ItemUse) {
        let mut current_path = Vec::new();
        if node.leading_colon.is_some() {
            current_path.push("crate".to_string());
        }
        rewrite_use_tree_mut(
            &mut node.tree,
            self.updates,
            &mut self.changed,
            &mut current_path,
            self.resolver,
        );
        syn::visit_mut::visit_item_use_mut(self, node);
    }
}


impl AliasGraph {
    pub fn analyze_visibility_leaks(
        &self,
        symbols: &HashMap<String, VisibilityScope>,
    ) -> VisibilityLeakAnalysis {
        let mut analysis = VisibilityLeakAnalysis {
            public_symbols: HashMap::new(),
            restricted_symbols: HashMap::new(),
            leaked_private_symbols: Vec::new(),
        };
        for (symbol_id, visibility) in symbols {
            let module = extract_module_from_path(symbol_id);
            let name = symbol_id.rsplit("::").next().unwrap_or(symbol_id);
            match visibility {
                VisibilityScope::Public => {
                    let chains = self.find_reexport_chains(symbol_id);
                    let has_chains = !chains.is_empty();
                    for chain in chains {
                        let reexport_modules: Vec<String> = chain
                            .iter()
                            .map(|node| node.module_path.clone())
                            .collect();
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
                    analysis
                        .restricted_symbols
                        .entry(module.clone())
                        .or_insert_with(Vec::new)
                        .push((name.to_string(), visibility.clone()));
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
    fn detect_visibility_leak(
        &self,
        symbol_id: &str,
        original_visibility: &VisibilityScope,
        origin_module: &str,
        leaked: &mut Vec<LeakedSymbol>,
    ) {
        let importers = self.get_importers(symbol_id);
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(symbol_id.to_string());
        for importer in importers {
            if matches!(importer.kind, UseKind::ReExport | UseKind::ReExportAliased)
                && importer.visibility == VisibilityScope::Public
            {
                if !self
                    .is_visible(
                        origin_module,
                        &importer.module_path,
                        original_visibility,
                    )
                {
                    leaked
                        .push(LeakedSymbol {
                            symbol_id: symbol_id.to_string(),
                            original_visibility: original_visibility.clone(),
                            leaked_to: importer.module_path.clone(),
                            leak_chain: vec![importer.id.clone()],
                        });
                }
                let reexport_path = format!(
                    "{}::{}", importer.module_path, importer.local_name
                );
                self.detect_visibility_leak_recursive(
                    &reexport_path,
                    original_visibility,
                    origin_module,
                    vec![importer.id.clone()],
                    leaked,
                    &mut visited,
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
        visited: &mut HashSet<String>,
    ) {
        if !visited.insert(current_path.to_string()) {
            return;
        }
        let importers = self.get_importers(current_path);
        for importer in importers {
            if matches!(importer.kind, UseKind::ReExport | UseKind::ReExportAliased)
                && importer.visibility == VisibilityScope::Public
            {
                let mut new_chain = chain.clone();
                new_chain.push(importer.id.clone());
                if !self
                    .is_visible(
                        origin_module,
                        &importer.module_path,
                        original_visibility,
                    )
                {
                    leaked
                        .push(LeakedSymbol {
                            symbol_id: current_path.to_string(),
                            original_visibility: original_visibility.clone(),
                            leaked_to: importer.module_path.clone(),
                            leak_chain: new_chain.clone(),
                        });
                }
                let reexport_path = format!(
                    "{}::{}", importer.module_path, importer.local_name
                );
                self.detect_visibility_leak_recursive(
                    &reexport_path,
                    original_visibility,
                    origin_module,
                    new_chain,
                    leaked,
                    visited,
                );
            }
        }
    }
}


impl PatternBindingCollector {
    pub fn new() -> Self {
        Self { bindings: Vec::new() }
    }
    /// Collect bindings from a pattern
    pub fn collect_from_pattern(pat: &Pat) -> Vec<(String, Option<String>)> {
        let mut collector = Self::new();
        collector.visit_pat(pat);
        collector.bindings
    }
}
