use std::collections::{BTreeSet, HashMap};

use crate::ir::{
    ConstItem, Module, ModuleEdge, PubUseItem, StaticItem, TypeAlias, Visibility, Word,
};

use super::super::parser::{ParsedFile, ParsedWorkspace};
use super::IngestError;
use super::ModulesBuild;
use super::types::{
    AliasBinding,
    attribute_to_string, collect_doc_string, convert_type, expr_to_string, flatten_use_tree,
    map_visibility, render_use_item, resolve_use_entry, slugify, to_pascal_case, word_from_ident,
};
pub(crate) fn build_modules(parsed: &ParsedWorkspace) -> Result<ModulesBuild, IngestError> {
    let mut acc: HashMap<String, ModuleAccumulator> = HashMap::new();
    for file in &parsed.files {
        // Skip synthetic layout-only files
        if file.path_string().ends_with("__items.rs") {
            continue;
        }

        let key = module_key(file);

        // Skip synthetic root entries (empty key)
        if key.is_empty() {
            continue;
        }

        acc.entry(key.clone())
            .or_insert_with(|| ModuleAccumulator::new(&key))
            .add_file(file);
    }
    let mut modules = Vec::new();
    let mut module_lookup = HashMap::new();
    let mut file_lookup = HashMap::new();
    for builder in acc.into_values() {
        module_lookup.insert(builder.key.clone(), builder.id.clone());
        modules.push(builder.into_module());
    }
    for file in &parsed.files {
        let path = file.path_string();
        let file_id = format!("file.{}", slugify(&path));
        file_lookup.insert(path.clone(), file_id);
    }
    Ok(ModulesBuild {
        modules,
        module_lookup,
        file_lookup,
    })
}

pub(crate) fn build_module_edges(
    parsed: &ParsedWorkspace,
    module_lookup: &HashMap<String, String>,
) -> Vec<ModuleEdge> {
    let mut acc: HashMap<(String, String), BTreeSet<String>> = HashMap::new();
    for file in &parsed.files {
        let module_key = module_key(file);
        if file.path_string().ends_with("__items.rs") {
            continue;
        }
        if module_key.is_empty() {
            continue;
        }
        let Some(target_id) = module_lookup.get(&module_key) else {
            continue;
        };
        for item in &file.ast.items {
            // Emit edges for `mod foo;` submodule declarations.
            if let syn::Item::Mod(item_mod) = item {
                let child_key = format!("{}::{}", module_key, item_mod.ident);
                if let Some(child_id) = module_lookup.get(&child_key) {
                    if child_id != target_id {
                        acc.entry((target_id.clone(), child_id.clone()))
                            .or_default()
                            .insert(item_mod.ident.to_string());
                    }
                }
            }
            if let syn::Item::Use(item_use) = item {
                let mut entries = Vec::new();
                flatten_use_tree(
                    Vec::new(),
                    &item_use.tree,
                    item_use.leading_colon.is_some(),
                    &mut entries,
                );
                for entry in entries {
                    let Some((source_key, imported)) = resolve_use_entry(&entry, &module_key)
                    else {
                        continue;
                    };
                    let Some(source_id) = module_lookup.get(&source_key) else {
                        continue;
                    };
                    if source_id == target_id {
                        continue;
                    }
                    // Skip edges resolving back to the crate root — these come
                    // from `use crate::X` absolute paths and are not real
                    // architectural dependencies; they would form cycles with
                    // every parent→child mod-declaration edge.
                    if source_key == "crate" {
                        continue;
                    }
                    // Skip edges where a submodule imports from its own
                    // parent namespace (e.g. ir::timeline -> ir).  These
                    // are re-export aliases, not real architectural deps,
                    // and would introduce a cycle with the mod-declaration
                    // edge emitted above.
                    if module_key.starts_with(&format!("{}::", source_key)) {
                        continue;
                    }
                    acc.entry((target_id.clone(), source_id.clone()))
                       .or_default()
                       .insert(imported);
                }
            }
        }
    }
    let mut edges: Vec<ModuleEdge> = acc
        .into_iter()
        .map(|((source, target), imports)| ModuleEdge {
            source,
            target,
            rationale: "ING-001: discovered use statements".to_owned(),
            imported_types: imports.into_iter().collect(),
        })
        .collect();
    edges.sort_by(|a, b| {
        a.source
            .cmp(&b.source)
            .then_with(|| a.target.cmp(&b.target))
    });
    edges
}

pub(crate) fn collect_use_aliases(
    file: &ParsedFile,
    module_key: &str,
    module_lookup: &HashMap<String, String>,
) -> HashMap<String, AliasBinding> {
    let mut aliases = HashMap::new();
    for item in &file.ast.items {
        if let syn::Item::Use(item_use) = item {
            let mut entries = Vec::new();
            flatten_use_tree(
                Vec::new(),
                &item_use.tree,
                item_use.leading_colon.is_some(),
                &mut entries,
            );
            for entry in entries {
                if entry.is_glob {
                    continue;
                }
                let Some((source_key, _)) = resolve_use_entry(&entry, module_key) else {
                    continue;
                };
                if module_lookup.get(&source_key).is_none() {
                    continue;
                }
                let Some(original_name) = entry.segments.last().cloned() else {
                    continue;
                };
                if original_name == "self" {
                    continue;
                }
                let alias_name = entry.alias.clone().unwrap_or_else(|| original_name.clone());
                if alias_name.is_empty() {
                    continue;
                }
                aliases.insert(
                    alias_name,
                    AliasBinding {
                        module_key: source_key.clone(),
                        function_slug: slugify(&original_name),
                    },
                );
            }
        }
    }
    aliases
}

pub(crate) fn module_key(file: &ParsedFile) -> String {
    if file.module_path.is_empty() {
        return "crate".to_owned();
    }

    let mut path = file.module_path.clone();

    if path.last().map(|s| s == "__items").unwrap_or(false) {
        path.pop();
    }

    if path.is_empty() {
        String::new()
    } else {
        path.join("::")
    }
}

// ── ModuleAccumulator ─────────────────────────────────────────────────────────

pub(crate) struct ModuleAccumulator {
    pub key: String,
    pub id: String,
    pub name: Word,
    description: String,
    pub_uses: Vec<PubUseItem>,
    constants: Vec<ConstItem>,
    type_aliases: Vec<TypeAlias>,
    statics: Vec<StaticItem>,
    attributes: Vec<String>,
}

impl ModuleAccumulator {
    pub fn new(key: &str) -> Self {
        let name = Word::new(to_pascal_case(key)).unwrap_or_else(|_| Word::new("Module").unwrap());
        let id = format!("module.{}", slugify(key));
        let description = format!("Ingested module `{key}`");
        Self {
            key: key.to_owned(),
            id,
            name,
            description,
            pub_uses: Vec::new(),
            constants: Vec::new(),
            type_aliases: Vec::new(),
            statics: Vec::new(),
            attributes: Vec::new(),
        }
    }

    pub fn add_file(&mut self, file: &ParsedFile) {
        self.collect_attributes(&file.ast.attrs);
        self.collect_items(&file.ast.items);
    }

    fn collect_attributes(&mut self, attrs: &[syn::Attribute]) {
        for attr in attrs {
            if !matches!(attr.style, syn::AttrStyle::Inner(_)) {
                continue;
            }
            if let Some(rendered) = attribute_to_string(attr) {
                if !self.attributes.contains(&rendered) {
                    self.attributes.push(rendered);
                }
            }
        }
    }

    fn collect_items(&mut self, items: &[syn::Item]) {
        for item in items {
            match item {
                syn::Item::Use(item_use) => {
                    if matches!(map_visibility(&item_use.vis), Visibility::Public) {
                        let rendered = render_use_item(item_use);
                        self.pub_uses.push(PubUseItem { path: rendered });
                    }
                }
                syn::Item::Const(item_const) => {
                    if matches!(map_visibility(&item_const.vis), Visibility::Public) {
                        self.constants.push(ConstItem {
                            name: word_from_ident(&item_const.ident, "Const"),
                            ty: convert_type(&item_const.ty),
                            value_expr: expr_to_string(&item_const.expr),
                        });
                    }
                }
                syn::Item::Type(item_type) => {
                    if matches!(map_visibility(&item_type.vis), Visibility::Public) {
                        self.type_aliases.push(TypeAlias {
                            name: word_from_ident(&item_type.ident, "TypeAlias"),
                            target: convert_type(&item_type.ty),
                        });
                    }
                }
                syn::Item::Static(item_static) => {
                    let doc = collect_doc_string(&item_static.attrs);
                    self.statics.push(StaticItem {
                        name: item_static.ident.to_string(),
                        ty: convert_type(&item_static.ty),
                        value_expr: expr_to_string(&item_static.expr),
                        mutable: matches!(item_static.mutability, syn::StaticMutability::Mut(_)),
                        doc,
                        visibility: map_visibility(&item_static.vis),
                    });
                }
                _ => {}
            }
        }
    }

    pub fn into_module(self) -> Module {
        Module {
            id: self.id,
            name: self.name,
            visibility: Visibility::Public,
            description: self.description,
            pub_uses: self.pub_uses,
            constants: self.constants,
            type_aliases: self.type_aliases,
            statics: self.statics,
            attributes: self.attributes,
        }
    }
}
