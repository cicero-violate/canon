use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use quote::ToTokens;

use crate::ir::{
    CallEdge, CanonicalIr, CanonicalMeta, ConstItem, EnumNode, EnumVariant, EnumVariantFields,
    Field, FileNode, Function, FunctionContract, GenericParam, ImplBlock, ImplFunctionBinding,
    Language, Module, ModuleEdge, Project, PubUseItem, Receiver, StaticItem, Struct, StructKind,
    Trait, TraitFunction, TypeAlias, TypeKind, TypeRef, ValuePort, VersionContract, Visibility,
    Word,
};

use super::IngestError;
use super::parser::{ParsedFile, ParsedWorkspace};
use syn::visit::{self, Visit};

/// Convert parsed files into a starter `CanonicalIr`.
///
/// TODO(ING-001): Populate structs, traits, impls, functions, and edges.
pub(crate) fn build_ir(root: &Path, parsed: ParsedWorkspace) -> Result<CanonicalIr, IngestError> {
    let project_name = derive_project_name(root);
    let ModulesBuild {
        modules,
        module_lookup,
        file_lookup,
    } = build_modules(&parsed)?;
    let module_edges = build_module_edges(&parsed, &module_lookup);
    let structs = build_structs(&parsed, &module_lookup, &file_lookup);
    let enums = build_enums(&parsed, &module_lookup);
    let traits = build_traits(&parsed, &module_lookup, &file_lookup);
    let (impl_blocks, functions) = build_impls_and_functions(&parsed, &module_lookup, &file_lookup);
    let call_edges = build_call_edges(&parsed, &module_lookup, &functions);
    let ir = CanonicalIr {
        meta: CanonicalMeta {
            version: "0.0.0-ingest".to_owned(),
            law_revision: Word::new("CanonIngest").expect("valid law word"),
            description: format!("Ingested workspace `{}`", root.display()),
        },
        version_contract: VersionContract {
            current: "0.0.0-ingest".to_owned(),
            compatible_with: vec![],
            migration_proofs: vec![],
        },
        project: Project {
            name: project_name,
            version: "0.0.0".to_owned(),
            language: Language::Rust,
        },
        modules,
        module_edges,
        structs,
        enums,
        traits,
        impl_blocks,
        functions,
        call_edges,
        tick_graphs: Vec::new(),
        system_graphs: Vec::new(),
        loop_policies: Vec::new(),
        ticks: Vec::new(),
        tick_epochs: Vec::new(),
        plans: Vec::new(),
        executions: Vec::new(),
        admissions: Vec::new(),
        applied_deltas: Vec::new(),
        gpu_functions: Vec::new(),
        proposals: Vec::new(),
        judgments: Vec::new(),
        judgment_predicates: Vec::new(),
        deltas: Vec::new(),
        proofs: Vec::new(),
        learning: Vec::new(),
        errors: Vec::new(),
        dependencies: Vec::new(),
        file_hashes: HashMap::new(),
    };
    Ok(ir)
}

fn derive_project_name(root: &Path) -> Word {
    let stem = root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("CanonProject");
    Word::new(to_pascal_case(stem)).unwrap_or_else(|_| Word::new("CanonProject").unwrap())
}

struct ModulesBuild {
    modules: Vec<Module>,
    module_lookup: HashMap<String, String>,
    file_lookup: HashMap<String, String>,
}

fn build_modules(parsed: &ParsedWorkspace) -> Result<ModulesBuild, IngestError> {
    let mut acc: HashMap<String, ModuleAccumulator> = HashMap::new();
    for file in &parsed.files {
        let key = module_key(file);
        acc.entry(key.clone())
            .or_insert_with(|| ModuleAccumulator::new(&key))
            .add_file(file);
    }
    let mut modules = Vec::new();
    let mut module_lookup = HashMap::new();
    let mut file_lookup = HashMap::new();
    for builder in acc.into_values() {
        module_lookup.insert(builder.key.clone(), builder.id.clone());
        for node in &builder.files {
            file_lookup.insert(node.name.clone(), node.id.clone());
        }
        modules.push(builder.into_module());
    }
    Ok(ModulesBuild {
        modules,
        module_lookup,
        file_lookup,
    })
}

fn build_module_edges(
    parsed: &ParsedWorkspace,
    module_lookup: &HashMap<String, String>,
) -> Vec<ModuleEdge> {
    let mut acc: HashMap<(String, String), BTreeSet<String>> = HashMap::new();
    for file in &parsed.files {
        let module_key = module_key(file);
        let Some(target_id) = module_lookup.get(&module_key) else {
            continue;
        };
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
                    acc.entry((source_id.clone(), target_id.clone()))
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

fn collect_use_aliases(
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

struct ModuleAccumulator {
    key: String,
    id: String,
    name: Word,
    description: String,
    files: Vec<FileNode>,
    pub_uses: Vec<PubUseItem>,
    constants: Vec<ConstItem>,
    type_aliases: Vec<TypeAlias>,
    statics: Vec<StaticItem>,
    attributes: Vec<String>,
}

#[derive(Clone, Debug)]
struct UseEntry {
    segments: Vec<String>,
    alias: Option<String>,
    is_glob: bool,
    leading_colon: bool,
}

#[derive(Clone)]
struct FunctionLookupEntry {
    id: String,
    visibility: Visibility,
}

#[derive(Clone)]
struct AliasBinding {
    module_key: String,
    function_slug: String,
}

impl ModuleAccumulator {
    fn new(key: &str) -> Self {
        let name = Word::new(to_pascal_case(key)).unwrap_or_else(|_| Word::new("Module").unwrap());
        let id = format!("module.{}", slugify(key));
        let description = format!("Ingested module `{key}`");
        Self {
            key: key.to_owned(),
            id,
            name,
            description,
            files: Vec::new(),
            pub_uses: Vec::new(),
            constants: Vec::new(),
            type_aliases: Vec::new(),
            statics: Vec::new(),
            attributes: Vec::new(),
        }
    }

    fn add_file(&mut self, file: &ParsedFile) {
        let display = file.path_string();
        let node = FileNode {
            id: format!("file.{}", slugify(&display)),
            name: display,
        };
        self.files.push(node);
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

    fn into_module(self) -> Module {
        Module {
            id: self.id,
            name: self.name,
            visibility: Visibility::Public,
            description: self.description,
            files: self.files,
            file_edges: Vec::new(),
            pub_uses: self.pub_uses,
            constants: self.constants,
            type_aliases: self.type_aliases,
            statics: self.statics,
            attributes: self.attributes,
        }
    }
}

fn slugify(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "root".to_owned()
    } else {
        out
    }
}

fn to_pascal_case(input: &str) -> String {
    let mut out = String::new();
    let mut capitalize = true;
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            if out.is_empty() || capitalize {
                out.push(ch.to_ascii_uppercase());
                capitalize = false;
            } else {
                out.push(ch.to_ascii_lowercase());
            }
        } else {
            capitalize = true;
        }
    }
    if out.is_empty() {
        "Module".to_owned()
    } else {
        out
    }
}

fn module_key(file: &ParsedFile) -> String {
    if file.module_path.is_empty() {
        "crate".to_owned()
    } else {
        file.module_path.join("::")
    }
}

fn render_use_item(item: &syn::ItemUse) -> String {
    let body = use_tree_to_string(&item.tree);
    if item.leading_colon.is_some() {
        format!("::{body}")
    } else {
        body
    }
}

fn use_tree_to_string(tree: &syn::UseTree) -> String {
    match tree {
        syn::UseTree::Path(path) => {
            let rest = use_tree_to_string(&path.tree);
            if rest.is_empty() {
                path.ident.to_string()
            } else {
                format!("{}::{}", path.ident, rest)
            }
        }
        syn::UseTree::Name(name) => name.ident.to_string(),
        syn::UseTree::Rename(rename) => format!("{} as {}", rename.ident, rename.rename),
        syn::UseTree::Glob(_) => "*".to_owned(),
        syn::UseTree::Group(group) => {
            let parts = group
                .items
                .iter()
                .map(use_tree_to_string)
                .collect::<Vec<_>>();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

fn flatten_use_tree(
    prefix: Vec<String>,
    tree: &syn::UseTree,
    leading_colon: bool,
    acc: &mut Vec<UseEntry>,
) {
    match tree {
        syn::UseTree::Path(path) => {
            let mut next = prefix;
            next.push(path.ident.to_string());
            flatten_use_tree(next, &path.tree, leading_colon, acc);
        }
        syn::UseTree::Name(name) => {
            let mut segments = prefix;
            segments.push(name.ident.to_string());
            acc.push(UseEntry {
                segments,
                alias: None,
                is_glob: false,
                leading_colon,
            });
        }
        syn::UseTree::Rename(rename) => {
            let mut segments = prefix;
            segments.push(rename.ident.to_string());
            acc.push(UseEntry {
                segments,
                alias: Some(rename.rename.to_string()),
                is_glob: false,
                leading_colon,
            });
        }
        syn::UseTree::Glob(_) => {
            acc.push(UseEntry {
                segments: prefix,
                alias: None,
                is_glob: true,
                leading_colon,
            });
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                flatten_use_tree(prefix.clone(), item, leading_colon, acc);
            }
        }
    }
}

fn resolve_use_entry(entry: &UseEntry, module_key: &str) -> Option<(String, String)> {
    let mut segments = entry.segments.clone();
    let mut base = if entry.leading_colon {
        Vec::new()
    } else {
        module_segments_from_key(module_key)
    };
    if let Some(first) = segments.first() {
        match first.as_str() {
            "crate" => {
                base.clear();
                segments.remove(0);
            }
            "self" => {
                base = module_segments_from_key(module_key);
                segments.remove(0);
            }
            "super" => {
                base = module_segments_from_key(module_key);
                while let Some(seg) = segments.first() {
                    if seg == "super" {
                        segments.remove(0);
                        if !base.is_empty() {
                            base.pop();
                        }
                    } else {
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    base.extend(segments);
    if entry.is_glob {
        let module_name = if base.is_empty() {
            module_key.to_owned()
        } else {
            base.join("::")
        };
        if module_name == module_key {
            return None;
        }
        return Some((module_name, "*".to_owned()));
    }
    if base.is_empty() {
        return None;
    }
    let item_name = base.pop()?;
    if item_name == "self" {
        return None;
    }
    let module_name = if base.is_empty() {
        "crate".to_owned()
    } else {
        base.join("::")
    };
    if module_name == module_key {
        return None;
    }
    let imported = if let Some(alias) = &entry.alias {
        format!("{item_name} as {alias}")
    } else {
        item_name
    };
    Some((module_name, imported))
}

fn module_segments_from_key(key: &str) -> Vec<String> {
    if key.is_empty() || key == "crate" {
        Vec::new()
    } else {
        key.split("::").map(|s| s.to_string()).collect()
    }
}

fn attribute_to_string(attr: &syn::Attribute) -> Option<String> {
    match &attr.meta {
        syn::Meta::Path(path) => Some(path_to_string(path)),
        syn::Meta::List(list) => {
            let path = path_to_string(&list.path);
            let tokens = list.tokens.to_string();
            if tokens.is_empty() {
                Some(path)
            } else {
                Some(format!("{path}{tokens}"))
            }
        }
        syn::Meta::NameValue(name_value) => Some(format!(
            "{} = {}",
            path_to_string(&name_value.path),
            expr_to_string(&name_value.value)
        )),
    }
}

fn collect_doc_string(attrs: &[syn::Attribute]) -> Option<String> {
    let mut docs = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        if let syn::Meta::NameValue(meta) = &attr.meta {
            if let syn::Expr::Lit(expr_lit) = &meta.value {
                if let syn::Lit::Str(lit) = &expr_lit.lit {
                    docs.push(lit.value());
                }
            }
        }
    }
    if docs.is_empty() {
        None
    } else {
        Some(docs.join("\n"))
    }
}

fn expr_to_string(expr: &syn::Expr) -> String {
    expr.to_token_stream().to_string()
}

fn collect_derives(attrs: &[syn::Attribute]) -> Vec<String> {
    let mut derives = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("derive") {
            continue;
        }
        if let Ok(list) = attr.parse_args_with(|input: syn::parse::ParseStream| {
            let punct: syn::punctuated::Punctuated<syn::Path, syn::Token![,]> =
                syn::punctuated::Punctuated::parse_terminated(input)?;
            Ok(punct)
        }) {
            for path in list {
                derives.push(path_to_string(&path));
            }
        }
    }
    derives
}

fn convert_fields(fields: &syn::Fields) -> (StructKind, Vec<Field>) {
    match fields {
        syn::Fields::Named(named) => {
            let mut out = Vec::new();
            for field in &named.named {
                let name = field
                    .ident
                    .as_ref()
                    .map(|ident| word_from_ident(ident, "Field"))
                    .unwrap_or_else(|| Word::new("Field").unwrap());
                out.push(Field {
                    name,
                    ty: convert_type(&field.ty),
                    visibility: map_visibility(&field.vis),
                    doc: None,
                });
            }
            (StructKind::Normal, out)
        }
        syn::Fields::Unnamed(unnamed) => {
            let mut out = Vec::new();
            for (idx, field) in unnamed.unnamed.iter().enumerate() {
                let fallback = format!("Field{idx}");
                out.push(Field {
                    name: word_from_string(&fallback, "Field"),
                    ty: convert_type(&field.ty),
                    visibility: map_visibility(&field.vis),
                    doc: None,
                });
            }
            (StructKind::Tuple, out)
        }
        syn::Fields::Unit => (StructKind::Unit, Vec::new()),
    }
}

fn map_visibility(vis: &syn::Visibility) -> Visibility {
    match vis {
        syn::Visibility::Public(_) => Visibility::Public,
        syn::Visibility::Restricted(restricted) => {
            if let Some(path) = &restricted.in_token {
                let _ = path; // placeholder to avoid unused warnings if format needed later
            }
            if restricted.path.is_ident("crate") {
                Visibility::PubCrate
            } else if restricted.path.is_ident("super") {
                Visibility::PubSuper
            } else {
                Visibility::Private
            }
        }
        _ => Visibility::Private,
    }
}

fn word_from_ident(ident: &syn::Ident, fallback: &str) -> Word {
    word_from_string(&ident.to_string(), fallback)
}

fn word_from_string(value: &str, fallback: &str) -> Word {
    Word::new(value)
        .or_else(|_| Word::new(to_pascal_case(value)))
        .unwrap_or_else(|_| Word::new(fallback).unwrap())
}

fn convert_type(ty: &syn::Type) -> TypeRef {
    match ty {
        syn::Type::Reference(r) => {
            let mut inner = convert_type(&r.elem);
            inner.ref_kind = if r.mutability.is_some() {
                crate::ir::RefKind::MutRef
            } else {
                crate::ir::RefKind::Ref
            };
            inner.lifetime = r.lifetime.as_ref().map(|lt| lt.to_string());
            inner
        }
        syn::Type::Path(path) => path_type(path),
        syn::Type::Tuple(tuple) => TypeRef {
            name: Word::new("Tuple").unwrap(),
            kind: TypeKind::Tuple,
            params: tuple.elems.iter().map(convert_type).collect(),
            ref_kind: crate::ir::RefKind::None,
            lifetime: None,
        },
        syn::Type::Never(_) => TypeRef {
            name: Word::new("Never").unwrap(),
            kind: TypeKind::Never,
            params: Vec::new(),
            ref_kind: crate::ir::RefKind::None,
            lifetime: None,
        },
        syn::Type::ImplTrait(impl_trait) => TypeRef {
            name: Word::new("ImplTrait").unwrap(),
            kind: TypeKind::ImplTrait,
            params: impl_trait
                .bounds
                .iter()
                .filter_map(type_from_bound)
                .collect(),
            ref_kind: crate::ir::RefKind::None,
            lifetime: None,
        },
        syn::Type::TraitObject(obj) => TypeRef {
            name: Word::new("DynTrait").unwrap(),
            kind: TypeKind::DynTrait,
            params: obj.bounds.iter().filter_map(type_from_bound).collect(),
            ref_kind: crate::ir::RefKind::None,
            lifetime: None,
        },
        syn::Type::Paren(paren) => convert_type(&paren.elem),
        _ => TypeRef {
            name: Word::new("External").unwrap(),
            kind: TypeKind::External,
            params: Vec::new(),
            ref_kind: crate::ir::RefKind::None,
            lifetime: None,
        },
    }
}

fn path_type(type_path: &syn::TypePath) -> TypeRef {
    let ident = type_path
        .path
        .segments
        .last()
        .map(|seg| seg.ident.to_string())
        .unwrap_or_else(|| "Type".to_owned());
    let kind = if ident == "Self" {
        TypeKind::SelfType
    } else {
        TypeKind::External
    };
    let mut params = Vec::new();
    if let Some(segment) = type_path.path.segments.last() {
        if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
            for arg in &args.args {
                if let syn::GenericArgument::Type(arg_ty) = arg {
                    params.push(convert_type(arg_ty));
                }
            }
        }
    }
    TypeRef {
        name: word_from_string(&ident, "Type"),
        kind,
        params,
        ref_kind: crate::ir::RefKind::None,
        lifetime: None,
    }
}

fn type_from_bound(bound: &syn::TypeParamBound) -> Option<TypeRef> {
    match bound {
        syn::TypeParamBound::Trait(trait_bound) => {
            let path = trait_bound.path.clone();
            Some(TypeRef {
                name: word_from_string(
                    &path
                        .segments
                        .last()
                        .map(|seg| seg.ident.to_string())
                        .unwrap_or_else(|| "Trait".to_owned()),
                    "Trait",
                ),
                kind: TypeKind::External,
                params: Vec::new(),
                ref_kind: crate::ir::RefKind::None,
                lifetime: None,
            })
        }
        _ => None,
    }
}

fn path_to_string(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}
fn build_structs(
    parsed: &ParsedWorkspace,
    module_lookup: &HashMap<String, String>,
    file_lookup: &HashMap<String, String>,
) -> Vec<Struct> {
    let mut structs = Vec::new();
    for file in &parsed.files {
        let module_key = module_key(file);
        let Some(module_id) = module_lookup.get(&module_key) else {
            continue;
        };
        let file_id = file_lookup.get(&file.path_string()).cloned();
        for item in &file.ast.items {
            if let syn::Item::Struct(item_struct) = item {
                structs.push(struct_from_syn(module_id, file_id.clone(), item_struct));
            }
        }
    }
    structs
}

fn build_enums(parsed: &ParsedWorkspace, module_lookup: &HashMap<String, String>) -> Vec<EnumNode> {
    let mut enums = Vec::new();
    for file in &parsed.files {
        let module_key = module_key(file);
        let Some(module_id) = module_lookup.get(&module_key) else {
            continue;
        };
        for item in &file.ast.items {
            if let syn::Item::Enum(item_enum) = item {
                enums.push(enum_from_syn(module_id, item_enum));
            }
        }
    }
    enums
}

fn build_traits(
    parsed: &ParsedWorkspace,
    module_lookup: &HashMap<String, String>,
    file_lookup: &HashMap<String, String>,
) -> Vec<Trait> {
    let mut traits = Vec::new();
    for file in &parsed.files {
        let module_key = module_key(file);
        let Some(module_id) = module_lookup.get(&module_key) else {
            continue;
        };
        let file_id = file_lookup.get(&file.path_string()).cloned();
        for item in &file.ast.items {
            if let syn::Item::Trait(trait_item) = item {
                traits.push(trait_from_syn(module_id, file_id.clone(), trait_item));
            }
        }
    }
    traits
}

fn build_impls_and_functions(
    parsed: &ParsedWorkspace,
    module_lookup: &HashMap<String, String>,
    file_lookup: &HashMap<String, String>,
) -> (Vec<ImplBlock>, Vec<Function>) {
    let mut impls = Vec::new();
    let mut functions = Vec::new();
    for file in &parsed.files {
        let module_key = module_key(file);
        let Some(module_id) = module_lookup.get(&module_key) else {
            continue;
        };
        let file_id = file_lookup.get(&file.path_string()).cloned();
        for syn_item in &file.ast.items {
            match syn_item {
                syn::Item::Fn(item_fn) => {
                    functions.push(function_from_syn(module_id, file_id.clone(), item_fn, None));
                }
                syn::Item::Impl(impl_block) => {
                    match impl_block_from_syn(module_id, file_id.clone(), impl_block) {
                        ImplMapping::Standalone(funcs) => {
                            functions.extend(funcs);
                        }
                        ImplMapping::ImplBlock(block, funcs) => {
                            functions.extend(funcs);
                            impls.push(block);
                        }
                        ImplMapping::Unsupported => {}
                    }
                }
                _ => {}
            }
        }
    }
    (impls, functions)
}

fn build_call_edges(
    parsed: &ParsedWorkspace,
    module_lookup: &HashMap<String, String>,
    functions: &[Function],
) -> Vec<CallEdge> {
    let mut module_reverse = HashMap::new();
    for (key, id) in module_lookup {
        module_reverse.insert(id.clone(), key.clone());
    }
    let mut function_lookup: HashMap<(String, String), FunctionLookupEntry> = HashMap::new();
    for function in functions {
        if let Some(module_key) = module_reverse.get(&function.module) {
            if let Some(slug) = function_name_slug(function) {
                function_lookup.insert(
                    (module_key.clone(), slug),
                    FunctionLookupEntry {
                        id: function.id.clone(),
                        visibility: function.visibility,
                    },
                );
            }
        }
    }
    let mut discovered: BTreeSet<(String, String)> = BTreeSet::new();
    for file in &parsed.files {
        let module_key = module_key(file);
        let Some(_module_id) = module_lookup.get(&module_key) else {
            continue;
        };
        let module_segments = module_segments_from_key(&module_key);
        let alias_map = collect_use_aliases(file, &module_key, module_lookup);
        for item in &file.ast.items {
            match item {
                syn::Item::Fn(item_fn) => {
                    if let Some(caller_id) =
                        lookup_function_id(&module_key, &item_fn.sig.ident, &function_lookup)
                    {
                        collect_calls_in_block(
                            &item_fn.block,
                            &module_segments,
                            &caller_id,
                            &function_lookup,
                            &alias_map,
                            &mut discovered,
                        );
                    }
                }
                syn::Item::Impl(impl_block) => {
                    for impl_item in &impl_block.items {
                        if let syn::ImplItem::Fn(method) = impl_item {
                            if let Some(caller_id) =
                                lookup_function_id(&module_key, &method.sig.ident, &function_lookup)
                            {
                                collect_calls_in_block(
                                    &method.block,
                                    &module_segments,
                                    &caller_id,
                                    &function_lookup,
                                    &alias_map,
                                    &mut discovered,
                                );
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    discovered
        .into_iter()
        .enumerate()
        .map(|(idx, (caller, callee))| CallEdge {
            id: format!("call_edge.{}", idx + 1),
            caller,
            callee,
            rationale: "ING-001: discovered call expression".to_owned(),
        })
        .collect()
}

fn function_from_syn(
    module_id: &str,
    file_id: Option<String>,
    item: &syn::ItemFn,
    impl_context: Option<(&str, Option<&syn::Path>)>,
) -> Function {
    let name = word_from_ident(&item.sig.ident, "Function");
    let visibility = map_visibility(&item.vis);
    let inputs = convert_inputs(&item.sig.inputs);
    let outputs = convert_return_type(&item.sig.output);
    let (impl_id, trait_function) = if let Some((impl_id, trait_path)) = impl_context {
        (
            impl_id.to_owned(),
            trait_path.map(|path| trait_path_to_trait_fn_id(path, module_id, &item.sig.ident)),
        )
    } else {
        (String::new(), None)
    };

    Function {
        id: format!(
            "function.{}.{}",
            slugify(module_id),
            slugify(&item.sig.ident.to_string())
        ),
        name,
        module: module_id.to_owned(),
        impl_id,
        trait_function: trait_function.unwrap_or_default(),
        visibility,
        doc: None,
        lifetime_params: item
            .sig
            .generics
            .lifetimes()
            .map(|lt| lt.lifetime.to_string())
            .collect(),
        receiver: convert_receiver(&item.sig.inputs),
        is_async: item.sig.asyncness.is_some(),
        is_unsafe: item.sig.unsafety.is_some(),
        generics: convert_generics(&item.sig.generics),
        where_clauses: Vec::new(),
        file_id,
        inputs,
        outputs,
        deltas: Vec::new(),
        contract: FunctionContract {
            total: true,
            deterministic: true,
            explicit_inputs: true,
            explicit_outputs: true,
            effects_are_deltas: true,
        },
        metadata: Default::default(),
    }
}

fn impl_block_from_syn(
    module_id: &str,
    file_id: Option<String>,
    block: &syn::ItemImpl,
) -> ImplMapping {
    let Some((_, trait_path, _)) = &block.trait_ else {
        let mut standalone = Vec::new();
        for item in &block.items {
            if let syn::ImplItem::Fn(method) = item {
                standalone.push(function_from_impl_item(
                    module_id,
                    file_id.clone(),
                    method,
                    None,
                ));
            }
        }
        return ImplMapping::Standalone(standalone);
    };
    let syn::Type::Path(self_path) = block.self_ty.as_ref() else {
        return ImplMapping::Unsupported;
    };
    let struct_id = type_path_to_struct_id(self_path, module_id);
    let trait_id = trait_path_to_trait_id(trait_path, module_id);
    let impl_id = format!(
        "impl.{}.{}.{}",
        slugify(module_id),
        slugify(&struct_id),
        slugify(&trait_id)
    );
    let mut bindings = Vec::new();
    let mut functions = Vec::new();
    for item in &block.items {
        if let syn::ImplItem::Fn(method) = item {
            let function = function_from_impl_item(
                module_id,
                file_id.clone(),
                method,
                Some((&impl_id, Some(trait_path))),
            );
            let fn_id = function.id.clone();
            let trait_fn_id = trait_path_to_trait_fn_id(trait_path, module_id, &method.sig.ident);
            bindings.push(ImplFunctionBinding {
                trait_fn: trait_fn_id,
                function: fn_id.clone(),
            });
            functions.push(function);
        }
    }
    if bindings.is_empty() {
        return ImplMapping::Unsupported;
    }
    ImplMapping::ImplBlock(
        ImplBlock {
            id: impl_id,
            module: module_id.to_owned(),
            struct_id,
            trait_id,
            functions: bindings,
        },
        functions,
    )
}

fn function_from_impl_item(
    module_id: &str,
    file_id: Option<String>,
    method: &syn::ImplItemFn,
    context: Option<(&str, Option<&syn::Path>)>,
) -> Function {
    let item_fn = syn::ItemFn {
        attrs: method.attrs.clone(),
        vis: syn::Visibility::Inherited,
        sig: method.sig.clone(),
        block: Box::new(method.block.clone()),
    };
    function_from_syn(module_id, file_id, &item_fn, context)
}

fn enum_from_syn(module_id: &str, item: &syn::ItemEnum) -> EnumNode {
    let name = word_from_ident(&item.ident, "Enum");
    let variants = item.variants.iter().map(enum_variant_from_syn).collect();
    EnumNode {
        id: format!(
            "enum.{}.{}",
            slugify(module_id),
            slugify(&item.ident.to_string())
        ),
        name,
        module: module_id.to_owned(),
        visibility: map_visibility(&item.vis),
        variants,
        history: Vec::new(),
    }
}

fn enum_variant_from_syn(variant: &syn::Variant) -> EnumVariant {
    let name = word_from_ident(&variant.ident, "Variant");
    let fields = match &variant.fields {
        syn::Fields::Unit => EnumVariantFields::Unit,
        syn::Fields::Unnamed(unnamed) => {
            let types = unnamed
                .unnamed
                .iter()
                .map(|f| convert_type(&f.ty))
                .collect();
            EnumVariantFields::Tuple { types }
        }
        syn::Fields::Named(named) => {
            let mut fields = Vec::new();
            for field in &named.named {
                let field_name = field
                    .ident
                    .as_ref()
                    .map(|ident| word_from_ident(ident, "Field"))
                    .unwrap_or_else(|| Word::new("Field").unwrap());
                fields.push(Field {
                    name: field_name,
                    ty: convert_type(&field.ty),
                    visibility: map_visibility(&field.vis),
                    doc: collect_doc_string(&field.attrs),
                });
            }
            EnumVariantFields::Struct { fields }
        }
    };
    EnumVariant { name, fields }
}

fn struct_from_syn(module_id: &str, file_id: Option<String>, item: &syn::ItemStruct) -> Struct {
    let name = word_from_ident(&item.ident, "Struct");
    let visibility = map_visibility(&item.vis);
    let derives = collect_derives(&item.attrs);
    let (kind, fields) = convert_fields(&item.fields);
    Struct {
        id: format!(
            "struct.{}.{}",
            slugify(module_id),
            slugify(&item.ident.to_string())
        ),
        name,
        module: module_id.to_owned(),
        visibility,
        file_id,
        derives,
        doc: None,
        kind,
        fields,
        history: Vec::new(),
    }
}

fn trait_from_syn(module_id: &str, file_id: Option<String>, item: &syn::ItemTrait) -> Trait {
    let name = word_from_ident(&item.ident, "Trait");
    let visibility = map_visibility(&item.vis);
    let trait_slug = slugify(&item.ident.to_string());
    let trait_id = format!("trait.{}.{}", slugify(module_id), trait_slug);
    let supertraits = item
        .supertraits
        .iter()
        .filter_map(|bound| match bound {
            syn::TypeParamBound::Trait(trait_bound) => Some(path_to_string(&trait_bound.path)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let functions = item
        .items
        .iter()
        .filter_map(|trait_item| {
            if let syn::TraitItem::Fn(fn_item) = trait_item {
                Some(trait_fn_from_syn(&trait_id, &item.ident, fn_item))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    Trait {
        id: trait_id,
        name,
        module: module_id.to_owned(),
        visibility,
        file_id,
        generic_params: convert_generics(&item.generics),
        functions,
        supertraits,
        associated_types: Vec::new(),
        associated_consts: Vec::new(),
    }
}

fn trait_fn_from_syn(
    trait_id: &str,
    _trait_name: &syn::Ident,
    item: &syn::TraitItemFn,
) -> TraitFunction {
    let fn_name = word_from_ident(&item.sig.ident, "TraitFn");
    let inputs = convert_inputs(&item.sig.inputs);
    let outputs = convert_return_type(&item.sig.output);
    let fn_slug = slugify(&item.sig.ident.to_string());
    TraitFunction {
        id: format!("trait_fn.{}.{}", trait_id, fn_slug),
        name: fn_name,
        inputs,
        outputs,
        default_body: None,
    }
}
fn convert_generics(generics: &syn::Generics) -> Vec<GenericParam> {
    let mut params = Vec::new();
    for param in &generics.params {
        if let syn::GenericParam::Type(ty) = param {
            let bounds = ty
                .bounds
                .iter()
                .filter_map(|bound| match bound {
                    syn::TypeParamBound::Trait(trait_bound) => {
                        Some(path_to_string(&trait_bound.path))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            params.push(GenericParam {
                name: word_from_ident(&ty.ident, "Param"),
                bounds,
            });
        }
    }
    params
}

fn convert_inputs(
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma>,
) -> Vec<ValuePort> {
    let mut result = Vec::new();
    for (idx, arg) in inputs.iter().enumerate() {
        match arg {
            syn::FnArg::Receiver(_) => continue,
            syn::FnArg::Typed(pat_ty) => {
                let name = match &*pat_ty.pat {
                    syn::Pat::Ident(ident) => ident.ident.to_string(),
                    _ => format!("param{idx}"),
                };
                result.push(ValuePort {
                    name: word_from_string(&name, "Param"),
                    ty: convert_type(&pat_ty.ty),
                });
            }
        }
    }
    result
}

fn convert_return_type(ret: &syn::ReturnType) -> Vec<ValuePort> {
    match ret {
        syn::ReturnType::Default => vec![ValuePort {
            name: word_from_string("Output", "Output"),
            ty: TypeRef {
                name: Word::new("Unit").unwrap(),
                kind: TypeKind::Tuple,
                params: Vec::new(),
                ref_kind: crate::ir::RefKind::None,
                lifetime: None,
            },
        }],
        syn::ReturnType::Type(_, ty) => vec![ValuePort {
            name: word_from_string("Output", "Output"),
            ty: convert_type(ty),
        }],
    }
}

fn convert_receiver(
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma>,
) -> Receiver {
    if let Some(syn::FnArg::Receiver(receiver)) = inputs.first() {
        match (receiver.reference.as_ref(), receiver.mutability.is_some()) {
            (None, _) => Receiver::SelfVal,
            (Some(_), false) => Receiver::SelfRef,
            (Some(_), true) => Receiver::SelfMutRef,
        }
    } else {
        Receiver::None
    }
}
fn trait_path_to_trait_fn_id(
    trait_path: &syn::Path,
    module_id: &str,
    fn_ident: &syn::Ident,
) -> String {
    let _trait_name = trait_path
        .segments
        .last()
        .map(|seg| seg.ident.to_string())
        .unwrap_or_else(|| "Trait".to_owned());
    let trait_id = trait_path_to_trait_id(trait_path, module_id);
    format!("trait_fn.{}.{}", trait_id, slugify(&fn_ident.to_string()))
}

fn trait_path_to_trait_id(path: &syn::Path, module_id: &str) -> String {
    let trait_name = path
        .segments
        .last()
        .map(|seg| seg.ident.to_string())
        .unwrap_or_else(|| "Trait".to_owned());
    format!("trait.{}.{}", slugify(module_id), slugify(&trait_name))
}

fn type_path_to_struct_id(path: &syn::TypePath, module_id: &str) -> String {
    let struct_name = path
        .path
        .segments
        .last()
        .map(|seg| seg.ident.to_string())
        .unwrap_or_else(|| "Struct".to_owned());
    format!("struct.{}.{}", slugify(module_id), slugify(&struct_name))
}
enum ImplMapping {
    Standalone(Vec<Function>),
    ImplBlock(ImplBlock, Vec<Function>),
    Unsupported,
}

fn lookup_function_id(
    module_key: &str,
    ident: &syn::Ident,
    functions: &HashMap<(String, String), FunctionLookupEntry>,
) -> Option<String> {
    let slug = slugify(&ident.to_string());
    functions
        .get(&(module_key.to_owned(), slug))
        .map(|entry| entry.id.clone())
}

fn collect_calls_in_block(
    block: &syn::Block,
    module_segments: &[String],
    caller_id: &str,
    functions: &HashMap<(String, String), FunctionLookupEntry>,
    aliases: &HashMap<String, AliasBinding>,
    discovered: &mut BTreeSet<(String, String)>,
) {
    let mut visitor = CallCollector {
        caller_id,
        module_segments,
        functions,
        aliases,
        discovered,
    };
    visitor.visit_block(block);
}

struct CallCollector<'a> {
    caller_id: &'a str,
    module_segments: &'a [String],
    functions: &'a HashMap<(String, String), FunctionLookupEntry>,
    aliases: &'a HashMap<String, AliasBinding>,
    discovered: &'a mut BTreeSet<(String, String)>,
}

impl<'ast, 'a> Visit<'ast> for CallCollector<'a> {
    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let Some(path) = extract_call_path(&node.func) {
            if let Some(entry) = resolve_function_path(self.module_segments, path, self.functions) {
                if entry.visibility == Visibility::Public && entry.id != self.caller_id {
                    self.discovered
                        .insert((self.caller_id.to_owned(), entry.id.clone()));
                }
            } else if path.segments.len() == 1 {
                if let Some(segment) = path.segments.first() {
                    let name = segment.ident.to_string();
                    if let Some(binding) = self.aliases.get(&name) {
                        let key = (binding.module_key.clone(), binding.function_slug.clone());
                        if let Some(entry) = self.functions.get(&key) {
                            if entry.visibility == Visibility::Public && entry.id != self.caller_id
                            {
                                self.discovered
                                    .insert((self.caller_id.to_owned(), entry.id.clone()));
                            }
                        }
                    }
                }
            }
        }
        visit::visit_expr_call(self, node);
    }
}

fn extract_call_path(expr: &syn::Expr) -> Option<&syn::Path> {
    match expr {
        syn::Expr::Path(p) => Some(&p.path),
        syn::Expr::Paren(paren) => extract_call_path(&paren.expr),
        syn::Expr::Group(group) => extract_call_path(&group.expr),
        syn::Expr::Reference(reference) => extract_call_path(&reference.expr),
        _ => None,
    }
}

fn resolve_function_path<'a>(
    module_segments: &[String],
    path: &syn::Path,
    functions: &'a HashMap<(String, String), FunctionLookupEntry>,
) -> Option<&'a FunctionLookupEntry> {
    let mut segments: Vec<String> = path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect();
    if segments.is_empty() {
        return None;
    }
    let mut base = if path.leading_colon.is_some() {
        Vec::new()
    } else {
        module_segments.to_vec()
    };
    if let Some(first) = segments.first() {
        match first.as_str() {
            "crate" => {
                base.clear();
                segments.remove(0);
            }
            "self" => {
                base = module_segments.to_vec();
                segments.remove(0);
            }
            "super" => {
                base = module_segments.to_vec();
                while let Some(seg) = segments.first() {
                    if seg == "super" {
                        segments.remove(0);
                        if !base.is_empty() {
                            base.pop();
                        }
                    } else {
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    base.extend(segments);
    if base.is_empty() {
        return None;
    }
    let fn_name = base.pop()?;
    let module_key = if base.is_empty() {
        "crate".to_owned()
    } else {
        base.join("::")
    };
    let slug = slugify(&fn_name);
    functions.get(&(module_key, slug))
}

fn function_name_slug(function: &Function) -> Option<String> {
    function.id.rsplit('.').next().map(|seg| seg.to_owned())
}
