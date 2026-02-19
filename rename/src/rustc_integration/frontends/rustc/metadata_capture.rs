#![cfg(feature = "rustc_frontend")]
//! Attribute, visibility, and generics metadata helpers.
use super::frontend_context::FrontendMetadata;
use crate::rename::core::symbol_id::normalize_symbol_id_with_crate;
use crate::state::builder::NodePayload;
use blake3::hash;
use rustc_hir::{def::DefKind, def_id::DefId, Attribute as HirAttribute};
use rustc_middle::ty::{GenericParamDefKind, TyCtxt};
use rustc_span::{symbol::sym, FileName, SourceFile};
use serde::Serialize;
use std::path::Path;
/// Enriches a node payload with common compiler metadata.
pub(super) fn apply_common_metadata<'tcx>(
    mut payload: NodePayload,
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    frontend: &FrontendMetadata,
) -> NodePayload {
    let crate_name = tcx.crate_name(def_id.krate).to_string();
    let def_path = normalize_symbol_id_with_crate(
        &tcx.def_path_str(def_id),
        Some(&crate_name),
    );
    let name = tcx
        .opt_item_name(def_id)
        .map(|sym| sym.to_string())
        .or_else(|| Some(def_path.split("::").last()?.to_string()))
        .unwrap_or_else(|| format!("{def_id:?}"));
    let module = module_path_from_def(&def_path);
    let parent = tcx.opt_parent(def_id);
    let def_kind = tcx.def_kind(def_id);
    let def_path_hash = tcx.def_path_hash(def_id);
    let source_map = tcx.sess.source_map();
    let span = tcx.def_span(def_id);
    let filename = source_map.span_to_filename(span);
    let lo = source_map.lookup_char_pos(span.lo());
    let hi = source_map.lookup_char_pos(span.hi());
    let absolute_path = resolve_absolute_path(&filename);
    let relative_path = resolve_relative_path(
        &absolute_path,
        frontend.workspace_root.as_deref(),
    );
    let file_stats = compute_source_file_stats(&lo.file);
    let snippet = source_map.span_to_snippet(span).unwrap_or_else(|_| String::new());
    payload = payload
        .with_metadata("def_path", def_path.clone())
        .with_metadata("def_path_hash", format!("{def_path_hash:?}"))
        .with_metadata("name", name.clone())
        .with_metadata("display", format_symbol_display(&module, &name))
        .with_metadata("module", module.clone())
        .with_metadata(
            "module_id",
            if module.is_empty() { "mod:root".into() } else { format!("mod:{module}") },
        );
    if let Some(relative) = relative_path {
        payload = payload.with_metadata("source_file_relative", relative);
    }
    payload = payload
        .with_metadata("crate", format!("crate:{}", tcx.crate_name(def_id.krate)))
        .with_metadata("node_kind", node_kind_label(def_kind))
        .with_metadata("symbol_kind", symbol_kind_label(def_kind))
        .with_metadata("source_file", absolute_path.clone())
        .with_metadata("line", lo.line.to_string())
        .with_metadata("column", lo.col.0.to_string())
        .with_metadata("span_end_line", hi.line.to_string())
        .with_metadata("span_end_column", hi.col.0.to_string())
        .with_metadata("source_snippet", snippet)
        .with_metadata("source_file_line_count", file_stats.line_count.to_string())
        .with_metadata("source_file_byte_len", file_stats.byte_len.to_string())
        .with_metadata("visibility", format!("{:?}", tcx.visibility(def_id)))
        .with_metadata("crate_type", frontend.crate_type.clone())
        .with_metadata("target_triple", frontend.target_triple.clone());
    if let Some(ident_span) = tcx.def_ident_span(def_id) {
        let ident_lo = source_map.lookup_char_pos(ident_span.lo());
        let ident_hi = source_map.lookup_char_pos(ident_span.hi());
        payload = payload
            .with_metadata("ident_line", ident_lo.line.to_string())
            .with_metadata("ident_column", ident_lo.col.0.to_string())
            .with_metadata("ident_end_line", ident_hi.line.to_string())
            .with_metadata("ident_end_column", ident_hi.col.0.to_string());
    }
    if let Some(hash) = file_stats.hash {
        payload = payload.with_metadata("source_file_hash", hash);
    }
    if let Some(parent_id) = parent {
        payload = payload.with_metadata("parent", format!("{parent_id:?}"));
        payload = payload
            .with_metadata("parent_kind", node_kind_label(tcx.def_kind(parent_id)));
        payload = payload
            .with_metadata(
                "parent_def_path_hash",
                format!("{:?}", tcx.def_path_hash(parent_id)),
            );
        payload = payload
            .with_metadata("container_kind", node_kind_label(tcx.def_kind(parent_id)));
    }
    payload = payload.with_metadata("crate_edition", frontend.edition.clone());
    if let Some(target_name) = &frontend.target_name {
        payload = payload.with_metadata("crate_target_name", target_name.clone());
    }
    if let Some(rust_version) = &frontend.rust_version {
        payload = payload.with_metadata("crate_rust_version", rust_version.clone());
    }
    if let Some(version) = &frontend.package_version {
        payload = payload.with_metadata("crate_version", version.clone());
    }
    if let Some(package_name) = &frontend.package_name {
        payload = payload.with_metadata("package_name", package_name.clone());
    }
    if let Some(root) = &frontend.workspace_root {
        payload = payload.with_metadata("workspace_root", root.clone());
    }
    if !frontend.package_features.is_empty() {
        if let Ok(json) = serde_json::to_string(&frontend.package_features) {
            payload = payload.with_metadata("crate_features", json);
        }
    }
    if !frontend.cfg_flags.is_empty() {
        if let Ok(json) = serde_json::to_string(&frontend.cfg_flags) {
            payload = payload.with_metadata("crate_cfg", json);
        }
    }
    if let Some(attr_data) = capture_attributes(tcx, def_id) {
        payload = payload.with_metadata("attributes", attr_data.raw);
        payload = apply_attribute_flags(payload, attr_data.flags);
    }
    if let Some(generics_json) = serialize_generic_params(tcx, def_id) {
        payload = payload.with_metadata("generics", generics_json);
    }
    payload
}
fn capture_attributes(tcx: TyCtxt<'_>, def_id: DefId) -> Option<AttributeData> {
    let local = def_id.as_local()?;
    let hir_id = tcx.local_def_id_to_hir_id(local);
    let attrs = tcx.hir_attrs(hir_id);
    if attrs.is_empty() {
        return None;
    }
    #[derive(Serialize)]
    struct AttributeCapture {
        path: Vec<String>,
        doc: Option<String>,
        args: Option<String>,
    }
    let mut flags = AttributeFlags::default();
    let mut doc_lines = Vec::new();
    let captures: Vec<AttributeCapture> = attrs
        .iter()
        .map(|attr| {
            if let Some(doc) = attr.doc_str() {
                doc_lines.push(doc.to_string());
            }
            flags.apply(attr);
            AttributeCapture {
                path: attribute_path(attr),
                doc: attr.doc_str().map(|sym| sym.to_string()),
                args: attribute_args(attr),
            }
        })
        .collect();
    if !doc_lines.is_empty() {
        flags.doc = Some(doc_lines.join("\n"));
    }
    serde_json::to_string(&captures).ok().map(|raw| AttributeData { raw, flags })
}
fn attribute_path(attr: &HirAttribute) -> Vec<String> {
    attr.path().into_iter().map(|sym| sym.to_string()).collect()
}
fn attribute_args(attr: &HirAttribute) -> Option<String> {
    if let Some(items) = attr.meta_item_list() {
        let rendered: Vec<String> = items
            .iter()
            .filter_map(|item| item.ident().map(|ident| ident.name.to_string()))
            .collect();
        if !rendered.is_empty() {
            return Some(rendered.join(", "));
        }
    }
    attr.value_str().map(|value| value.to_string())
}
fn serialize_generic_params(tcx: TyCtxt<'_>, def_id: DefId) -> Option<String> {
    let generics = tcx.generics_of(def_id);
    let params: Vec<GenericParamCapture> = generics
        .own_params
        .iter()
        .map(|param| GenericParamCapture {
            index: param.index,
            name: param.name.to_string(),
            kind: format_generic_param_kind(&param.kind),
            has_default: generic_param_has_default(&param.kind),
        })
        .collect();
    let predicates = serialize_where_predicates(tcx, def_id).unwrap_or_default();
    if params.is_empty() && predicates.is_empty() && generics.parent.is_none() {
        return None;
    }
    let capture = GenericsCapture {
        params,
        parent: generics.parent.map(|parent| tcx.def_path_str(parent)),
        predicates,
    };
    serde_json::to_string(&capture).ok()
}
fn serialize_where_predicates(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Vec<String>> {
    let predicates = tcx.predicates_of(def_id).instantiate_identity(tcx);
    if predicates.predicates.is_empty() {
        return None;
    }
    Some(
        predicates.predicates.iter().map(|predicate| format!("{predicate:?}")).collect(),
    )
}
fn format_generic_param_kind(kind: &GenericParamDefKind) -> String {
    match kind {
        GenericParamDefKind::Lifetime => "lifetime".into(),
        GenericParamDefKind::Type { .. } => "type".into(),
        GenericParamDefKind::Const { .. } => "const".into(),
    }
}
fn generic_param_has_default(kind: &GenericParamDefKind) -> bool {
    match kind {
        GenericParamDefKind::Lifetime => false,
        GenericParamDefKind::Type { has_default, .. } => *has_default,
        GenericParamDefKind::Const { has_default, .. } => *has_default,
    }
}
#[derive(Serialize)]
struct GenericsCapture {
    params: Vec<GenericParamCapture>,
    parent: Option<String>,
    predicates: Vec<String>,
}
#[derive(Serialize)]
struct GenericParamCapture {
    index: u32,
    name: String,
    kind: String,
    has_default: bool,
}
struct AttributeData {
    raw: String,
    flags: AttributeFlags,
}
#[derive(Default)]
struct AttributeFlags {
    inline: Option<bool>,
    inline_hint: Option<String>,
    test: bool,
    bench: bool,
    no_mangle: bool,
    doc_hidden: bool,
    thread_local: bool,
    reprs: Vec<String>,
    derives: Vec<String>,
    doc: Option<String>,
}
impl AttributeFlags {
    fn apply(&mut self, attr: &HirAttribute) {
        if attr.has_name(sym::inline) {
            self.inline = Some(true);
            if let Some(items) = attr.meta_item_list() {
                if let Some(ident) = items.iter().find_map(|item| item.ident()) {
                    self.inline_hint = Some(ident.name.to_string());
                }
            }
        } else if attr.has_name(sym::test) {
            self.test = true;
        } else if attr.has_name(sym::bench) {
            self.bench = true;
        } else if attr.has_name(sym::no_mangle) {
            self.no_mangle = true;
        } else if attr.has_name(sym::thread_local) {
            self.thread_local = true;
        } else if attr.has_name(sym::derive) {
            if let Some(items) = attr.meta_item_list() {
                for item in items.into_iter() {
                    if let Some(ident) = item.ident() {
                        self.derives.push(ident.name.to_string());
                    }
                }
            }
        } else if attr.has_name(sym::repr) {
            if let Some(items) = attr.meta_item_list() {
                for item in items.into_iter() {
                    if let Some(ident) = item.ident() {
                        self.reprs.push(ident.name.to_string());
                    } else if let Some(value) = item.value_str() {
                        self.reprs.push(value.to_string());
                    }
                }
            }
        } else if attr.has_name(sym::doc) {
            if let Some(value) = attr.value_str() {
                if value.as_str() == "hidden" {
                    self.doc_hidden = true;
                }
            }
            if let Some(items) = attr.meta_item_list() {
                for item in items {
                    if let Some(ident) = item.ident() {
                        if ident.name == sym::hidden {
                            self.doc_hidden = true;
                        }
                    }
                }
            }
        }
    }
}
fn apply_attribute_flags(
    mut payload: NodePayload,
    flags: AttributeFlags,
) -> NodePayload {
    if let Some(inline) = flags.inline {
        payload = payload.with_metadata("inline", inline.to_string());
    }
    if let Some(hint) = flags.inline_hint {
        payload = payload.with_metadata("inline_hint", hint);
    }
    if flags.test {
        payload = payload.with_metadata("test", "true");
    }
    if flags.bench {
        payload = payload.with_metadata("bench", "true");
    }
    if flags.no_mangle {
        payload = payload.with_metadata("no_mangle", "true");
    }
    if flags.doc_hidden {
        payload = payload.with_metadata("doc_hidden", "true");
    }
    if let Some(doc) = flags.doc {
        payload = payload.with_metadata("doc", doc);
    }
    if flags.thread_local {
        payload = payload.with_metadata("thread_local", "true");
    }
    if !flags.reprs.is_empty() {
        if let Ok(json) = serde_json::to_string(&flags.reprs) {
            payload = payload.with_metadata("repr", json);
        }
    }
    if !flags.derives.is_empty() {
        if let Ok(json) = serde_json::to_string(&flags.derives) {
            payload = payload.with_metadata("derive", json);
        }
    }
    payload
}
struct FileStats {
    hash: Option<String>,
    line_count: usize,
    byte_len: usize,
}
fn compute_source_file_stats(file: &SourceFile) -> FileStats {
    let hash_value = if let Some(src) = &file.src {
        let digest = hash(src.as_bytes());
        Some(format!("blake3:{}", digest.to_hex()))
    } else {
        Some(format!("{:?}", file.src_hash))
    };
    let line_count = file.count_lines();
    let byte_len = file
        .src
        .as_ref()
        .map(|src| src.len())
        .unwrap_or(file.unnormalized_source_len as usize);
    FileStats {
        hash: hash_value,
        line_count,
        byte_len,
    }
}
fn resolve_absolute_path(filename: &FileName) -> String {
    if let Some(path) = filename.clone().into_local_path() {
        path.display().to_string()
    } else {
        filename.prefer_local_unconditionally().to_string()
    }
}
fn resolve_relative_path(path: &str, workspace_root: Option<&str>) -> Option<String> {
    let root = workspace_root?;
    let abs = Path::new(path);
    let root_path = Path::new(root);
    abs.strip_prefix(root_path).ok().map(|p| p.to_string_lossy().to_string())
}
fn node_kind_label(def_kind: DefKind) -> String {
    let kind = match def_kind {
        DefKind::Mod => "module",
        DefKind::Struct => "struct",
        DefKind::Enum => "enum",
        DefKind::Union => "union",
        DefKind::Trait => "trait",
        DefKind::Impl { .. } => "impl",
        DefKind::Fn => "function",
        DefKind::AssocFn => "method",
        DefKind::Const => "const",
        DefKind::Static { .. } => "static",
        DefKind::AssocConst => "const",
        DefKind::TyAlias => "type_alias",
        DefKind::Variant => "variant",
        _ => "unknown",
    };
    kind.into()
}
fn module_path_from_def(def_path: &str) -> String {
    def_path.rsplitn(2, "::").nth(1).unwrap_or("").to_string()
}
fn format_symbol_display(module: &str, name: &str) -> String {
    if module.is_empty() { name.to_string() } else { format!("{module}::{name}") }
}
fn symbol_kind_label(def_kind: DefKind) -> String {
    let kind = match def_kind {
        DefKind::Mod => "module",
        DefKind::Struct | DefKind::Enum | DefKind::Union | DefKind::Trait => "type",
        DefKind::Fn | DefKind::AssocFn | DefKind::AssocConst | DefKind::Variant => {
            "value"
        }
        DefKind::Const | DefKind::Static { .. } | DefKind::AnonConst => "value",
        DefKind::TyAlias => "type",
        _ => "value",
    };
    kind.into()
}
