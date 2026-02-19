#![cfg(feature = "rustc_frontend")]

use super::context::FrontendMetadata;
use super::metadata;
use crate::rename::core::symbol_id::normalize_symbol_id_with_crate;
use crate::state::builder::{KernelGraphBuilder, NodePayload};
use crate::state::ids::NodeId;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, TyCtxt};
use serde::Serialize;
use std::collections::HashMap;

pub(super) fn capture_adt<'tcx>(
    builder: &mut KernelGraphBuilder,
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    cache: &mut HashMap<DefId, NodeId>,
    metadata: &FrontendMetadata,
) -> NodeId {
    if let Some(&id) = cache.get(&def_id) {
        return id;
    }

    let adt_def = tcx.adt_def(def_id);
    let raw_def_path = tcx.def_path_str(def_id);
    let node_key = format!("{def_id:?}");
    let crate_name = tcx.crate_name(def_id.krate).to_string();
    let def_path = normalize_symbol_id_with_crate(&raw_def_path, Some(&crate_name));

    let mut payload = NodePayload::new(&node_key, def_path.clone())
        .with_metadata("type", "adt")
        .with_metadata("adt_kind", format!("{:?}", adt_def.adt_kind()))
        .with_metadata("type_kind", classify_adt_kind(adt_def))
        .with_metadata("is_struct", adt_def.is_struct().to_string())
        .with_metadata("is_enum", adt_def.is_enum().to_string())
        .with_metadata("is_union", adt_def.is_union().to_string())
        .with_metadata("repr", format!("{:?}", adt_def.repr()));

    if let Some(fields) = serialize_struct_fields(tcx, adt_def) {
        payload = payload.with_metadata("type_fields", fields);
    }

    if let Some(v) = serialize_enum_variants(tcx, adt_def) {
        payload = payload.with_metadata("type_variants", v);
    }

    if def_id.is_local() {
        payload = payload.with_metadata("crate_edition", metadata.edition.clone());
    }

    payload = metadata::apply_common_metadata(payload, tcx, def_id, metadata);

    let node_id = builder.add_node(payload).expect("adt node");
    cache.insert(def_id, node_id);
    node_id
}

pub(super) fn capture_type_alias<'tcx>(
    builder: &mut KernelGraphBuilder,
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    cache: &mut HashMap<DefId, NodeId>,
    metadata: &FrontendMetadata,
) -> NodeId {
    if let Some(&id) = cache.get(&def_id) {
        return id;
    }

    let raw_def_path = tcx.def_path_str(def_id);
    let node_key = format!("{def_id:?}");
    let crate_name = tcx.crate_name(def_id.krate).to_string();
    let def_path = normalize_symbol_id_with_crate(&raw_def_path, Some(&crate_name));
    let aliased_ty = tcx.type_of(def_id).instantiate_identity();

    let mut payload = NodePayload::new(&node_key, def_path.clone())
        .with_metadata("type", "type_alias")
        .with_metadata("type_kind", "type_alias")
        .with_metadata("aliased_type", format!("{:?}", aliased_ty));

    if def_id.is_local() {
        payload = payload.with_metadata("crate_edition", metadata.edition.clone());
    }

    payload = metadata::apply_common_metadata(payload, tcx, def_id, metadata);

    let node_id = builder.add_node(payload).expect("type_alias node");
    cache.insert(def_id, node_id);
    node_id
}

pub(super) fn capture_const_static<'tcx>(
    builder: &mut KernelGraphBuilder,
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    cache: &mut HashMap<DefId, NodeId>,
    metadata: &FrontendMetadata,
) -> NodeId {
    if let Some(&id) = cache.get(&def_id) {
        return id;
    }

    let raw_def_path = tcx.def_path_str(def_id);
    let node_key = format!("{def_id:?}");
    let crate_name = tcx.crate_name(def_id.krate).to_string();
    let def_path = normalize_symbol_id_with_crate(&raw_def_path, Some(&crate_name));
    let const_ty = tcx.type_of(def_id).instantiate_identity();

    let mut payload = NodePayload::new(&node_key, def_path.clone())
        .with_metadata("type", "const_or_static")
        .with_metadata("type_kind", "const_or_static")
        .with_metadata("value_type", format!("{:?}", const_ty));

    if def_id.is_local() {
        payload = payload.with_metadata("crate_edition", metadata.edition.clone());
    }

    payload = metadata::apply_common_metadata(payload, tcx, def_id, metadata);

    let node_id = builder.add_node(payload).expect("const/static node");
    cache.insert(def_id, node_id);
    node_id
}

fn serialize_struct_fields<'tcx>(tcx: TyCtxt<'tcx>, adt_def: ty::AdtDef<'tcx>) -> Option<String> {
    if !adt_def.is_struct() && !adt_def.is_union() {
        return None;
    }

    #[derive(Serialize)]
    struct FieldInfo {
        name: String,
        ty: String,
        vis: String,
        index: usize,
    }

    let variant = adt_def.non_enum_variant();
    let fields: Vec<FieldInfo> = variant
        .fields
        .iter()
        .enumerate()
        .map(|(idx, field)| FieldInfo {
            name: field.name.to_string(),
            ty: format!("{:?}", tcx.type_of(field.did).skip_binder()),
            vis: format!("{:?}", tcx.visibility(field.did)),
            index: idx,
        })
        .collect();

    if fields.is_empty() {
        None
    } else {
        serde_json::to_string(&fields).ok()
    }
}

fn serialize_enum_variants<'tcx>(tcx: TyCtxt<'tcx>, adt_def: ty::AdtDef<'tcx>) -> Option<String> {
    #[derive(Serialize)]
    struct VariantInfo {
        name: String,
        discr: u128,
        fields: Vec<FieldInfo>,
    }

    #[derive(Serialize)]
    struct FieldInfo {
        name: String,
        ty: String,
        vis: String,
        index: usize,
    }

    if !adt_def.is_enum() {
        return None;
    }

    let variants: Vec<VariantInfo> = adt_def
        .variants()
        .iter_enumerated()
        .map(|(idx, variant)| {
            let fields = variant
                .fields
                .iter()
                .enumerate()
                .map(|(idx, field)| FieldInfo {
                    name: field.name.to_string(),
                    ty: format!("{:?}", tcx.type_of(field.did).skip_binder()),
                    vis: format!("{:?}", tcx.visibility(field.did)),
                    index: idx,
                })
                .collect();

            VariantInfo {
                name: variant.name.to_string(),
                discr: adt_def.discriminant_for_variant(tcx, idx).val,
                fields,
            }
        })
        .collect();

    serde_json::to_string(&variants).ok()
}

fn classify_adt_kind(adt_def: ty::AdtDef<'_>) -> String {
    if adt_def.is_enum() {
        "enum".into()
    } else if adt_def.is_struct() {
        "struct".into()
    } else if adt_def.is_union() {
        "union".into()
    } else {
        "adt".into()
    }
}
