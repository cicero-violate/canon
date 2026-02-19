#![cfg(feature = "rustc_frontend")]

use super::context::FrontendMetadata;
use super::items::capture_adt;
use super::traits::capture_trait;
use crate::state::builder::KernelGraphBuilder;
use crate::state::ids::NodeId;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::ty::{self, TyCtxt, TyKind};
use std::collections::{HashMap, HashSet};

pub(super) fn capture_types_from_function<'tcx>(
    builder: &mut KernelGraphBuilder,
    tcx: TyCtxt<'tcx>,
    local_def: LocalDefId,
    cache: &mut HashMap<DefId, NodeId>,
    metadata: &FrontendMetadata,
) {
    let body = tcx.optimized_mir(local_def);
    let mut seen: HashSet<DefId> = HashSet::new();

    for local_decl in &body.local_decls {
        collect_types_from_ty(tcx, local_decl.ty, &mut seen);
    }

    let def_id = local_def.to_def_id();
    if matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn) {
        let sig = tcx.fn_sig(def_id).skip_binder();
        for input in sig.inputs().skip_binder() {
            collect_types_from_ty(tcx, *input, &mut seen);
        }
        collect_types_from_ty(tcx, sig.output().skip_binder(), &mut seen);
    }

    for ty_def in seen {
        match tcx.def_kind(ty_def) {
            DefKind::Struct | DefKind::Enum | DefKind::Union => {
                capture_adt(builder, tcx, ty_def, cache, metadata);
            }
            DefKind::Trait => {
                capture_trait(builder, tcx, ty_def, cache, metadata);
            }
            _ => {}
        }
    }
}

fn collect_types_from_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: ty::Ty<'tcx>, seen: &mut HashSet<DefId>) {
    use std::cell::RefCell;

    thread_local! {
        static VISITED_TYS: RefCell<HashSet<usize>> = RefCell::new(HashSet::new());
    }

    let ty_ptr = &ty as *const _ as usize;
    let already_seen = VISITED_TYS.with(|v| {
        let mut v = v.borrow_mut();
        if v.contains(&ty_ptr) {
            true
        } else {
            v.insert(ty_ptr);
            false
        }
    });

    if already_seen {
        return;
    }

    match ty.kind() {
        TyKind::Adt(adt_def, _) => {
            seen.insert(adt_def.did());
        }
        TyKind::Dynamic(bounds, _) => {
            use rustc_middle::ty::ExistentialPredicate;

            for pred in bounds.iter() {
                if let ExistentialPredicate::Trait(trait_ref) = pred.skip_binder() {
                    seen.insert(trait_ref.def_id);
                }
            }
        }
        TyKind::FnDef(_, substs) => {
            for arg in substs.iter() {
                if let Some(t) = arg.as_type() {
                    collect_types_from_ty(tcx, t, seen);
                }
            }
        }
        TyKind::FnPtr(sig, _) => {
            for input in sig.inputs().iter() {
                collect_types_from_ty(tcx, *input.skip_binder(), seen);
            }
            collect_types_from_ty(tcx, sig.output().skip_binder(), seen);
        }
        TyKind::Alias(_, alias_ty) => {
            collect_types_from_ty(tcx, alias_ty.to_ty(tcx), seen);
        }
        TyKind::Ref(_, inner, _) | TyKind::Slice(inner) => {
            collect_types_from_ty(tcx, *inner, seen);
        }
        TyKind::Array(inner, _) => {
            collect_types_from_ty(tcx, *inner, seen);
        }
        TyKind::Tuple(list) => {
            for t in list.iter() {
                collect_types_from_ty(tcx, t, seen);
            }
        }
        _ => {}
    }
}
