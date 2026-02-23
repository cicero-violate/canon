#![cfg(feature = "rustc_frontend")]
//! HIR body capture helpers (parameter patterns, generators, etc.).
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::TyCtxt;
use serde::Serialize;
/// Serializes the HIR body owned by the provided definition into JSON.
pub(super) fn encode_hir_body_json<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> Option<String> {
    let body = tcx.hir_maybe_body_owned_by(def_id)?;
    #[derive(Serialize)]
    struct HirBodyCapture {
        params: Vec<String>,
        body: String,
    }
    let params = body.params.iter().map(|param| format!("{:?}", param.pat)).collect();
    let capture = HirBodyCapture { params, body: format!("{:?}", body.value) };
    serde_json::to_string(&capture).ok()
}
