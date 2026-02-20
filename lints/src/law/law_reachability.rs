use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty, TyCtxt};

thread_local! {
    static LIVE_SET: RefCell<HashSet<LocalDefId>> = RefCell::new(HashSet::new());
    static LIVE_SIGNATURES: RefCell<HashMap<String, FnSignature>> = RefCell::new(HashMap::new());
}

const NOISE_TYPES: &[&str] = &["String", "&str", "bool", "u64", "usize", "f64", "f32", "i32", "()", "std::string::String"];
const NOISE_DEF_FRAGMENTS: &[&str] = &["as std::clone", "as std::fmt", "as std::error", "as std::convert", "as std::default", "as std::ops"];

#[derive(Clone)]
pub struct DeadItem {
    pub def_path: String,
    pub module_path: String,
    pub domain_inputs: Vec<String>,
}

#[derive(Clone)]
pub struct FnSignature {
    pub module_path: String,
    pub domain_inputs: Vec<String>,
}

struct MatchResult {
    path: String,
    score: f32,
    module: String,
}
pub fn reset_reachability(cx: &LateContext<'_>) {
    let Ok((live_symbols, _)) = cx.tcx.live_symbols_and_ignored_derived_traits(()).as_ref() else {
        LIVE_SET.with(|slot| slot.borrow_mut().clear());
        LIVE_SIGNATURES.with(|slot| slot.borrow_mut().clear());
        return;
    };

    LIVE_SET.with(|slot| {
        let mut set = slot.borrow_mut();
        set.clear();
        set.extend(cx.tcx.hir_body_owners().filter(|id| live_symbols.contains(id)));
    });

    LIVE_SIGNATURES.with(|slot| {
        let mut map = slot.borrow_mut();
        map.clear();
        for def_id in cx.tcx.hir_body_owners() {
            if !live_symbols.contains(&def_id) || !is_function_like(cx, def_id) {
                continue;
            }
            if let Some((def_path, sig)) = build_signature(cx, def_id) {
                if is_noise_def(&def_path) {
                    continue;
                }
                map.insert(def_path, sig);
            }
        }
    });
}

pub fn collect_dead_items<'tcx>(cx: &LateContext<'tcx>) -> Vec<DeadItem> {
    cx.tcx
        .hir_body_owners()
        .filter(|&def_id| is_function_like(cx, def_id) && !is_live(def_id))
        .filter_map(|def_id| build_signature(cx, def_id))
        .filter(|(def_path, _)| !is_noise_def(def_path))
        .map(|(def_path, sig)| DeadItem { def_path, module_path: sig.module_path.clone(), domain_inputs: sig.domain_inputs.clone() })
        .collect()
}

pub fn best_reconnect_target(dead: &DeadItem) -> (String, f32) {
    let result = LIVE_SIGNATURES.with(|slot| {
        let map = slot.borrow();
        if map.is_empty() {
            return None;
        }

        let first_pass: Vec<_> = map.iter().filter(|(_, sig)| sig.module_path == dead.module_path).collect();

        if let Some(best) = select_best(dead, first_pass) {
            if best.1 > 0.0 {
                return Some(best);
            }
        }

        if let Some(parent) = parent_module(&dead.module_path) {
            let second_pass: Vec<_> = map.iter().filter(|(_, sig)| sig.module_path.starts_with(parent) && shares_module_tokens(dead, sig)).collect();

            if let Some(best) = select_best(dead, second_pass) {
                if best.1 > 0.0 {
                    return Some(best);
                }
            }
        }

        let fallback: Vec<_> = map.iter().filter(|(_, sig)| compatibility(dead, sig) > 0.0).collect();
        select_best(dead, fallback)
    });

    if let Some(res) = result {
        res
    } else {
        ("<no compatible live function>".into(), 0.0)
    }
}

fn select_best<'a>(dead: &DeadItem, candidates: Vec<(&'a String, &'a FnSignature)>) -> Option<(String, f32)> {
    if candidates.is_empty() {
        return None;
    }
    candidates
        .into_iter()
        .map(|(path, sig)| {
            let score = compatibility(dead, sig);
            MatchResult { path: path.clone(), score, module: sig.module_path.clone() }
        })
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal).then_with(|| b.module.len().cmp(&a.module.len())))
        .map(|res| (res.path, res.score))
}

fn is_live(def_id: LocalDefId) -> bool {
    LIVE_SET.with(|slot| slot.borrow().contains(&def_id))
}

fn is_function_like<'tcx>(cx: &LateContext<'tcx>, def_id: LocalDefId) -> bool {
    matches!(cx.tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn)
}

fn build_signature<'tcx>(cx: &LateContext<'tcx>, def_id: LocalDefId) -> Option<(String, FnSignature)> {
    let def_path = cx.tcx.def_path_str(def_id.to_def_id());
    let module_path = module_of(&def_path).to_string();
    let sig = cx.tcx.fn_sig(def_id.to_def_id()).skip_binder();
    let inputs = domain_input_types(cx.tcx, sig.inputs().skip_binder());
    Some((def_path, FnSignature { module_path, domain_inputs: inputs }))
}

fn domain_input_types<'tcx>(tcx: TyCtxt<'tcx>, inputs: &[Ty<'tcx>]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for ty in inputs {
        collect_domain_types(tcx, *ty, &mut seen, &mut out);
    }
    out
}

fn collect_domain_types<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, seen: &mut HashSet<String>, out: &mut Vec<String>) {
    let ty = ty.peel_refs();
    match ty.kind() {
        ty::Adt(adt, _) => record_domain_type(tcx, adt.did(), seen, out),
        ty::Array(elem, _) | ty::Slice(elem) => collect_domain_types(tcx, *elem, seen, out),
        ty::Tuple(elems) => {
            for elem in elems.iter() {
                collect_domain_types(tcx, elem, seen, out);
            }
        }
        ty::Alias(_, alias) => record_domain_type(tcx, alias.def_id, seen, out),
        ty::FnDef(def_id, _) => {
            if def_id.is_local() {
                record_path(tcx, *def_id, seen, out);
            }
        }
        _ => {}
    }
}

fn record_domain_type<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, seen: &mut HashSet<String>, out: &mut Vec<String>) {
    if def_id.is_local() {
        record_path(tcx, def_id, seen, out);
    }
}

fn record_path(tcx: TyCtxt<'_>, def_id: DefId, seen: &mut HashSet<String>, out: &mut Vec<String>) {
    let name = tcx.def_path_str(def_id);
    if is_noise_type(&name) {
        return;
    }
    if seen.insert(name.clone()) {
        out.push(name);
    }
}

fn compatibility(dead: &DeadItem, live: &FnSignature) -> f32 {
    if dead.domain_inputs.is_empty() {
        return 0.0;
    }
    let live_set: HashSet<&str> = live.domain_inputs.iter().map(|t| t.as_str()).collect();
    let matches = dead.domain_inputs.iter().filter(|t| live_set.contains(t.as_str())).count();
    matches as f32 / (dead.domain_inputs.len() as f32 + 1.0)
}

fn shares_module_tokens(dead: &DeadItem, sig: &FnSignature) -> bool {
    let dead_tokens: Vec<&str> = dead.module_path.split("::").collect();
    sig.domain_inputs.iter().any(|ty| dead_tokens.iter().all(|tok| ty.contains(tok)))
}

fn parent_module(module: &str) -> Option<&str> {
    module.rsplit_once("::").map(|(parent, _)| parent)
}

fn module_of(def_path: &str) -> &str {
    def_path.rsplit_once("::").map(|(module, _)| module).unwrap_or(def_path)
}

fn is_noise_type(name: &str) -> bool {
    NOISE_TYPES.iter().any(|noise| name.ends_with(noise))
}

fn is_noise_def(def_path: &str) -> bool {
    NOISE_DEF_FRAGMENTS.iter().any(|frag| def_path.contains(frag))
}
