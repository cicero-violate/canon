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
    let Ok((live_symbols, _)) = cx
        .tcx
        .live_symbols_and_ignored_derived_traits(())
        .as_ref()
    else {
        LIVE_SET.with(|slot| slot.borrow_mut().clear());
        LIVE_SIGNATURES.with(|slot| slot.borrow_mut().clear());
        return;
    };

    LIVE_SET.with(|slot| {
        let mut set = slot.borrow_mut();
        set.clear();
        for def_id in cx.tcx.hir_body_owners() {
            if live_symbols.contains(&def_id) {
                set.insert(def_id);
            }
        }
    });

    LIVE_SIGNATURES.with(|slot| {
        let mut map = slot.borrow_mut();
        map.clear();
        for def_id in cx.tcx.hir_body_owners() {
            if !live_symbols.contains(&def_id) || !is_function_like(cx, def_id) {
                continue;
            }
            if let Some((def_path, signature)) = build_signature(cx, def_id) {
                map.insert(def_path, signature);
            }
        }
    });
}

pub fn collect_dead_items<'tcx>(cx: &LateContext<'tcx>) -> Vec<DeadItem> {
    cx.tcx
        .hir_body_owners()
        .filter(|&def_id| is_function_like(cx, def_id) && !is_live(def_id))
        .filter_map(|def_id| build_signature(cx, def_id))
        .map(|(def_path, signature)| DeadItem {
            def_path,
            module_path: signature.module_path.clone(),
            domain_inputs: signature.domain_inputs.clone(),
        })
        .collect()

