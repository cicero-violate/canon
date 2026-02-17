use std::sync::Mutex;

use lazy_static::lazy_static;
use rustc_hir::{Item, ItemKind};
use rustc_lint::LateContext;

lazy_static! {
    /// Accumulates (def_path, kind, module_path) for unreachable items.
    pub static ref DEAD_ITEMS: Mutex<Vec<DeadItem>> = Mutex::new(Vec::new());
}

pub struct DeadItem {
    pub def_path: String,
    pub kind: String,
    pub module_path: String,
    pub reachable_caller: String,
}

pub fn reset_reachability() {
    DEAD_ITEMS.lock().unwrap().clear();
}

/// Check if item is reachable from main via tcx.reachable_set().
/// If not reachable, find the closest reachable item in the same module
/// and record it as the reconnect target.
pub fn collect_dead_item<'tcx>(cx: &LateContext<'tcx>, item: &Item<'tcx>) {
    let def_id = item.owner_id.def_id;

    // Skip items with no meaningful body or identity.
    match item.kind {
        ItemKind::Fn { .. }
        | ItemKind::Struct(..)
        | ItemKind::Enum(..)
        | ItemKind::Trait(..)
        | ItemKind::Impl(..) => {}
        _ => return,
    }

    // tcx.reachable_set() returns LocalDefId set of all reachable items.
    let reachable_set = cx.tcx.reachable_set(());

    if reachable_set.contains(&def_id) {
        return;
    }

    let def_path = cx.tcx.def_path_str(def_id.to_def_id());
    let def_path_clone = def_path.clone();
    let module_path = module_of(&def_path_clone);

    // Find closest reachable item in same module by walking all local def ids.
    let reachable_caller = cx.tcx
        .hir_body_owners()
        .filter_map(|local_def_id| {
            if !reachable_set.contains(&local_def_id) {
                return None;
            }
            let p = cx.tcx.def_path_str(local_def_id.to_def_id());
            if module_of(&p) == module_path {
                Some(p)
            } else {
                None
            }
        })
        .next()
        .unwrap_or_else(|| "<no reachable peer in module>".to_string());

    let kind = match item.kind {
        ItemKind::Fn { .. } => "fn",
        ItemKind::Struct(..) => "struct",
        ItemKind::Enum(..) => "enum",
        ItemKind::Trait(..) => "trait",
        ItemKind::Impl(..) => "impl",
        _ => "other",
    };

    DEAD_ITEMS.lock().unwrap().push(DeadItem {
        def_path,
        kind: kind.to_string(),
        module_path: module_path.to_string(),
        reachable_caller,
    });
}

fn module_of(def_path: &str) -> &str {
    def_path.rsplit_once("::").map(|(m, _)| m).unwrap_or(def_path)
}
