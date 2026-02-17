use rustc_hir::Item;
use rustc_lint::LintContext;
use rustc_lint::{LateContext, LateLintPass};
use serde_json;

use crate::classify::classify_item;
use crate::law::{
    best_reconnect_target, collect_dead_items, reset_reachability, DEAD_INTEGRATION, FILE_TOO_LONG,
    enforce_dead_integration, enforce_file_length, reset_cache,
};
use crate::policy::API_TRAITS_ONLY;
use crate::signal::{LINT_SIGNALS, LintSignal};

use rustc_session::declare_lint_pass;

declare_lint_pass!(ApiTraitsOnly => [API_TRAITS_ONLY, FILE_TOO_LONG, DEAD_INTEGRATION]);

impl<'tcx> LateLintPass<'tcx> for ApiTraitsOnly {
    fn check_crate(&mut self, cx: &LateContext<'tcx>) {
        reset_cache();
        reset_reachability(cx);
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        enforce_file_length(cx, item.span);
        enforce_dead_integration(cx, item);
        // Only public items
        let vis = cx.tcx.visibility(item.owner_id.def_id);
        if !vis.is_public() {
            return;
        }

        // Find enclosing module via def path
        let def_path = cx.tcx.def_path(item.owner_id.def_id.to_def_id());
        // Require that *any* enclosing path component is `api`
        let in_api = def_path.data.iter().any(|d| match d.data {
            rustc_hir::definitions::DefPathData::TypeNs(sym) => sym.as_str() == "api",
            _ => false,
        });

        if !in_api {
            return;
        }

        // Ignore the module item itself
        if matches!(item.kind, rustc_hir::ItemKind::Mod(..)) {
            return;
        }

        // Allow traits without emitting signals or lints
        if matches!(item.kind, rustc_hir::ItemKind::Trait(..)) {
            return;
        }

        // Classify item
        let (kind, severity) = classify_item(&item.kind);
        let signal = LintSignal {
            policy: "API_TRAITS_ONLY".into(),
            def_path: cx.tcx.def_path_str(item.owner_id.def_id.to_def_id()),
            kind: kind.to_string(),
            module: "api".into(),
            severity,
        };

        let signal_json = serde_json::to_string_pretty(&signal)
            .unwrap_or_else(|err| format!(r#"{{"serialization_error":"{err}"}}"#));

        // ---- Human-facing lint emission (PROOF) ----
        cx.span_lint(API_TRAITS_ONLY, item.span, |diag| {
            diag.note(format!(
                "Canon lint signal (also stored under canon_store/lint_signals):\n{}",
                signal_json
            ));
        });

        // Emit judgment signal
        LINT_SIGNALS.lock().unwrap().push(signal);

        // Signal-only; no human-facing lint emission
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        let dead_items = collect_dead_items(cx);
        for item in dead_items.iter() {
            let (reconnect, score) = best_reconnect_target(item);
            let signal = crate::signal::LintSignal {
                policy: "DEAD_INTEGRATION".into(),
                def_path: item.def_path.clone(),
                kind: "fn".into(),
                module: reconnect,
                severity: score,
            };
            crate::signal::LINT_SIGNALS.lock().unwrap().push(signal);
        }
    }
}
