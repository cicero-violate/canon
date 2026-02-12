use rustc_hir::Item;
use rustc_lint::LintContext;
use rustc_lint::{LateContext, LateLintPass};

use crate::classify::classify_item;
use crate::policy::API_TRAITS_ONLY;
use crate::signal::{LINT_SIGNALS, LintSignal};

use rustc_session::declare_lint_pass;

declare_lint_pass!(ApiTraitsOnly => [API_TRAITS_ONLY]);

impl<'tcx> LateLintPass<'tcx> for ApiTraitsOnly {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
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

        // ---- Human-facing lint emission (PROOF) ----
        cx.span_lint(API_TRAITS_ONLY, item.span, |_diag| {
            // emission handled by rustc; closure is for notes/help only
        });

        // Emit judgment signal
        LINT_SIGNALS.lock().unwrap().push(LintSignal {
            policy: "API_TRAITS_ONLY",
            def_path: cx.tcx.def_path_str(item.owner_id.def_id.to_def_id()),
            kind,
            module: "api".into(),
            severity,
        });

        // Signal-only; no human-facing lint emission
    }
}
