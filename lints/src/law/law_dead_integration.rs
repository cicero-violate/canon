use rustc_lint::{LateContext, LintContext, LintStore};
use rustc_session::declare_lint;

declare_lint! {
    pub DEAD_INTEGRATION,
    Warn,
    "item marked #[allow(dead_code)] is a disconnected integration point"
}

pub fn register_law(store: &mut LintStore) {
    store.register_lints(&[&DEAD_INTEGRATION]);
}

pub fn enforce_dead_integration(cx: &LateContext<'_>, item: &rustc_hir::Item<'_>) {
    let attrs = cx.tcx.hir_attrs(item.hir_id());
    let has_allow_dead_code = attrs.iter().any(|attr| {
        attr.meta_item_list()
            .map(|list| {
                list.iter().any(|nested| {
                    nested
                        .meta_item()
                        .map(|m| {
                            m.path
                                .segments
                                .iter()
                                .any(|s| s.ident.name.as_str() == "dead_code")
                        })
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false)
    });

    if !has_allow_dead_code {
        return;
    }

    let name = match &item.kind {
        rustc_hir::ItemKind::Fn { ident, .. } => ident.as_str().to_string(),
        rustc_hir::ItemKind::Struct(ident, ..) => ident.as_str().to_string(),
        rustc_hir::ItemKind::Enum(ident, ..) => ident.as_str().to_string(),
        rustc_hir::ItemKind::Trait(..) => "<trait>".to_string(),
        _ => "<item>".to_string(),
    };

    cx.span_lint(DEAD_INTEGRATION, item.span, |diag| {
        diag.note(format!(
            "dead integration point: `{name}` is suppressed with #[allow(dead_code)] — reconnect it",
        ));
    });
}
