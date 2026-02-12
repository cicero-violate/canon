use rustc_hir::ItemKind;

pub fn classify_item(kind: &ItemKind<'_>) -> (&'static str, f32) {
    match kind {
        ItemKind::Struct(..) => ("struct", 0.9),
        ItemKind::Enum(..) => ("enum", 0.9),
        ItemKind::Const(..) => ("const", 0.7),
        ItemKind::TyAlias(..) => ("type", 0.4),
        ItemKind::Trait(..) => ("trait", 0.0),
        _ => ("other", 0.2),
    }
}
