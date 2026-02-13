use crate::ir::Visibility;

pub(crate) fn map_visibility(vis: &syn::Visibility) -> Visibility {
    match vis {
        syn::Visibility::Public(_) => Visibility::Public,
        syn::Visibility::Restricted(restricted) => {
            if let Some(path) = &restricted.in_token {
                let _ = path;
            }
            if restricted.path.is_ident("crate") {
                Visibility::PubCrate
            } else if restricted.path.is_ident("super") {
                Visibility::PubSuper
            } else {
                Visibility::Private
            }
        }
        _ => Visibility::Private,
    }
}
