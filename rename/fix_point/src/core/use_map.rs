use std::collections::HashMap;


pub(crate) fn build_use_map(
    ast: &syn::File,
    module_path: &str,
) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for item in &ast.items {
        let syn::Item::Use(u) = item else { continue };
        let mut prefix = Vec::new();
        if u.leading_colon.is_some() {
            prefix.push("crate".to_string());
        }
        use_tree_to_map(&u.tree, &mut prefix, module_path, &mut map);
    }
    map
}


pub(crate) fn type_path_string(ty: &syn::Type, module_path: &str) -> String {
    if let syn::Type::Path(tp) = ty {
        path_to_string(&tp.path, module_path)
    } else {
        format!("{}::{}", module_path, "Self")
    }
}
