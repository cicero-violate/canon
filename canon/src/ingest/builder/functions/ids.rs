use super::super::types::slugify;

pub(crate) fn trait_path_to_trait_id(
    path: &syn::Path,
    module_id: &str,
    trait_name_to_id: &std::collections::HashMap<String, String>,
) -> String {
    let trait_name = path
        .segments
        .last()
        .map(|seg| seg.ident.to_string())
        .unwrap_or_else(|| "Trait".to_owned());
    let name_key = trait_name.to_ascii_lowercase();
    if let Some(id) = trait_name_to_id.get(&name_key) {
        id.clone()
    } else {
        format!("trait.{}.{}", slugify(module_id), slugify(&trait_name))
    }
}

pub(crate) fn trait_path_to_trait_fn_id(
    trait_path: &syn::Path,
    module_id: &str,
    fn_ident: &syn::Ident,
    trait_name_to_id: &std::collections::HashMap<String, String>,
) -> String {
    let trait_id = trait_path_to_trait_id(trait_path, module_id, trait_name_to_id);
    format!("trait_fn.{}.{}", trait_id, slugify(&fn_ident.to_string()))
}

pub(crate) fn type_path_to_struct_id(
    path: &syn::TypePath,
    module_id: &str,
    type_slug_to_id: &std::collections::HashMap<String, String>,
) -> String {
    let struct_name = path
        .path
        .segments
        .last()
        .map(|seg| seg.ident.to_string())
        .unwrap_or_else(|| "Struct".to_owned());
    let slug = slugify(&struct_name);
    type_slug_to_id
        .get(slug.as_str())
        .cloned()
        .unwrap_or_else(|| format!("struct.{}.{}", slugify(module_id), slug))
}
