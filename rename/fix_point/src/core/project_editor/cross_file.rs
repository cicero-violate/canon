use super::utils::find_project_root_sync;


use super::QueuedOp;


use crate::core::paths::module_path_for_file;


use crate::core::symbol_id::normalize_symbol_id;


use crate::module_path::{compute_new_file_path, ModulePath};


use crate::state::NodeRegistry;


use anyhow::Result;


use std::collections::HashSet;


use std::path::PathBuf;


use std::sync::Arc;


use syn::spanned::Spanned;


use syn::visit_mut::VisitMut;


fn build_super_tree(tail: &syn::UseTree) -> syn::UseTree {
    let super_ident = syn::Ident::new("super", proc_macro2::Span::call_site());
    syn::UseTree::Path(syn::UsePath {
        ident: super_ident,
        colon2_token: syn::token::PathSep::default(),
        tree: Box::new(tail.clone()),
    })
}


fn ensure_source_loaded(registry: &mut NodeRegistry, file: &PathBuf) -> Result<()> {
    if registry.sources.contains_key(file) {
        return Ok(());
    }
    if file.exists() {
        let content = std::fs::read_to_string(file)?;
        registry.sources.insert(file.clone(), Arc::new(content));
        return Ok(());
    }
    registry.sources.insert(file.clone(), Arc::new(String::new()));
    Ok(())
}


fn pub_crate() -> syn::Visibility {
    syn::parse_quote!(pub (crate))
}


fn superize_use_if_from_src(item: syn::ItemUse, src_crate_path: &str) -> syn::ItemUse {
    let rewritten = superize_tree(item.tree.clone(), src_crate_path);
    syn::ItemUse {
        tree: rewritten,
        ..item
    }
}
