use anyhow::Result;
use prettyplease;
use quote::ToTokens;
use std::path::Path;

use crate::rename::rewrite::RewriteBufferSet;

pub fn render_node<T: ToTokens>(node: T) -> String {
    let tokens = quote::quote! { #node };
    match syn::parse_file(&tokens.to_string()) {
        Ok(file) => prettyplease::unparse(&file),
        Err(_) => tokens.to_string(),
    }
}

pub fn render_function(func: &syn::ItemFn) -> String {
    render_node(func)
}

pub fn render_impl(item: &syn::ItemImpl) -> String {
    render_node(item)
}

pub fn render_struct(item: &syn::ItemStruct) -> String {
    render_node(item)
}

pub fn render_trait(item: &syn::ItemTrait) -> String {
    render_node(item)
}

pub fn replace_with_node<T: ToTokens>(
    buffers: &mut RewriteBufferSet,
    file: &Path,
    content: &str,
    start: usize,
    end: usize,
    node: &T,
) -> Result<()> {
    let rendered = render_node(node);
    let buffer = buffers.ensure_buffer(file, content);
    buffer.replace(start, end, rendered)?;
    Ok(())
}

pub fn insert_node<T: ToTokens>(
    buffers: &mut RewriteBufferSet,
    file: &Path,
    content: &str,
    offset: usize,
    node: &T,
) -> Result<()> {
    let rendered = render_node(node);
    let buffer = buffers.ensure_buffer(file, content);
    buffer.insert(offset, rendered)?;
    Ok(())
}
