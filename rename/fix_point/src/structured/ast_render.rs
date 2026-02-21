pub fn render_node<T: ToTokens>(node: T) -> String {
    let tokens = quote::quote! {
        # node
    };
    match syn::parse_file(&tokens.to_string()) {
        Ok(file) => prettyplease::unparse(&file),
        Err(_) => tokens.to_string(),
    }
}


pub fn render_node<T: ToTokens>(node: T) -> String {
    let tokens = quote::quote! {
        # node
    };
    match syn::parse_file(&tokens.to_string()) {
        Ok(file) => prettyplease::unparse(&file),
        Err(_) => tokens.to_string(),
    }
}


pub fn render_file(ast: &syn::File) -> String {
    prettyplease::unparse(ast)
}


pub fn render_file(ast: &syn::File) -> String {
    prettyplease::unparse(ast)
}


pub fn render_fn_item(func: &syn::ItemFn) -> String {
    render_node(func)
}


pub fn render_fn_item(func: &syn::ItemFn) -> String {
    render_node(func)
}


pub fn render_impl_item(item: &syn::ItemImpl) -> String {
    render_node(item)
}


pub fn render_impl_item(item: &syn::ItemImpl) -> String {
    render_node(item)
}


pub fn render_struct(item: &syn::ItemStruct) -> String {
    render_node(item)
}


pub fn render_struct(item: &syn::ItemStruct) -> String {
    render_node(item)
}


pub fn render_trait(item: &syn::ItemTrait) -> String {
    render_node(item)
}


pub fn render_trait(item: &syn::ItemTrait) -> String {
    render_node(item)
}


pub fn render_node<T: ToTokens>(node: T) -> String {
    let tokens = quote::quote! {
        # node
    };
    match syn::parse_file(&tokens.to_string()) {
        Ok(file) => prettyplease::unparse(&file),
        Err(_) => tokens.to_string(),
    }
}


pub fn render_node<T: ToTokens>(node: T) -> String {
    let tokens = quote::quote! {
        # node
    };
    match syn::parse_file(&tokens.to_string()) {
        Ok(file) => prettyplease::unparse(&file),
        Err(_) => tokens.to_string(),
    }
}


pub fn render_file(ast: &syn::File) -> String {
    prettyplease::unparse(ast)
}


pub fn render_file(ast: &syn::File) -> String {
    prettyplease::unparse(ast)
}


pub fn render_fn_item(func: &syn::ItemFn) -> String {
    render_node(func)
}


pub fn render_fn_item(func: &syn::ItemFn) -> String {
    render_node(func)
}


pub fn render_impl_item(item: &syn::ItemImpl) -> String {
    render_node(item)
}


pub fn render_impl_item(item: &syn::ItemImpl) -> String {
    render_node(item)
}


pub fn render_struct(item: &syn::ItemStruct) -> String {
    render_node(item)
}


pub fn render_struct(item: &syn::ItemStruct) -> String {
    render_node(item)
}


pub fn render_trait(item: &syn::ItemTrait) -> String {
    render_node(item)
}


pub fn render_trait(item: &syn::ItemTrait) -> String {
    render_node(item)
}
