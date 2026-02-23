use prettyplease;


use quote::ToTokens;


pub fn render_file(ast: &syn::File) -> String {
    prettyplease::unparse(ast)
}
