use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};
impl<'ast> Visit<'ast> for IncludeVisitor<'_> {
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if mac.path.is_ident("include") {
            if let Ok(lit) = mac.parse_body::<syn::LitStr>() {
                let target = self.base_path.join(lit.value());
                if target.exists()
                    && target.extension().and_then(|s| s.to_str()) == Some("rs")
                {
                    self.targets.push(target);
                }
            }
        }
        visit::visit_macro(self, mac);
    }
    fn visit_expr_macro(&mut self, expr: &'ast syn::ExprMacro) {
        if expr.mac.path.is_ident("include") {
            if let Ok(lit) = expr.mac.parse_body::<syn::LitStr>() {
                let target = self.base_path.join(lit.value());
                if target.exists()
                    && target.extension().and_then(|s| s.to_str()) == Some("rs")
                {
                    self.targets.push(target);
                }
            }
        }
        visit::visit_expr_macro(self, expr);
    }
}
pub(crate) struct IncludeVisitor<'a> {
    pub(crate) base_path: &'a Path,
    pub(crate) targets: &'a mut Vec<PathBuf>,
}
