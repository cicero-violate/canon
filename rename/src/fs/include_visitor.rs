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
pub(crate) fn is_ignored_dir(path: &Path) -> bool {
    if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
        matches!(name, "target" | ".git" | ".semantic-lint" | "dogfood-output")
    } else {
        false
    }
}
