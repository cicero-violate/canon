//! Lower a `syn::Block` into the Canon JSON AST format consumed by
//! `FunctionMetadata::ast` and the `render_fn/ast` renderer.
//!
//! Every node carries a `"kind"` discriminant matching the whitelist in
//! `validate/check_artifacts/ast.rs`.

mod expr;
mod pat;
mod util;

use serde_json::{json, Value as JsonValue};

pub(crate) use expr::lower_expr;

/// Entry point: lower a full function body block.
/// Returns `None` when the block is empty so `metadata.ast` stays absent.
pub(crate) fn lower_block(block: &syn::Block) -> Option<JsonValue> {
    if block.stmts.is_empty() {
        return None;
    }
    let stmts: Vec<JsonValue> = block.stmts.iter().filter_map(lower_stmt).collect();
    if stmts.is_empty() {
        return None;
    }
    Some(json!({ "kind": "block", "stmts": stmts }))
}

pub(crate) fn lower_stmt(stmt: &syn::Stmt) -> Option<JsonValue> {
    match stmt {
        syn::Stmt::Local(local) => Some(lower_let(local)),
        syn::Stmt::Expr(e, Some(_)) => Some(expr::lower_expr_stmt(e)),
        syn::Stmt::Expr(e, None) => Some(expr::lower_expr(e)),
        syn::Stmt::Item(_) => None,
        syn::Stmt::Macro(mac) => Some(lower_macro_stmt(mac)),
    }
}

fn lower_let(local: &syn::Local) -> JsonValue {
    let name = pat::pat_to_string(&local.pat);
    let mutable = pat::pat_is_mut(&local.pat);
    let value = local.init.as_ref().map(|init| expr::lower_expr(&init.expr));
    match value {
        Some(v) => json!({ "kind": "let", "name": name, "mutable": mutable, "value": v }),
        None => json!({ "kind": "let", "name": name, "mutable": mutable }),
    }
}

fn lower_macro_stmt(mac: &syn::StmtMacro) -> JsonValue {
    let tokens = mac.mac.tokens.to_string();
    json!({ "kind": "call", "func": util::path_to_str(&mac.mac.path), "args": [{ "kind": "lit", "value": tokens }] })
}
