mod expr;
mod stmt;
mod utils;

use serde_json::Value as JsonValue;

/// Public entry-point: render the body of an AST node into a Rust source string.
pub fn render_ast_body(ast: &JsonValue, depth: usize) -> String {
    stmt::render_ast_body_inner(ast, depth)
}

/// Re-export for callers that need to render a single AST node into an existing buffer.
pub use stmt::render_ast_node;

/// Re-export for callers that need expression rendering directly.
pub use expr::render_ast_expr;
