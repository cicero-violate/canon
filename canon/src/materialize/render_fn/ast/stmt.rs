use serde_json::Value as JsonValue;
use super::utils::indent;
use super::expr::{render_ast_expr, render_call_expr};

/// Render all statements inside an AST body node, returning the accumulated string.
/// Exposed so that `expr.rs` can call it for closure bodies.
pub fn render_ast_body_inner(ast: &JsonValue, depth: usize) -> String {
    let mut out = String::new();
    render_ast_node(ast, depth, &mut out);
    out
}

pub fn render_ast_node(node: &JsonValue, depth: usize, out: &mut String) {
    let kind = node.get("kind").and_then(|k| k.as_str()).unwrap_or("lit");
    match kind {
        "block"           => render_block(node, depth, out),
        "let"             => render_let(node, depth, out),
        "if"              => render_if(node, depth, out),
        "match"           => render_match(node, depth, out),
        "while"           => render_while(node, depth, out),
        "for"             => render_for(node, depth, out),
        "loop"            => render_loop(node, depth, out),
        "break"           => render_break(node, depth, out),
        "continue"        => out.push_str(&format!("{}continue;\n", indent(depth))),
        "return"          => render_return(node, depth, out),
        "assign"          => render_assign(node, depth, out),
        "compound_assign" => render_compound_assign(node, depth, out),
        "call"            => out.push_str(&format!("{}{};\n", indent(depth), render_call_expr(node))),
        "lit"             => render_literal(node, depth, out),
        _                 => render_expression_stmt(node, depth, out),
    }
}

fn render_block(node: &JsonValue, depth: usize, out: &mut String) {
    if let Some(stmts) = node.get("stmts").and_then(|s| s.as_array()) {
        for stmt in stmts {
            render_ast_node(stmt, depth, out);
        }
    }
}

fn render_let(node: &JsonValue, depth: usize, out: &mut String) {
    let mut_part = match node.get("mutable").and_then(|m| m.as_bool()) {
        Some(true) => "mut ",
        _          => "",
    };
    let name = node.get("name").and_then(|n| n.as_str()).unwrap_or("_");
    let rhs  = node.get("value").map(render_ast_expr).unwrap_or_else(|| "()".to_owned());
    out.push_str(&format!("{}let {mut_part}{name} = {rhs};\n", indent(depth)));
}

fn render_if(node: &JsonValue, depth: usize, out: &mut String) {
    let cond = node.get("cond").map(render_ast_expr).unwrap_or_else(|| "true".to_owned());
    out.push_str(&format!("{}if {cond} {{\n", indent(depth)));
    if let Some(then_branch) = node.get("then") {
        render_ast_node(then_branch, depth + 1, out);
    }
    if let Some(else_branch) = node.get("else") {
        out.push_str(&format!("{}}} else {{\n", indent(depth)));
        render_ast_node(else_branch, depth + 1, out);
    }
    out.push_str(&format!("{}}}\n", indent(depth)));
}

fn render_match(node: &JsonValue, depth: usize, out: &mut String) {
    let expr = node.get("expr").map(render_ast_expr).unwrap_or_else(|| "_".to_owned());
    out.push_str(&format!("{}match {expr} {{\n", indent(depth)));
    if let Some(arms) = node.get("arms").and_then(|a| a.as_array()) {
        for arm in arms {
            let pat = arm.get("pattern").and_then(|p| p.as_str()).unwrap_or("_");
            out.push_str(&format!("{}    {pat} => {{\n", indent(depth)));
            if let Some(body) = arm.get("body") {
                render_ast_node(body, depth + 2, out);
            }
            out.push_str(&format!("{}    }}\n", indent(depth)));
        }
    }
    out.push_str(&format!("{}}}\n", indent(depth)));
}

fn render_while(node: &JsonValue, depth: usize, out: &mut String) {
    let cond = node.get("cond").map(render_ast_expr).unwrap_or_else(|| "true".to_owned());
    out.push_str(&format!("{}while {cond} {{\n", indent(depth)));
    if let Some(body) = node.get("body") {
        render_ast_node(body, depth + 1, out);
    }
    out.push_str(&format!("{}}}\n", indent(depth)));
}

fn render_for(node: &JsonValue, depth: usize, out: &mut String) {
    let pat  = node.get("pat").and_then(|p| p.as_str()).unwrap_or("_");
    let iter = node.get("iter").map(render_ast_expr).unwrap_or_else(|| "()".to_owned());
    out.push_str(&format!("{}for {pat} in {iter} {{\n", indent(depth)));
    if let Some(body) = node.get("body") {
        render_ast_node(body, depth + 1, out);
    }
    out.push_str(&format!("{}}}\n", indent(depth)));
}

fn render_loop(node: &JsonValue, depth: usize, out: &mut String) {
    out.push_str(&format!("{}loop {{\n", indent(depth)));
    if let Some(body) = node.get("body") {
        render_ast_node(body, depth + 1, out);
    }
    out.push_str(&format!("{}}}\n", indent(depth)));
}

fn render_break(node: &JsonValue, depth: usize, out: &mut String) {
    match node.get("value") {
        Some(v) => out.push_str(&format!("{}break {};\n", indent(depth), render_ast_expr(v))),
        None    => out.push_str(&format!("{}break;\n", indent(depth))),
    }
}

fn render_return(node: &JsonValue, depth: usize, out: &mut String) {
    match node.get("value") {
        Some(v) => out.push_str(&format!("{}return {};\n", indent(depth), render_ast_expr(v))),
        None    => out.push_str(&format!("{}return;\n", indent(depth))),
    }
}

fn render_assign(node: &JsonValue, depth: usize, out: &mut String) {
    let target = node.get("target").map(render_ast_expr).unwrap_or_else(|| "_".to_owned());
    let value  = node.get("value").map(render_ast_expr).unwrap_or_else(|| "()".to_owned());
    out.push_str(&format!("{}{target} = {value};\n", indent(depth)));
}

fn render_compound_assign(node: &JsonValue, depth: usize, out: &mut String) {
    let target = node.get("target").map(render_ast_expr).unwrap_or_else(|| "_".to_owned());
    let value  = node.get("value").map(render_ast_expr).unwrap_or_else(|| "()".to_owned());
    let op     = node.get("op").and_then(|o| o.as_str()).unwrap_or("+=");
    out.push_str(&format!("{}{target} {op} {value};\n", indent(depth)));
}

fn render_literal(node: &JsonValue, depth: usize, out: &mut String) {
    let value = node.get("value").and_then(|v| v.as_str()).unwrap_or("()");
    out.push_str(&format!("{}{value};\n", indent(depth)));
}

fn render_expression_stmt(node: &JsonValue, depth: usize, out: &mut String) {
    out.push_str(&format!("{}{};\n", indent(depth), render_ast_expr(node)));
}
