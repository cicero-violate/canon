use serde_json::Value as JsonValue;
use crate::ir::{Function, TypeRef, ValuePort};
use super::render_struct::render_visibility;

// ── type rendering ───────────────────────────────────────────────────────────

pub fn render_type(ty: &TypeRef) -> String {
    ty.name.as_str().to_owned()
}

// ── function signature ───────────────────────────────────────────────────────

pub fn render_fn_signature(inputs: &[ValuePort], outputs: &[ValuePort]) -> String {
    let params = inputs
        .iter()
        .map(|p| format!("{}: {}", p.name, render_type(&p.ty)))
        .collect::<Vec<_>>()
        .join(", ");

    let mut sig = format!("({params})");
    if let Some(ret) = render_output_types(outputs) {
        sig.push_str(" -> ");
        sig.push_str(&ret);
    }
    sig
}

fn render_output_types(outputs: &[ValuePort]) -> Option<String> {
    match outputs.len() {
        0 => None,
        1 => Some(render_type(&outputs[0].ty)),
        _ => Some(format!(
            "({})",
            outputs.iter().map(|o| render_type(&o.ty)).collect::<Vec<_>>().join(", ")
        )),
    }
}

// ── impl function ─────────────────────────────────────────────────────────────

pub fn render_impl_function(function: &Function) -> String {
    let sig = render_fn_signature(&function.inputs, &function.outputs);
    let body = match &function.metadata.ast {
        Some(ast) => render_ast_body(ast, 1),
        None => format!(
            "    // Canon runtime stub\n    canon_runtime::execute_function(\"{}\");\n",
            function.id
        ),
    };
    format!(
        "    {}fn {}{} {{\n{body}    }}",
        render_visibility(function.visibility),
        function.name,
        sig,
    )
}

// ── AST renderer ─────────────────────────────────────────────────────────────
// JSON shape per node kind:
//
//   block  { "kind": "block",  "stmts": [AstNode] }
//   let    { "kind": "let",    "name": str, "value": AstNode }
//   if     { "kind": "if",     "cond": AstNode, "then": AstNode, "else": AstNode? }
//   match  { "kind": "match",  "expr": AstNode, "arms": [{"pattern": str, "body": AstNode}] }
//   while  { "kind": "while",  "cond": AstNode, "body": AstNode }
//   return { "kind": "return", "value": AstNode? }
//   call   { "kind": "call",   "func": str, "args": [AstNode] }
//   lit    { "kind": "lit",    "value": str }

pub fn render_ast_body(ast: &JsonValue, depth: usize) -> String {
    let mut out = String::new();
    render_ast_node(ast, depth, &mut out);
    out
}

fn indent(depth: usize) -> String {
    "    ".repeat(depth)
}

fn render_ast_node(node: &JsonValue, depth: usize, out: &mut String) {
    let kind = node.get("kind").and_then(|k| k.as_str()).unwrap_or("lit");
    match kind {
        "block" => render_block(node, depth, out),
        "let"   => render_let(node, depth, out),
        "if"    => render_if(node, depth, out),
        "match" => render_match(node, depth, out),
        "while" => render_while(node, depth, out),
        "return"=> render_return(node, depth, out),
        "call"  => render_call_stmt(node, depth, out),
        _       => render_lit(node, depth, out),
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
    let name = node.get("name").and_then(|n| n.as_str()).unwrap_or("_");
    let value = node.get("value");
    let rhs = value.map(|v| render_ast_expr(v)).unwrap_or_else(|| "()".to_owned());
    out.push_str(&format!("{}let {name} = {rhs};\n", indent(depth)));
}

fn render_if(node: &JsonValue, depth: usize, out: &mut String) {
    let cond = node.get("cond").map(|c| render_ast_expr(c)).unwrap_or_else(|| "true".to_owned());
    out.push_str(&format!("{}if {cond} {{\n", indent(depth)));
    if let Some(then) = node.get("then") {
        render_ast_node(then, depth + 1, out);
    }
    if let Some(else_branch) = node.get("else") {
        out.push_str(&format!("{}}} else {{\n", indent(depth)));
        render_ast_node(else_branch, depth + 1, out);
    }
    out.push_str(&format!("{}}}\n", indent(depth)));
}

fn render_match(node: &JsonValue, depth: usize, out: &mut String) {
    let expr = node.get("expr").map(|e| render_ast_expr(e)).unwrap_or_else(|| "_".to_owned());
    out.push_str(&format!("{}match {expr} {{\n", indent(depth)));
    if let Some(arms) = node.get("arms").and_then(|a| a.as_array()) {
        for arm in arms {
            let pattern = arm.get("pattern").and_then(|p| p.as_str()).unwrap_or("_");
            out.push_str(&format!("{}    {pattern} => {{\n", indent(depth)));
            if let Some(body) = arm.get("body") {
                render_ast_node(body, depth + 2, out);
            }
            out.push_str(&format!("{}    }}\n", indent(depth)));
        }
    }
    out.push_str(&format!("{}}}\n", indent(depth)));
}

fn render_while(node: &JsonValue, depth: usize, out: &mut String) {
    let cond = node.get("cond").map(|c| render_ast_expr(c)).unwrap_or_else(|| "true".to_owned());
    out.push_str(&format!("{}while {cond} {{\n", indent(depth)));
    if let Some(body) = node.get("body") {
        render_ast_node(body, depth + 1, out);
    }
    out.push_str(&format!("{}}}\n", indent(depth)));
}

fn render_return(node: &JsonValue, depth: usize, out: &mut String) {
    match node.get("value") {
        Some(v) => out.push_str(&format!("{}return {};\n", indent(depth), render_ast_expr(v))),
        None    => out.push_str(&format!("{}return;\n", indent(depth))),
    }
}

fn render_call_stmt(node: &JsonValue, depth: usize, out: &mut String) {
    out.push_str(&format!("{}{};\n", indent(depth), render_call_expr(node)));
}

fn render_lit(node: &JsonValue, depth: usize, out: &mut String) {
    let value = node.get("value").and_then(|v| v.as_str()).unwrap_or("()");
    out.push_str(&format!("{}{value};\n", indent(depth)));
}

// ── expression renderers (return String, no trailing newline/semicolon) ───────

fn render_ast_expr(node: &JsonValue) -> String {
    let kind = node.get("kind").and_then(|k| k.as_str()).unwrap_or("lit");
    match kind {
        "call"  => render_call_expr(node),
        "if"    => render_if_expr(node),
        "match" => render_match_expr(node),
        _       => node.get("value").and_then(|v| v.as_str()).unwrap_or("()").to_owned(),
    }
}

fn render_call_expr(node: &JsonValue) -> String {
    let func = node.get("func").and_then(|f| f.as_str()).unwrap_or("unknown");
    let args = node
        .get("args")
        .and_then(|a| a.as_array())
        .map(|arr| arr.iter().map(render_ast_expr).collect::<Vec<_>>().join(", "))
        .unwrap_or_default();
    format!("{func}({args})")
}

fn render_if_expr(node: &JsonValue) -> String {
    let cond = node.get("cond").map(|c| render_ast_expr(c)).unwrap_or_else(|| "true".to_owned());
    let then = node.get("then").map(|t| render_ast_expr(t)).unwrap_or_else(|| "()".to_owned());
    match node.get("else") {
        Some(e) => format!("if {cond} {{ {then} }} else {{ {} }}", render_ast_expr(e)),
        None    => format!("if {cond} {{ {then} }}"),
    }
}

fn render_match_expr(node: &JsonValue) -> String {
    let expr = node.get("expr").map(|e| render_ast_expr(e)).unwrap_or_else(|| "_".to_owned());
    let arms = node
        .get("arms")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .map(|arm| {
                    let pat = arm.get("pattern").and_then(|p| p.as_str()).unwrap_or("_");
                    let body = arm.get("body").map(|b| render_ast_expr(b)).unwrap_or_else(|| "()".to_owned());
                    format!("{pat} => {body}")
                })
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    format!("match {expr} {{ {arms} }}")
}
