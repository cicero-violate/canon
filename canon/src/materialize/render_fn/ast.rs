use serde_json::Value as JsonValue;

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
        "let" => render_let(node, depth, out),
        "if" => render_if(node, depth, out),
        "match" => render_match(node, depth, out),
        "while" => render_while(node, depth, out),
        "for" => render_for(node, depth, out),
        "loop" => render_loop(node, depth, out),
        "break" => render_break(node, depth, out),
        "continue" => out.push_str(&format!("{}continue;\n", indent(depth))),
        "return" => render_return(node, depth, out),
        "assign" => render_assign(node, depth, out),
        "compound_assign" => render_compound_assign(node, depth, out),
        "call" => render_call_stmt(node, depth, out),
        "lit" => render_literal(node, depth, out),
        _ => render_expression_stmt(node, depth, out),
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
    let mut_parts = match node.get("mutable").and_then(|m| m.as_bool()) {
        Some(true) => "mut ",
        _ => "",
    };
    let name = node.get("name").and_then(|n| n.as_str()).unwrap_or("_");
    let rhs = node
        .get("value")
        .map(render_ast_expr)
        .unwrap_or_else(|| "()".to_owned());
    out.push_str(&format!(
        "{}let {mut_parts}{name} = {rhs};\n",
        indent(depth)
    ));
}

fn render_if(node: &JsonValue, depth: usize, out: &mut String) {
    let cond = node
        .get("cond")
        .map(render_ast_expr)
        .unwrap_or_else(|| "true".to_owned());
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
    let expr = node
        .get("expr")
        .map(render_ast_expr)
        .unwrap_or_else(|| "_".to_owned());
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
    let cond = node
        .get("cond")
        .map(render_ast_expr)
        .unwrap_or_else(|| "true".to_owned());
    out.push_str(&format!("{}while {cond} {{\n", indent(depth)));
    if let Some(body) = node.get("body") {
        render_ast_node(body, depth + 1, out);
    }
    out.push_str(&format!("{}}}\n", indent(depth)));
}

fn render_for(node: &JsonValue, depth: usize, out: &mut String) {
    let pat = node.get("pat").and_then(|p| p.as_str()).unwrap_or("_");
    let iter = node
        .get("iter")
        .map(render_ast_expr)
        .unwrap_or_else(|| "()".to_owned());
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
        Some(value) => out.push_str(&format!(
            "{}break {};\n",
            indent(depth),
            render_ast_expr(value)
        )),
        None => out.push_str(&format!("{}break;\n", indent(depth))),
    }
}

fn render_return(node: &JsonValue, depth: usize, out: &mut String) {
    match node.get("value") {
        Some(value) => out.push_str(&format!(
            "{}return {};\n",
            indent(depth),
            render_ast_expr(value)
        )),
        None => out.push_str(&format!("{}return;\n", indent(depth))),
    }
}

fn render_assign(node: &JsonValue, depth: usize, out: &mut String) {
    let target = node
        .get("target")
        .map(render_ast_expr)
        .unwrap_or_else(|| "_".to_owned());
    let value = node
        .get("value")
        .map(render_ast_expr)
        .unwrap_or_else(|| "()".to_owned());
    out.push_str(&format!("{}{target} = {value};\n", indent(depth)));
}

fn render_compound_assign(node: &JsonValue, depth: usize, out: &mut String) {
    let target = node
        .get("target")
        .map(render_ast_expr)
        .unwrap_or_else(|| "_".to_owned());
    let value = node
        .get("value")
        .map(render_ast_expr)
        .unwrap_or_else(|| "()".to_owned());
    let op = node.get("op").and_then(|o| o.as_str()).unwrap_or("+=");
    out.push_str(&format!("{}{target} {op} {value};\n", indent(depth)));
}

fn render_call_stmt(node: &JsonValue, depth: usize, out: &mut String) {
    out.push_str(&format!("{}{};\n", indent(depth), render_call_expr(node)));
}

fn render_literal(node: &JsonValue, depth: usize, out: &mut String) {
    let value = node.get("value").and_then(|v| v.as_str()).unwrap_or("()");
    out.push_str(&format!("{}{value};\n", indent(depth)));
}

fn render_expression_stmt(node: &JsonValue, depth: usize, out: &mut String) {
    let expr = render_ast_expr(node);
    out.push_str(&format!("{}{expr};\n", indent(depth)));
}

// ── expressions --------------------------------------------------------------

fn render_ast_expr(node: &JsonValue) -> String {
    let kind = node.get("kind").and_then(|k| k.as_str()).unwrap_or("lit");
    match kind {
        "call" => render_call_expr(node),
        "method" => render_method_expr(node),
        "closure" => render_closure_expr(node),
        "bin" | "cmp" | "logical" => render_binary_expr(node),
        "unary" => render_unary_expr(node),
        "field" => render_field_expr(node),
        "index" => render_index_expr(node),
        "struct_lit" => render_struct_lit_expr(node),
        "tuple" => render_tuple_expr(node),
        "array" => render_array_expr(node),
        "ref" => render_ref_expr(node),
        "range" => render_range_expr(node),
        "cast" => render_cast_expr(node),
        "question" => node
            .get("expr")
            .map(render_ast_expr)
            .map(|expr| format!("{expr}?"))
            .unwrap_or_else(|| "?".to_owned()),
        _ => node
            .get("value")
            .and_then(|v| v.as_str())
            .unwrap_or("()")
            .to_owned(),
    }
}

fn render_call_expr(node: &JsonValue) -> String {
    let func = node
        .get("func")
        .and_then(|f| f.as_str())
        .unwrap_or("unknown");
    let args = node
        .get("args")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .map(render_ast_expr)
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    format!("{func}({args})")
}

fn render_method_expr(node: &JsonValue) -> String {
    let receiver = node
        .get("receiver")
        .map(render_ast_expr)
        .unwrap_or_else(|| "self".to_owned());
    let method = node
        .get("method")
        .and_then(|m| m.as_str())
        .unwrap_or("call");
    let args = node
        .get("args")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .map(render_ast_expr)
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    format!("{receiver}.{method}({args})")
}

fn render_binary_expr(node: &JsonValue) -> String {
    let lhs = node
        .get("lhs")
        .map(render_ast_expr)
        .unwrap_or_else(|| "_".to_owned());
    let rhs = node
        .get("rhs")
        .map(render_ast_expr)
        .unwrap_or_else(|| "_".to_owned());
    let op = node.get("op").and_then(|o| o.as_str()).unwrap_or("+");
    format!("{lhs} {op} {rhs}")
}

fn render_unary_expr(node: &JsonValue) -> String {
    let op = node.get("op").and_then(|o| o.as_str()).unwrap_or("-");
    let expr = node
        .get("expr")
        .map(render_ast_expr)
        .unwrap_or_else(|| "_".to_owned());
    format!("{op}{expr}")
}

fn render_field_expr(node: &JsonValue) -> String {
    let expr = node
        .get("expr")
        .map(render_ast_expr)
        .unwrap_or_else(|| "_".to_owned());
    let field = node.get("field").and_then(|f| f.as_str()).unwrap_or("_");
    format!("{expr}.{field}")
}

fn render_index_expr(node: &JsonValue) -> String {
    let expr = node
        .get("expr")
        .map(render_ast_expr)
        .unwrap_or_else(|| "_".to_owned());
    let index = node
        .get("index")
        .map(render_ast_expr)
        .unwrap_or_else(|| "0".to_owned());
    format!("{expr}[{index}]")
}

fn render_struct_lit_expr(node: &JsonValue) -> String {
    let name = node
        .get("name")
        .and_then(|n| n.as_str())
        .unwrap_or("Struct");
    let fields = node
        .get("fields")
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .map(|field| {
                    let fname = field.get("name").and_then(|n| n.as_str()).unwrap_or("_");
                    let fvalue = field
                        .get("value")
                        .map(render_ast_expr)
                        .unwrap_or_else(|| "()".to_owned());
                    format!("{fname}: {fvalue}")
                })
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    format!("{name} {{ {fields} }}")
}

fn render_tuple_expr(node: &JsonValue) -> String {
    match node.get("elems").and_then(|e| e.as_array()) {
        Some(elems) if elems.len() == 1 => {
            let single = render_ast_expr(&elems[0]);
            format!("({single},)")
        }
        Some(elems) => format!(
            "({})",
            elems
                .iter()
                .map(render_ast_expr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        None => "()".to_owned(),
    }
}

fn render_array_expr(node: &JsonValue) -> String {
    let elems = node
        .get("elems")
        .and_then(|e| e.as_array())
        .map(|arr| {
            arr.iter()
                .map(render_ast_expr)
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    format!("[{elems}]")
}

fn render_ref_expr(node: &JsonValue) -> String {
    let expr = node
        .get("expr")
        .map(render_ast_expr)
        .unwrap_or_else(|| "_".to_owned());
    let mut_kw = match node.get("mut").and_then(|m| m.as_bool()) {
        Some(true) => "mut ",
        _ => "",
    };
    format!("&{mut_kw}{expr}")
}

fn render_range_expr(node: &JsonValue) -> String {
    let from = node.get("from").map(render_ast_expr);
    let to = node.get("to").map(render_ast_expr);
    let inclusive = node
        .get("inclusive")
        .and_then(|i| i.as_bool())
        .unwrap_or(false);
    let op = if inclusive { "..=" } else { ".." };
    match (from, to) {
        (Some(start), Some(end)) => format!("{start}{op}{end}"),
        (Some(start), None) => format!("{start}{op}"),
        (None, Some(end)) => format!("{op}{end}"),
        _ => format!("0{op}0"),
    }
}

fn render_cast_expr(node: &JsonValue) -> String {
    let expr = node
        .get("expr")
        .map(render_ast_expr)
        .unwrap_or_else(|| "_".to_owned());
    let ty = node.get("ty").and_then(|t| t.as_str()).unwrap_or("()");
    format!("{expr} as {ty}")
}

fn render_closure_expr(node: &JsonValue) -> String {
    let params = node
        .get("params")
        .and_then(|p| p.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|value| value.as_str().map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let params_rendered = if params.is_empty() {
        "||".to_owned()
    } else {
        format!("|{}|", params.join(", "))
    };
    let move_prefix = if node.get("move").and_then(|m| m.as_bool()).unwrap_or(false) {
        "move "
    } else {
        ""
    };
    let Some(body) = node.get("body") else {
        return format!("{move_prefix}{params_rendered} ()");
    };
    let body_kind = body.get("kind").and_then(|k| k.as_str()).unwrap_or("");
    if body_kind == "block" {
        let rendered = render_ast_body(body, 1);
        let trimmed = rendered.trim_end();
        format!("{move_prefix}{params_rendered} {{\n{trimmed}\n}}")
    } else {
        let expr = render_ast_expr(body);
        format!("{move_prefix}{params_rendered} {expr}")
    }
}
