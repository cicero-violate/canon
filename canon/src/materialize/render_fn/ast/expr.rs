use serde_json::Value as JsonValue;
use super::utils::indent;

pub fn render_ast_expr(node: &JsonValue) -> String {
    let kind = node.get("kind").and_then(|k| k.as_str()).unwrap_or("lit");
    match kind {
        "call"              => render_call_expr(node),
        "method"            => render_method_expr(node),
        "closure"           => render_closure_expr(node),
        "bin" | "cmp" | "logical" => render_binary_expr(node),
        "unary"             => render_unary_expr(node),
        "field"             => render_field_expr(node),
        "index"             => render_index_expr(node),
        "struct_lit"        => render_struct_lit_expr(node),
        "tuple"             => render_tuple_expr(node),
        "array"             => render_array_expr(node),
        "ref"               => render_ref_expr(node),
        "range"             => render_range_expr(node),
        "cast"              => render_cast_expr(node),
        "question"          => node
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

pub fn render_call_expr(node: &JsonValue) -> String {
    let func = node.get("func").and_then(|f| f.as_str()).unwrap_or("unknown");
    let args = render_args(node);
    format!("{func}({args})")
}

fn render_method_expr(node: &JsonValue) -> String {
    let receiver = node
        .get("receiver")
        .map(render_ast_expr)
        .unwrap_or_else(|| "self".to_owned());
    let method = node.get("method").and_then(|m| m.as_str()).unwrap_or("call");
    let args = render_args(node);
    format!("{receiver}.{method}({args})")
}

fn render_binary_expr(node: &JsonValue) -> String {
    let lhs = node.get("lhs").map(render_ast_expr).unwrap_or_else(|| "_".to_owned());
    let rhs = node.get("rhs").map(render_ast_expr).unwrap_or_else(|| "_".to_owned());
    let op  = node.get("op").and_then(|o| o.as_str()).unwrap_or("+");
    format!("{lhs} {op} {rhs}")
}

fn render_unary_expr(node: &JsonValue) -> String {
    let op   = node.get("op").and_then(|o| o.as_str()).unwrap_or("-");
    let expr = node.get("expr").map(render_ast_expr).unwrap_or_else(|| "_".to_owned());
    format!("{op}{expr}")
}

fn render_field_expr(node: &JsonValue) -> String {
    let expr  = node.get("expr").map(render_ast_expr).unwrap_or_else(|| "_".to_owned());
    let field = node.get("field").and_then(|f| f.as_str()).unwrap_or("_");
    format!("{expr}.{field}")
}

fn render_index_expr(node: &JsonValue) -> String {
    let expr  = node.get("expr").map(render_ast_expr).unwrap_or_else(|| "_".to_owned());
    let index = node.get("index").map(render_ast_expr).unwrap_or_else(|| "0".to_owned());
    format!("{expr}[{index}]")
}

fn render_struct_lit_expr(node: &JsonValue) -> String {
    let name = node.get("name").and_then(|n| n.as_str()).unwrap_or("Struct");
    let fields = node
        .get("fields")
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .map(|field| {
                    let fname  = field.get("name").and_then(|n| n.as_str()).unwrap_or("_");
                    let fvalue = field.get("value").map(render_ast_expr).unwrap_or_else(|| "()".to_owned());
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
            format!("({},)", render_ast_expr(&elems[0]))
        }
        Some(elems) => format!(
            "({})",
            elems.iter().map(render_ast_expr).collect::<Vec<_>>().join(", ")
        ),
        None => "()".to_owned(),
    }
}

fn render_array_expr(node: &JsonValue) -> String {
    let elems = node
        .get("elems")
        .and_then(|e| e.as_array())
        .map(|arr| arr.iter().map(render_ast_expr).collect::<Vec<_>>().join(", "))
        .unwrap_or_default();
    format!("[{elems}]")
}

fn render_ref_expr(node: &JsonValue) -> String {
    let expr    = node.get("expr").map(render_ast_expr).unwrap_or_else(|| "_".to_owned());
    let mut_kw  = match node.get("mut").and_then(|m| m.as_bool()) {
        Some(true) => "mut ",
        _          => "",
    };
    format!("&{mut_kw}{expr}")
}

fn render_range_expr(node: &JsonValue) -> String {
    let from      = node.get("from").map(render_ast_expr);
    let to        = node.get("to").map(render_ast_expr);
    let inclusive = node.get("inclusive").and_then(|i| i.as_bool()).unwrap_or(false);
    let op        = if inclusive { "..=" } else { ".." };
    match (from, to) {
        (Some(s), Some(e)) => format!("{s}{op}{e}"),
        (Some(s), None)    => format!("{s}{op}"),
        (None,    Some(e)) => format!("{op}{e}"),
        _                  => format!("0{op}0"),
    }
}

fn render_cast_expr(node: &JsonValue) -> String {
    let expr = node.get("expr").map(render_ast_expr).unwrap_or_else(|| "_".to_owned());
    let ty   = node.get("ty").and_then(|t| t.as_str()).unwrap_or("()");
    format!("{expr} as {ty}")
}

fn render_closure_expr(node: &JsonValue) -> String {
    use super::stmt::render_ast_body_inner;
    let params = node
        .get("params")
        .and_then(|p| p.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect::<Vec<_>>())
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
    if body.get("kind").and_then(|k| k.as_str()).unwrap_or("") == "block" {
        let rendered = render_ast_body_inner(body, 1);
        let trimmed  = rendered.trim_end();
        format!("{move_prefix}{params_rendered} {{\n{trimmed}\n}}")
    } else {
        let expr = render_ast_expr(body);
        format!("{move_prefix}{params_rendered} {expr}")
    }
}

// ── shared helper ────────────────────────────────────────────────────────────

fn render_args(node: &JsonValue) -> String {
    node.get("args")
        .and_then(|a| a.as_array())
        .map(|arr| arr.iter().map(render_ast_expr).collect::<Vec<_>>().join(", "))
        .unwrap_or_default()
}
