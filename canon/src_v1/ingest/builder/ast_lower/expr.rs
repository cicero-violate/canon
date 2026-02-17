use serde_json::{Value as JsonValue, json};

use super::pat::{pat_is_mut, pat_to_string};
use super::util::{expr_to_call_str, path_to_str, type_to_str};

pub(crate) fn lower_expr(expr: &syn::Expr) -> JsonValue {
    match expr {
        syn::Expr::Lit(l) => lower_lit(l),
        syn::Expr::Path(p) => json!({ "kind": "lit", "value": path_to_str(&p.path) }),
        syn::Expr::Binary(b) => lower_binary(b),
        syn::Expr::Unary(u) => lower_unary(u),
        syn::Expr::Call(c) => lower_call(c),
        syn::Expr::MethodCall(m) => lower_method_call(m),
        syn::Expr::Field(f) => lower_field(f),
        syn::Expr::Index(i) => lower_index(i),
        syn::Expr::Struct(s) => lower_struct_lit(s),
        syn::Expr::Tuple(t) => lower_tuple(t),
        syn::Expr::Array(a) => lower_array(a),
        syn::Expr::Reference(r) => lower_ref(r),
        syn::Expr::Range(r) => lower_range(r),
        syn::Expr::Cast(c) => lower_cast(c),
        syn::Expr::Closure(c) => lower_closure(c),
        syn::Expr::If(i) => lower_if(i),
        syn::Expr::Match(m) => lower_match(m),
        syn::Expr::Block(b) => lower_block_stmts(&b.block),
        syn::Expr::Return(r) => lower_return(r),
        syn::Expr::Break(b) => lower_break(b),
        syn::Expr::Continue(_) => json!({ "kind": "continue" }),
        syn::Expr::Assign(a) => lower_assign(a),
        syn::Expr::While(w) => lower_while(w),
        syn::Expr::ForLoop(f) => lower_for(f),
        syn::Expr::Loop(l) => lower_loop(l),
        syn::Expr::Paren(p) => lower_expr(&p.expr),
        syn::Expr::Group(g) => lower_expr(&g.expr),
        syn::Expr::Try(t) => json!({ "kind": "question", "expr": lower_expr(&t.expr) }),
        syn::Expr::Repeat(r) => json!({
            "kind": "array",
            "elems": [lower_expr(&r.expr)],
            "repeat_len": lower_expr(&r.len),
        }),
        syn::Expr::Macro(m) => lower_macro_expr(m),
        syn::Expr::Await(a) => json!({
            "kind": "method",
            "receiver": lower_expr(&a.base),
            "method": "await",
            "args": [],
        }),
        syn::Expr::Async(a) => json!({
            "kind": "closure",
            "params": [],
            "body": super::lower_block(&a.block),
            "async": true,
        }),
        syn::Expr::Unsafe(u) => lower_block_stmts(&u.block),
        _ => json!({ "kind": "lit", "value": "()" }),
    }
}

/// Lower a semicolon-terminated expression as a statement.
pub(crate) fn lower_expr_stmt(expr: &syn::Expr) -> JsonValue {
    match expr {
        syn::Expr::Return(r) => lower_return(r),
        syn::Expr::Break(b) => lower_break(b),
        syn::Expr::Continue(_) => json!({ "kind": "continue" }),
        syn::Expr::Assign(a) => lower_assign(a),
        syn::Expr::If(i) => lower_if(i),
        syn::Expr::While(w) => lower_while(w),
        syn::Expr::ForLoop(f) => lower_for(f),
        syn::Expr::Loop(l) => lower_loop(l),
        syn::Expr::Match(m) => lower_match(m),
        syn::Expr::Block(b) => lower_block_stmts(&b.block),
        syn::Expr::Macro(m) => lower_macro_expr(m),
        _ => json!({
            "kind": "call",
            "func": expr_to_call_str(expr),
            "args": [],
            "_expr": lower_expr(expr),
        }),
    }
}

pub(crate) fn lower_block_stmts(block: &syn::Block) -> JsonValue {
    let stmts: Vec<JsonValue> = block.stmts.iter().filter_map(super::lower_stmt).collect();
    json!({ "kind": "block", "stmts": stmts })
}

// ── individual expression lowerers ───────────────────────────────────────────

fn lower_lit(l: &syn::ExprLit) -> JsonValue {
    let value = match &l.lit {
        syn::Lit::Int(i) => i.base10_digits().to_owned(),
        syn::Lit::Float(f) => f.base10_digits().to_owned(),
        syn::Lit::Bool(b) => b.value.to_string(),
        syn::Lit::Str(s) => format!("\"{}\"", s.value()),
        syn::Lit::Char(c) => format!("'{}'", c.value()),
        syn::Lit::Byte(b) => format!("{}", b.value()),
        _ => "()".to_owned(),
    };
    json!({ "kind": "lit", "value": value })
}

fn lower_binary(b: &syn::ExprBinary) -> JsonValue {
    let (kind, op) = bin_op_to_kind(&b.op);
    json!({ "kind": kind, "lhs": lower_expr(&b.left), "rhs": lower_expr(&b.right), "op": op })
}

fn bin_op_to_kind(op: &syn::BinOp) -> (&'static str, &'static str) {
    match op {
        syn::BinOp::Add(_) => ("bin", "+"),
        syn::BinOp::Sub(_) => ("bin", "-"),
        syn::BinOp::Mul(_) => ("bin", "*"),
        syn::BinOp::Div(_) => ("bin", "/"),
        syn::BinOp::Rem(_) => ("bin", "%"),
        syn::BinOp::BitAnd(_) => ("bin", "&"),
        syn::BinOp::BitOr(_) => ("bin", "|"),
        syn::BinOp::BitXor(_) => ("bin", "^"),
        syn::BinOp::Shl(_) => ("bin", "<<"),
        syn::BinOp::Shr(_) => ("bin", ">>"),
        syn::BinOp::Eq(_) => ("cmp", "=="),
        syn::BinOp::Ne(_) => ("cmp", "!="),
        syn::BinOp::Lt(_) => ("cmp", "<"),
        syn::BinOp::Le(_) => ("cmp", "<="),
        syn::BinOp::Gt(_) => ("cmp", ">"),
        syn::BinOp::Ge(_) => ("cmp", ">="),
        syn::BinOp::And(_) => ("logical", "&&"),
        syn::BinOp::Or(_) => ("logical", "||"),
        syn::BinOp::AddAssign(_) => ("bin", "+="),
        syn::BinOp::SubAssign(_) => ("bin", "-="),
        syn::BinOp::MulAssign(_) => ("bin", "*="),
        syn::BinOp::DivAssign(_) => ("bin", "/="),
        syn::BinOp::RemAssign(_) => ("bin", "%="),
        _ => ("bin", "?"),
    }
}

fn lower_unary(u: &syn::ExprUnary) -> JsonValue {
    let op = match &u.op {
        syn::UnOp::Not(_) => "!",
        syn::UnOp::Neg(_) => "-",
        syn::UnOp::Deref(_) => "*",
        _ => "?",
    };
    json!({ "kind": "unary", "op": op, "expr": lower_expr(&u.expr) })
}

fn lower_call(c: &syn::ExprCall) -> JsonValue {
    let args: Vec<JsonValue> = c.args.iter().map(lower_expr).collect();
    json!({ "kind": "call", "func": expr_to_call_str(&c.func), "args": args })
}

fn lower_method_call(m: &syn::ExprMethodCall) -> JsonValue {
    let args: Vec<JsonValue> = m.args.iter().map(lower_expr).collect();
    json!({ "kind": "method", "receiver": lower_expr(&m.receiver), "method": m.method.to_string(), "args": args })
}

fn lower_field(f: &syn::ExprField) -> JsonValue {
    let field = match &f.member {
        syn::Member::Named(i) => i.to_string(),
        syn::Member::Unnamed(i) => i.index.to_string(),
    };
    json!({ "kind": "field", "expr": lower_expr(&f.base), "field": field })
}

fn lower_index(i: &syn::ExprIndex) -> JsonValue {
    json!({ "kind": "index", "expr": lower_expr(&i.expr), "index": lower_expr(&i.index) })
}

fn lower_struct_lit(s: &syn::ExprStruct) -> JsonValue {
    let fields: Vec<JsonValue> = s
        .fields
        .iter()
        .map(|fv| {
            let fname = match &fv.member {
                syn::Member::Named(i) => i.to_string(),
                syn::Member::Unnamed(i) => i.index.to_string(),
            };
            json!({ "name": fname, "value": lower_expr(&fv.expr) })
        })
        .collect();
    json!({ "kind": "struct_lit", "name": path_to_str(&s.path), "fields": fields })
}

fn lower_tuple(t: &syn::ExprTuple) -> JsonValue {
    let elems: Vec<JsonValue> = t.elems.iter().map(lower_expr).collect();
    json!({ "kind": "tuple", "elems": elems })
}

fn lower_array(a: &syn::ExprArray) -> JsonValue {
    let elems: Vec<JsonValue> = a.elems.iter().map(lower_expr).collect();
    json!({ "kind": "array", "elems": elems })
}

fn lower_ref(r: &syn::ExprReference) -> JsonValue {
    json!({ "kind": "ref", "expr": lower_expr(&r.expr), "mut": r.mutability.is_some() })
}

fn lower_range(r: &syn::ExprRange) -> JsonValue {
    let from = r.start.as_deref().map(lower_expr);
    let to = r.end.as_deref().map(lower_expr);
    let inclusive = matches!(r.limits, syn::RangeLimits::Closed(_));
    json!({ "kind": "range", "from": from, "to": to, "inclusive": inclusive })
}

fn lower_cast(c: &syn::ExprCast) -> JsonValue {
    json!({ "kind": "cast", "expr": lower_expr(&c.expr), "ty": type_to_str(&c.ty) })
}

fn lower_closure(c: &syn::ExprClosure) -> JsonValue {
    let params: Vec<String> = c.inputs.iter().map(pat_to_string).collect();
    json!({ "kind": "closure", "params": params, "move": c.capture.is_some(), "body": lower_expr(&c.body) })
}

fn lower_if(i: &syn::ExprIf) -> JsonValue {
    let else_branch = i.else_branch.as_ref().map(|(_, e)| lower_expr(e));
    json!({ "kind": "if", "cond": lower_expr(&i.cond), "then": lower_block_stmts(&i.then_branch), "else": else_branch })
}

fn lower_while(w: &syn::ExprWhile) -> JsonValue {
    json!({ "kind": "while", "cond": lower_expr(&w.cond), "body": lower_block_stmts(&w.body) })
}

fn lower_for(f: &syn::ExprForLoop) -> JsonValue {
    json!({ "kind": "for", "pat": pat_to_string(&f.pat), "iter": lower_expr(&f.expr), "body": lower_block_stmts(&f.body) })
}

fn lower_loop(l: &syn::ExprLoop) -> JsonValue {
    json!({ "kind": "loop", "body": lower_block_stmts(&l.body) })
}

fn lower_match(m: &syn::ExprMatch) -> JsonValue {
    let arms: Vec<JsonValue> = m
        .arms
        .iter()
        .map(|arm| {
            json!({
                "pattern": pat_to_string(&arm.pat),
                "body":    lower_expr(&arm.body),
                "guard":   arm.guard.as_ref().map(|(_, g)| lower_expr(g)),
            })
        })
        .collect();
    json!({ "kind": "match", "expr": lower_expr(&m.expr), "arms": arms })
}

fn lower_return(r: &syn::ExprReturn) -> JsonValue {
    json!({ "kind": "return", "value": r.expr.as_deref().map(lower_expr) })
}

fn lower_break(b: &syn::ExprBreak) -> JsonValue {
    json!({ "kind": "break", "value": b.expr.as_deref().map(lower_expr) })
}

fn lower_assign(a: &syn::ExprAssign) -> JsonValue {
    json!({ "kind": "assign", "target": lower_expr(&a.left), "value": lower_expr(&a.right) })
}

pub(crate) fn lower_macro_expr(m: &syn::ExprMacro) -> JsonValue {
    let tokens = m.mac.tokens.to_string();
    json!({ "kind": "call", "func": path_to_str(&m.mac.path), "args": [{ "kind": "lit", "value": tokens }] })
}
