//! Lower a `syn::Block` into the Canon JSON AST format consumed by
//! `FunctionMetadata::ast` and the `render_fn/ast.rs` renderer.
//!
//! Every node carries a `"kind"` discriminant matching the whitelist in
//! `validate/check_artifacts.rs::check_ast_node_kinds`.

use serde_json::{Value as JsonValue, json};

/// Entry point: lower a full function body block.
/// Returns `None` when the block has no statements (so `metadata.ast`
/// stays absent rather than storing an empty object).
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

// ── Statements ────────────────────────────────────────────────────────────────

fn lower_stmt(stmt: &syn::Stmt) -> Option<JsonValue> {
    match stmt {
        syn::Stmt::Local(local) => Some(lower_let(local)),
        syn::Stmt::Expr(expr, semi) => {
            if semi.is_some() {
                Some(lower_expr_stmt(expr))
            } else {
                // Tail expression — treat as implicit return value
                Some(lower_expr_as_stmt(expr))
            }
        }
        // Inline item definitions (fn, struct inside fn) — skip
        syn::Stmt::Item(_) => None,
        syn::Stmt::Macro(mac) => Some(lower_macro_stmt(mac)),
    }
}

fn lower_let(local: &syn::Local) -> JsonValue {
    let name = pat_to_string(&local.pat);
    let mutable = pat_is_mut(&local.pat);
    let value = local.init.as_ref().map(|init| lower_expr(&init.expr));
    match value {
        Some(v) => json!({
            "kind": "let",
            "name": name,
            "mutable": mutable,
            "value": v,
        }),
        None => json!({
            "kind": "let",
            "name": name,
            "mutable": mutable,
        }),
    }
}

fn lower_expr_stmt(expr: &syn::Expr) -> JsonValue {
    // Expressions with semicolons — special-case control flow nodes that have
    // their own statement kinds; everything else becomes an expression_stmt.
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
        syn::Expr::Block(b) => lower_block_expr(b),
        syn::Expr::Macro(m) => lower_macro_expr(m),
        _ => {
            let inner = lower_expr(expr);
            json!({ "kind": "call", "func": expr_to_call_str(expr), "args": [], "_expr": inner })
        }
    }
}

fn lower_expr_as_stmt(expr: &syn::Expr) -> JsonValue {
    // Tail / no-semi expression — lower directly as an expression node
    lower_expr(expr)
}

fn lower_macro_stmt(mac: &syn::StmtMacro) -> JsonValue {
    let path = path_to_str(&mac.mac.path);
    let tokens = mac.mac.tokens.to_string();
    json!({ "kind": "call", "func": path, "args": [{ "kind": "lit", "value": tokens }] })
}

// ── Expressions ──────────────────────────────────────────────────────────────

pub(crate) fn lower_expr(expr: &syn::Expr) -> JsonValue {
    match expr {
        syn::Expr::Lit(l) => lower_lit(l),
        syn::Expr::Path(p) => {
            let s = path_to_str(&p.path);
            json!({ "kind": "lit", "value": s })
        }
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
        syn::Expr::Block(b) => lower_block_expr(b),
        syn::Expr::Return(r) => lower_return(r),
        syn::Expr::Break(b) => lower_break(b),
        syn::Expr::Continue(_) => json!({ "kind": "continue" }),
        syn::Expr::Assign(a) => lower_assign(a),
        syn::Expr::While(w) => lower_while(w),
        syn::Expr::ForLoop(f) => lower_for(f),
        syn::Expr::Loop(l) => lower_loop(l),
        syn::Expr::Paren(p) => lower_expr(&p.expr),
        syn::Expr::Group(g) => lower_expr(&g.expr),
        syn::Expr::Try(t) => {
            let inner = lower_expr(&t.expr);
            json!({ "kind": "question", "expr": inner })
        }
        syn::Expr::Repeat(r) => {
            let elem = lower_expr(&r.expr);
            let len = lower_expr(&r.len);
            json!({ "kind": "array", "elems": [elem], "repeat_len": len })
        }
        syn::Expr::Macro(m) => lower_macro_expr(m),
        syn::Expr::Await(a) => {
            let inner = lower_expr(&a.base);
            json!({ "kind": "method", "receiver": inner, "method": "await", "args": [] })
        }
        syn::Expr::Async(a) => {
            let body = lower_block(&a.block);
            json!({ "kind": "closure", "params": [], "body": body, "async": true })
        }
        syn::Expr::Unsafe(u) => lower_block_stmts(&u.block),
        _ => json!({ "kind": "lit", "value": "()" }),
    }
}

// ── Individual expression lowerers ───────────────────────────────────────────

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
    let lhs = lower_expr(&b.left);
    let rhs = lower_expr(&b.right);
    let (kind, op) = bin_op_to_kind(&b.op);
    json!({ "kind": kind, "lhs": lhs, "rhs": rhs, "op": op })
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
    let expr = lower_expr(&u.expr);
    json!({ "kind": "unary", "op": op, "expr": expr })
}

fn lower_call(c: &syn::ExprCall) -> JsonValue {
    let func = expr_to_call_str(&c.func);
    let args: Vec<JsonValue> = c.args.iter().map(lower_expr).collect();
    json!({ "kind": "call", "func": func, "args": args })
}

fn lower_method_call(m: &syn::ExprMethodCall) -> JsonValue {
    let receiver = lower_expr(&m.receiver);
    let method = m.method.to_string();
    let args: Vec<JsonValue> = m.args.iter().map(lower_expr).collect();
    json!({ "kind": "method", "receiver": receiver, "method": method, "args": args })
}

fn lower_field(f: &syn::ExprField) -> JsonValue {
    let expr = lower_expr(&f.base);
    let field = match &f.member {
        syn::Member::Named(ident) => ident.to_string(),
        syn::Member::Unnamed(idx) => idx.index.to_string(),
    };
    json!({ "kind": "field", "expr": expr, "field": field })
}

fn lower_index(i: &syn::ExprIndex) -> JsonValue {
    let expr = lower_expr(&i.expr);
    let index = lower_expr(&i.index);
    json!({ "kind": "index", "expr": expr, "index": index })
}

fn lower_struct_lit(s: &syn::ExprStruct) -> JsonValue {
    let name = path_to_str(&s.path);
    let fields: Vec<JsonValue> = s
        .fields
        .iter()
        .map(|fv| {
            let fname = match &fv.member {
                syn::Member::Named(i) => i.to_string(),
                syn::Member::Unnamed(i) => i.index.to_string(),
            };
            let fvalue = lower_expr(&fv.expr);
            json!({ "name": fname, "value": fvalue })
        })
        .collect();
    json!({ "kind": "struct_lit", "name": name, "fields": fields })
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
    let expr = lower_expr(&r.expr);
    let is_mut = r.mutability.is_some();
    json!({ "kind": "ref", "expr": expr, "mut": is_mut })
}

fn lower_range(r: &syn::ExprRange) -> JsonValue {
    let from = r.start.as_deref().map(lower_expr);
    let to = r.end.as_deref().map(lower_expr);
    let inclusive = matches!(r.limits, syn::RangeLimits::Closed(_));
    json!({ "kind": "range", "from": from, "to": to, "inclusive": inclusive })
}

fn lower_cast(c: &syn::ExprCast) -> JsonValue {
    let expr = lower_expr(&c.expr);
    let ty = type_to_str(&c.ty);
    json!({ "kind": "cast", "expr": expr, "ty": ty })
}

fn lower_closure(c: &syn::ExprClosure) -> JsonValue {
    let params: Vec<String> = c.inputs.iter().map(pat_to_string).collect();
    let is_move = c.capture.is_some();
    let body = lower_expr(&c.body);
    json!({ "kind": "closure", "params": params, "move": is_move, "body": body })
}

fn lower_if(i: &syn::ExprIf) -> JsonValue {
    let cond = lower_expr(&i.cond);
    let then = lower_block_stmts(&i.then_branch);
    let else_branch = i.else_branch.as_ref().map(|(_, e)| lower_expr(e));
    json!({ "kind": "if", "cond": cond, "then": then, "else": else_branch })
}

fn lower_while(w: &syn::ExprWhile) -> JsonValue {
    let cond = lower_expr(&w.cond);
    let body = lower_block_stmts(&w.body);
    json!({ "kind": "while", "cond": cond, "body": body })
}

fn lower_for(f: &syn::ExprForLoop) -> JsonValue {
    let pat = pat_to_string(&f.pat);
    let iter = lower_expr(&f.expr);
    let body = lower_block_stmts(&f.body);
    json!({ "kind": "for", "pat": pat, "iter": iter, "body": body })
}

fn lower_loop(l: &syn::ExprLoop) -> JsonValue {
    let body = lower_block_stmts(&l.body);
    json!({ "kind": "loop", "body": body })
}

fn lower_match(m: &syn::ExprMatch) -> JsonValue {
    let expr = lower_expr(&m.expr);
    let arms: Vec<JsonValue> = m
        .arms
        .iter()
        .map(|arm| {
            let pattern = pat_to_string(&arm.pat);
            let body = lower_expr(&arm.body);
            let guard = arm.guard.as_ref().map(|(_, g)| lower_expr(g));
            json!({ "pattern": pattern, "body": body, "guard": guard })
        })
        .collect();
    json!({ "kind": "match", "expr": expr, "arms": arms })
}

fn lower_block_expr(b: &syn::ExprBlock) -> JsonValue {
    lower_block_stmts(&b.block)
}

fn lower_block_stmts(block: &syn::Block) -> JsonValue {
    let stmts: Vec<JsonValue> = block.stmts.iter().filter_map(lower_stmt).collect();
    json!({ "kind": "block", "stmts": stmts })
}

fn lower_return(r: &syn::ExprReturn) -> JsonValue {
    let value = r.expr.as_deref().map(lower_expr);
    json!({ "kind": "return", "value": value })
}

fn lower_break(b: &syn::ExprBreak) -> JsonValue {
    let value = b.expr.as_deref().map(lower_expr);
    json!({ "kind": "break", "value": value })
}

fn lower_assign(a: &syn::ExprAssign) -> JsonValue {
    let target = lower_expr(&a.left);
    let value = lower_expr(&a.right);
    json!({ "kind": "assign", "target": target, "value": value })
}

fn lower_macro_expr(m: &syn::ExprMacro) -> JsonValue {
    let path = path_to_str(&m.mac.path);
    let tokens = m.mac.tokens.to_string();
    json!({ "kind": "call", "func": path, "args": [{ "kind": "lit", "value": tokens }] })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn expr_to_call_str(expr: &syn::Expr) -> String {
    match expr {
        syn::Expr::Path(p) => path_to_str(&p.path),
        syn::Expr::Field(f) => {
            let base = expr_to_call_str(&f.base);
            let member = match &f.member {
                syn::Member::Named(i) => i.to_string(),
                syn::Member::Unnamed(i) => i.index.to_string(),
            };
            format!("{base}.{member}")
        }
        _ => "fn_ptr".to_owned(),
    }
}

fn path_to_str(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

fn type_to_str(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(p) => path_to_str(&p.path),
        syn::Type::Reference(_) => "&_".to_owned(),
        syn::Type::Slice(_) => "[_]".to_owned(),
        syn::Type::Array(_) => "[_; N]".to_owned(),
        syn::Type::Tuple(_) => "()".to_owned(),
        _ => "_".to_owned(),
    }
}

fn pat_to_string(pat: &syn::Pat) -> String {
    match pat {
        syn::Pat::Ident(i) => i.ident.to_string(),
        syn::Pat::Wild(_) => "_".to_owned(),
        syn::Pat::Tuple(t) => {
            let parts: Vec<_> = t.elems.iter().map(pat_to_string).collect();
            format!("({})", parts.join(", "))
        }
        syn::Pat::TupleStruct(ts) => {
            let name = path_to_str(&ts.path);
            let fields: Vec<_> = ts.elems.iter().map(pat_to_string).collect();
            format!("{}({})", name, fields.join(", "))
        }
        syn::Pat::Struct(s) => {
            let name = path_to_str(&s.path);
            let fields: Vec<_> = s
                .fields
                .iter()
                .map(|f| match &f.member {
                    syn::Member::Named(i) => i.to_string(),
                    syn::Member::Unnamed(i) => i.index.to_string(),
                })
                .collect();
            format!("{} {{ {} }}", name, fields.join(", "))
        }
        syn::Pat::Path(p) => path_to_str(&p.path),
        syn::Pat::Lit(l) => match &l.lit {
            syn::Lit::Int(i) => i.base10_digits().to_owned(),
            syn::Lit::Str(s) => format!("\"{}\"", s.value()),
            syn::Lit::Bool(b) => b.value.to_string(),
            syn::Lit::Char(c) => format!("'{}'", c.value()),
            syn::Lit::Byte(b) => b.value().to_string(),
            _ => "_".to_owned(),
        },
        syn::Pat::Range(_) => "_..=_".to_owned(),
        syn::Pat::Reference(r) => {
            let inner = pat_to_string(&r.pat);
            format!("&{inner}")
        }
        syn::Pat::Or(o) => {
            let parts: Vec<_> = o.cases.iter().map(pat_to_string).collect();
            parts.join(" | ")
        }
        syn::Pat::Slice(s) => {
            let parts: Vec<_> = s.elems.iter().map(pat_to_string).collect();
            format!("[{}]", parts.join(", "))
        }
        _ => "_".to_owned(),
    }
}

fn pat_is_mut(pat: &syn::Pat) -> bool {
    match pat {
        syn::Pat::Ident(i) => i.mutability.is_some(),
        _ => false,
    }
}
