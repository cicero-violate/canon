//! Emit valid Rust source from ModelIR.
//!
//! Variables:
//!   ir      : &ModelIR              — source of truth
//!   out     : String                — accumulated source text
//!   indent  : usize                 — current indentation level
//!
//! Equations:
//!   emit(Module m) = concat( emit(n) for n in children(m) in module_graph )
//!   emit(Struct s) = vis "struct" name generics "{" fields "}"
//!   emit(Fn f)     = vis "fn" name generics "(" params ")" "->" ret "{" body "}"
//!   emit(Body)     = emit_blocks(blocks) | raw_source

use model::ir::{
    model_ir::ModelIR,
    node::{BasicBlock, Body, Field, GenericParam, NodeId, NodeKind, Param, Stmt, Terminator, TraitMethod, Visibility},
    edge::EdgeKind,
};

// ── public entry ────────────────────────────────────────────────────────────

/// Emit all modules in the IR as (file_path, source) pairs.
pub fn emit_files(ir: &ModelIR) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for node in &ir.nodes {
        if let NodeKind::Module { file, .. } = &node.kind {
            // Collect children of this module via module_graph.
            let src = emit_module_children(ir, node.id);
            out.push((file.clone(), src));
        }
    }
    out
}

// ── module ───────────────────────────────────────────────────────────────────

fn emit_module_children(ir: &ModelIR, module_id: NodeId) -> String {
    let mut out = String::new();
    // Walk Contains edges from this module node.
    if module_id.index() < ir.module_graph.vertex_count() {
        for (child_id, _edge) in ir.module_graph.neighbours(module_id) {
            out.push_str(&emit_node(ir, child_id));
            out.push('\n');
        }
    }
    out
}

// ── node dispatch ────────────────────────────────────────────────────────────

pub fn emit_node(ir: &ModelIR, id: NodeId) -> String {
    let node = ir.node(id);
    match &node.kind {
        NodeKind::Struct  { name, vis, generics, fields }                   => emit_struct(vis, name, generics, fields),
        NodeKind::Trait   { name, vis, generics, methods }                  => emit_trait(vis, name, generics, methods),
        NodeKind::Impl    { for_struct, for_trait, generics }               => emit_impl(ir, id, for_struct, for_trait, generics),
        NodeKind::Function{ name, vis, generics, params, ret, body }        => emit_fn(vis, name, generics, params, ret, body),
        NodeKind::Method  { name, vis, generics, params, ret, body }        => emit_fn(vis, name, generics, params, ret, body),
        NodeKind::Module  { path, .. }                                      => format!("// module: {}\n", path),
        NodeKind::Crate   { name, .. }                                      => format!("// crate: {}\n", name),
        NodeKind::TypeRef { name }                                           => format!("// type: {}\n", name),
    }
}

// ── struct ───────────────────────────────────────────────────────────────────

fn emit_struct(vis: &Visibility, name: &str, generics: &[GenericParam], fields: &[Field]) -> String {
    let mut s = format!("{}struct {}{} {{\n", vis.to_token(), name, emit_generics(generics));
    for f in fields {
        s.push_str(&emit_field(f));
    }
    s.push_str("}\n");
    s
}

fn emit_field(f: &Field) -> String {
    match &f.name {
        Some(n) => format!("    {}{}: {},\n", f.vis.to_token(), n, f.ty),
        None    => format!("    {}{},\n",     f.vis.to_token(), f.ty),
    }
}

// ── trait ────────────────────────────────────────────────────────────────────

fn emit_trait(vis: &Visibility, name: &str, generics: &[GenericParam], methods: &[TraitMethod]) -> String {
    let mut s = format!("{}trait {}{} {{\n", vis.to_token(), name, emit_generics(generics));
    for m in methods {
        s.push_str(&emit_trait_method(m));
    }
    s.push_str("}\n");
    s
}

fn emit_trait_method(m: &TraitMethod) -> String {
    let sig = format!("    {}fn {}{}{} -> {}",
        m.vis.to_token(), m.name, emit_generics(&m.generics),
        emit_params(&m.params), m.ret);
    match &m.body {
        Body::None       => format!("{};\n", sig),
        Body::Blocks(bb) => format!("{} {{\n{}}}\n", sig, emit_blocks(bb, 2)),
        Body::Raw(src)   => format!("{} {{\n        {}\n    }}\n", sig, src),
    }
}

// ── impl ─────────────────────────────────────────────────────────────────────

fn emit_impl(ir: &ModelIR, impl_id: NodeId, for_struct: &str, for_trait: &Option<String>, generics: &[GenericParam]) -> String {
    let header = match for_trait {
        Some(tr) => format!("impl{} {} for {} {{\n", emit_generics(generics), tr, for_struct),
        None     => format!("impl{} {} {{\n",        emit_generics(generics), for_struct),
    };
    let mut s = header;
    // Methods are children of impl via module_graph Contains edges.
    if impl_id.index() < ir.module_graph.vertex_count() {
        for (child_id, _) in ir.module_graph.neighbours(impl_id) {
            let child = ir.node(child_id);
            if let NodeKind::Method { name, vis, generics, params, ret, body } = &child.kind {
                let method_src = emit_fn(vis, name, generics, params, ret, body);
                // indent one level
                for line in method_src.lines() {
                    s.push_str("    ");
                    s.push_str(line);
                    s.push('\n');
                }
            }
        }
    }
    s.push_str("}\n");
    s
}

// ── function / method ────────────────────────────────────────────────────────

fn emit_fn(vis: &Visibility, name: &str, generics: &[GenericParam], params: &[Param], ret: &str, body: &Body) -> String {
    let sig = format!("{}fn {}{}{}", vis.to_token(), name, emit_generics(generics), emit_params(params));
    let ret_part = if ret == "()" { String::new() } else { format!(" -> {}", ret) };
    let sig = format!("{}{}", sig, ret_part);
    match body {
        Body::None       => format!("{};\n", sig),
        Body::Blocks(bb) => format!("{} {{\n{}}}\n", sig, emit_blocks(bb, 1)),
        Body::Raw(src)   => format!("{} {{\n    {}\n}}\n", sig, src),
    }
}

// ── body / CFG ───────────────────────────────────────────────────────────────

fn emit_blocks(blocks: &[BasicBlock], indent: usize) -> String {
    // Simple linear emit: follow Goto chain from block 0.
    // Branch blocks emit if/else.
    if blocks.is_empty() { return String::new(); }
    let pad = "    ".repeat(indent);
    let mut out = String::new();
    let mut visited = vec![false; blocks.len()];
    emit_block_rec(blocks, 0, &pad, &mut visited, &mut out);
    out
}

fn emit_block_rec(blocks: &[BasicBlock], idx: usize, pad: &str, visited: &mut Vec<bool>, out: &mut String) {
    if idx >= blocks.len() || visited[idx] { return; }
    visited[idx] = true;
    let bb = &blocks[idx];
    for stmt in &bb.stmts {
        out.push_str(&emit_stmt(stmt, pad));
    }
    match &bb.terminator {
        Terminator::Goto(t) => {
            emit_block_rec(blocks, *t as usize, pad, visited, out);
        }
        Terminator::Branch { cond, true_bb, false_bb } => {
            out.push_str(&format!("{}if {} {{\n", pad, cond));
            let inner = format!("{}    ", pad);
            emit_block_rec(blocks, *true_bb as usize, &inner, visited, out);
            out.push_str(&format!("{}}} else {{\n", pad));
            emit_block_rec(blocks, *false_bb as usize, &inner, visited, out);
            out.push_str(&format!("{}}}\n", pad));
        }
        Terminator::Return | Terminator::None => {}
    }
}

fn emit_stmt(stmt: &Stmt, pad: &str) -> String {
    match stmt {
        Stmt::Let { pat, ty, init } => {
            let ty_part   = ty.as_deref().map(|t| format!(": {}", t)).unwrap_or_default();
            let init_part = init.as_deref().map(|e| format!(" = {}", e)).unwrap_or_default();
            format!("{}let {}{}{};\n", pad, pat, ty_part, init_part)
        }
        Stmt::Expr(e)       => format!("{}{};\n", pad, e),
        Stmt::Return(None)  => format!("{}return;\n", pad),
        Stmt::Return(Some(e)) => format!("{}return {};\n", pad, e),
        Stmt::Raw(s)        => format!("{}{}\n", pad, s),
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn emit_generics(gs: &[GenericParam]) -> String {
    if gs.is_empty() { return String::new(); }
    let inner: Vec<String> = gs.iter().map(|g| {
        if g.is_lifetime {
            format!("'{}", g.name)
        } else if g.bounds.is_empty() {
            g.name.clone()
        } else {
            format!("{}: {}", g.name, g.bounds.join(" + "))
        }
    }).collect();
    format!("<{}>", inner.join(", "))
}

fn emit_params(params: &[Param]) -> String {
    let inner: Vec<String> = params.iter().map(|p| {
        if p.is_self {
            if p.mutable { "&mut self".into() } else { "&self".into() }
        } else if p.mutable {
            format!("mut {}: {}", p.name, p.ty)
        } else {
            format!("{}: {}", p.name, p.ty)
        }
    }).collect();
    format!("({})", inner.join(", "))
}
