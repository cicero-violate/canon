//! Emit valid Rust source from ModelIR.
//!
//! Variables:
//!   ir     : &ModelIR   — source of truth
//!   pad    : &str       — indentation prefix
//!
//! Equations:
//!   emit(n)         = Emit::emit(&emitter_for(n), n, ir, "")
//!   emit(Module m)  = concat( emit(child) for child in module_graph.neighbours(m) )
//!   emit(Struct s)  = vis "struct" name generics "{" fields "}"
//!   emit(Fn f)      = vis "fn" name generics params ret "{" body "}"
//!   emit(Body)      = emit_blocks(blocks, pad) | indent_raw(src, pad)

use model::ir::{
    edge::EdgeKind,
    model_ir::ModelIR,
    node::{
        BasicBlock, Body, Field, GenericParam, NodeId, NodeKind,
        Param, Stmt, Terminator, TraitMethod, Visibility,
    },
};

// ═══════════════════════════════════════════════════════════════════════════
// TypeAliasEmitter
// ═══════════════════════════════════════════════════════════════════════════

struct TypeAliasEmitter<'a> {
    name:     &'a str,
    vis:      &'a Visibility,
    generics: &'a [GenericParam],
    ty:       &'a str,
}

impl Emit for TypeAliasEmitter<'_> {
    /// Equation:
    ///   emit(TypeAlias) = vis "type" name generics "=" ty ";"
    fn emit(&self, _ir: &ModelIR, pad: &str) -> String {
        format!("{}{}type {}{} = {};\n",
            pad, self.vis.to_token(), self.name,
            fmt_generics(self.generics), self.ty)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Core trait
// ═══════════════════════════════════════════════════════════════════════════

/// Every NodeKind has a corresponding emitter that implements this trait.
///
/// Equation:
///   Emit::emit(self, ir, pad) -> String
trait Emit {
    fn emit(&self, ir: &ModelIR, pad: &str) -> String;
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry
// ═══════════════════════════════════════════════════════════════════════════

/// Emit all Module nodes as (file_path, source) pairs.
pub fn emit_files(ir: &ModelIR) -> Vec<(String, String)> {
    ir.nodes.iter()
        .filter_map(|n| {
            if let NodeKind::Module { file, .. } = &n.kind {
                Some((file.clone(), ModuleEmitter(n.id).emit(ir, "")))
            } else {
                None
            }
        })
        .collect()
}

/// Emit a single node by id.
pub fn emit_node(ir: &ModelIR, id: NodeId) -> String {
    dispatch(ir, id, "")
}

// ═══════════════════════════════════════════════════════════════════════════
// Dispatch
// ═══════════════════════════════════════════════════════════════════════════

fn dispatch(ir: &ModelIR, id: NodeId, pad: &str) -> String {
    let node = ir.node(id);
    match &node.kind {
        NodeKind::Module   { .. }                                     => ModuleEmitter(id).emit(ir, pad),
        NodeKind::Struct   { name, vis, generics, fields }            => StructEmitter { name, vis, generics, fields }.emit(ir, pad),
        NodeKind::Trait    { name, vis, generics, methods }           => TraitEmitter  { name, vis, generics, methods }.emit(ir, pad),
        NodeKind::Impl     { for_struct, for_trait, generics }        => ImplEmitter   { id, for_struct, for_trait, generics }.emit(ir, pad),
        NodeKind::Function { name, vis, generics, params, ret, body } => FnEmitter     { name, vis, generics, params, ret, body }.emit(ir, pad),
        NodeKind::Method   { name, vis, generics, params, ret, body } => FnEmitter     { name, vis, generics, params, ret, body }.emit(ir, pad),
        NodeKind::TypeRef  { name }                                   => TypeRefEmitter { name }.emit(ir, pad),
        NodeKind::Crate    { .. }                                     => String::new(),
        NodeKind::TypeAlias { name, vis, generics, ty }               => TypeAliasEmitter { name, vis, generics, ty }.emit(ir, pad),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ModuleEmitter
// ═══════════════════════════════════════════════════════════════════════════

struct ModuleEmitter(NodeId);

impl Emit for ModuleEmitter {
    /// Equation:
    ///   emit(Module m) =
    ///     for (child, Contains) in module_graph.neighbours(m):
    ///       if child is Module  -> "pub mod <name>;"
    ///       else                -> emit(child)
    fn emit(&self, ir: &ModelIR, pad: &str) -> String {
        let mut out = String::new();
        let id = self.0;
        if id.index() >= ir.module_graph.vertex_count() {
            return out;
        }
        for (child_id, edge) in ir.module_graph.neighbours(id) {
            if *edge != EdgeKind::Contains { continue; }
            let child = ir.node(child_id);
            match &child.kind {
                // Sub-module declared inline as `pub mod name;`
                NodeKind::Module { path, .. } => {
                    let name = path.rsplit("::").next().unwrap_or(path.as_str());
                    out.push_str(&format!("{}pub mod {};\n", pad, name));
                }
                // Everything else emits its full source
                _ => {
                    let src = dispatch(ir, child_id, pad);
                    out.push_str(&src);
                    out.push('\n');
                }
            }
        }
        out
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// StructEmitter
// ═══════════════════════════════════════════════════════════════════════════

struct StructEmitter<'a> {
    name:     &'a str,
    vis:      &'a Visibility,
    generics: &'a [GenericParam],
    fields:   &'a [Field],
}

impl Emit for StructEmitter<'_> {
    /// Equation:
    ///   emit(Struct) = vis "struct" name generics "{" fields "}"
    fn emit(&self, _ir: &ModelIR, pad: &str) -> String {
        let mut s = format!("{}{}struct {}{} {{\n",
            pad, self.vis.to_token(), self.name, fmt_generics(self.generics));
        for f in self.fields {
            s.push_str(&fmt_field(f, &format!("{}    ", pad)));
        }
        s.push_str(&format!("{}}}\n", pad));
        s
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TraitEmitter
// ═══════════════════════════════════════════════════════════════════════════

struct TraitEmitter<'a> {
    name:     &'a str,
    vis:      &'a Visibility,
    generics: &'a [GenericParam],
    methods:  &'a [TraitMethod],
}

impl Emit for TraitEmitter<'_> {
    /// Equation:
    ///   emit(Trait) = vis "trait" name generics "{" methods "}"
    fn emit(&self, ir: &ModelIR, pad: &str) -> String {
        let inner = format!("{}    ", pad);
        let mut s = format!("{}{}trait {}{} {{\n",
            pad, self.vis.to_token(), self.name, fmt_generics(self.generics));
        for m in self.methods {
            s.push_str(&fmt_trait_method(m, ir, &inner));
        }
        s.push_str(&format!("{}}}\n", pad));
        s
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ImplEmitter
// ═══════════════════════════════════════════════════════════════════════════

struct ImplEmitter<'a> {
    id:         NodeId,
    for_struct: &'a str,
    for_trait:  &'a Option<String>,
    generics:   &'a [GenericParam],
}

impl Emit for ImplEmitter<'_> {
    /// Equation:
    ///   emit(Impl) = "impl" generics [trait "for"] struct "{" methods "}"
    ///   methods    = children via module_graph Contains edges
    fn emit(&self, ir: &ModelIR, pad: &str) -> String {
        let header = match self.for_trait {
            Some(tr) => format!("{}impl{} {} for {} {{\n",
                pad, fmt_generics(self.generics), tr, self.for_struct),
            None     => format!("{}impl{} {} {{\n",
                pad, fmt_generics(self.generics), self.for_struct),
        };
        let mut s = header;
        let inner = format!("{}    ", pad);
        if self.id.index() < ir.module_graph.vertex_count() {
            for (child_id, edge) in ir.module_graph.neighbours(self.id) {
                if *edge != EdgeKind::Contains { continue; }
                let src = dispatch(ir, child_id, &inner);
                s.push_str(&src);
            }
        }
        s.push_str(&format!("{}}}\n", pad));
        s
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FnEmitter  (Function + Method share the same layout)
// ═══════════════════════════════════════════════════════════════════════════

struct FnEmitter<'a> {
    name:     &'a str,
    vis:      &'a Visibility,
    generics: &'a [GenericParam],
    params:   &'a [Param],
    ret:      &'a str,
    body:     &'a Body,
}

impl Emit for FnEmitter<'_> {
    /// Equation:
    ///   emit(Fn) = vis "fn" name generics "(" params ")" ["->" ret] "{" body "}"
    fn emit(&self, _ir: &ModelIR, pad: &str) -> String {
        let ret_part = if self.ret == "()" {
            String::new()
        } else {
            format!(" -> {}", self.ret)
        };
        let sig = format!("{}{}fn {}{}{}{}", pad,
            self.vis.to_token(), self.name,
            fmt_generics(self.generics), fmt_params(self.params), ret_part);
        let inner = format!("{}    ", pad);
        match self.body {
            Body::None       => format!("{};\n", sig),
            Body::Blocks(bb) => format!("{} {{\n{}{}}}\n", sig, emit_blocks(bb, &inner), pad),
            Body::Raw(src)   => format!("{} {{\n{}{}}}\n", sig, indent_raw(src, &inner), pad),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TypeRefEmitter
// ═══════════════════════════════════════════════════════════════════════════

struct TypeRefEmitter<'a> {
    name: &'a str,
}

impl Emit for TypeRefEmitter<'_> {
    /// TypeRef with no expansion emits a placeholder comment.
    /// Use NodeKind::TypeAlias (future) for full `type X = Y;` emit.
    fn emit(&self, _ir: &ModelIR, pad: &str) -> String {
        format!("{}// type alias: {}\n", pad, self.name)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Body / CFG helpers
// ═══════════════════════════════════════════════════════════════════════════

fn emit_blocks(blocks: &[BasicBlock], pad: &str) -> String {
    if blocks.is_empty() { return String::new(); }
    let mut out     = String::new();
    let mut visited = vec![false; blocks.len()];
    emit_block_rec(blocks, 0, pad, &mut visited, &mut out);
    out
}

fn emit_block_rec(
    blocks:  &[BasicBlock],
    idx:     usize,
    pad:     &str,
    visited: &mut Vec<bool>,
    out:     &mut String,
) {
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
            let inner = format!("{}    ", pad);
            out.push_str(&format!("{}if {} {{\n", pad, cond));
            emit_block_rec(blocks, *true_bb  as usize, &inner, visited, out);
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
            let ty_s   = ty.as_deref().map(|t| format!(": {}", t)).unwrap_or_default();
            let init_s = init.as_deref().map(|e| format!(" = {}", e)).unwrap_or_default();
            format!("{}let {}{}{};\n", pad, pat, ty_s, init_s)
        }
        Stmt::Expr(e)         => format!("{}{};\n",        pad, e),
        Stmt::Return(None)    => format!("{}return;\n",    pad),
        Stmt::Return(Some(e)) => format!("{}return {};\n", pad, e),
        Stmt::Raw(s)          => format!("{}{}\n",         pad, s),
    }
}

/// Indent every line of a raw body string by `pad`.
fn indent_raw(src: &str, pad: &str) -> String {
    src.lines()
        .map(|line| format!("{}{}\n", pad, line))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Trait method helper (not a NodeKind — lives inside Trait node directly)
// ═══════════════════════════════════════════════════════════════════════════

fn fmt_trait_method(m: &TraitMethod, ir: &ModelIR, pad: &str) -> String {
    let ret_part = if m.ret == "()" { String::new() } else { format!(" -> {}", m.ret) };
    let sig = format!("{}{}fn {}{}{}{}",
        pad, m.vis.to_token(), m.name,
        fmt_generics(&m.generics), fmt_params(&m.params), ret_part);
    let inner = format!("{}    ", pad);
    match &m.body {
        Body::None       => format!("{};\n", sig),
        Body::Blocks(bb) => format!("{} {{\n{}{}}}\n", sig, emit_blocks(bb, &inner), pad),
        Body::Raw(src)   => format!("{} {{\n{}}}\n",   sig, indent_raw(src, &inner)),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Formatting helpers
// ═══════════════════════════════════════════════════════════════════════════

fn fmt_generics(gs: &[GenericParam]) -> String {
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

fn fmt_params(params: &[Param]) -> String {
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

fn fmt_field(f: &Field, pad: &str) -> String {
    match &f.name {
        Some(n) => format!("{}{}{}: {},\n", pad, f.vis.to_token(), n, f.ty),
        None    => format!("{}{}{},\n",     pad, f.vis.to_token(), f.ty),
    }
}
