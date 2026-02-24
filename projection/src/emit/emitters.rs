use model::ir::{
    edge::EdgeKind,
    model_ir::ModelIR,
    node::{
        Body, Field, GenericParam, NodeId, NodeKind, Param, TraitMethod, Visibility,
    },
};

use crate::emit::body::{emit_blocks, indent_raw};
use crate::emit::fmt::{fmt_field, fmt_generics, fmt_params, fmt_trait_method};

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
    ir.nodes
        .iter()
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
        NodeKind::Module { .. } => ModuleEmitter(id).emit(ir, pad),
        NodeKind::Struct { name, vis, generics, fields, derives } => {
            StructEmitter { name, vis, generics, fields, derives }.emit(ir, pad)
        }
        NodeKind::Trait { name, vis, generics, methods } => {
            TraitEmitter { name, vis, generics, methods }.emit(ir, pad)
        }
        NodeKind::Impl { for_struct, for_trait, generics } => {
            ImplEmitter { id, for_struct, for_trait, generics }.emit(ir, pad)
        }
        NodeKind::Function { name, vis, generics, params, ret, body } => {
            FnEmitter { name, vis, generics, params, ret, body }.emit(ir, pad)
        }
        NodeKind::Method { name, vis, generics, params, ret, body } => {
            FnEmitter { name, vis, generics, params, ret, body }.emit(ir, pad)
        }
        NodeKind::Use { path, alias } => {
            UseEmitter { path, alias }.emit(ir, pad)
        }
        NodeKind::TypeRef { name } => TypeRefEmitter { name }.emit(ir, pad),
        NodeKind::TypeAlias { name, vis, generics, ty } => {
            TypeAliasEmitter { name, vis, generics, ty }.emit(ir, pad)
        }
        NodeKind::Crate { .. } => String::new(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TypeAliasEmitter
// ═══════════════════════════════════════════════════════════════════════════

struct TypeAliasEmitter<'a> {
    name: &'a str,
    vis: &'a Visibility,
    generics: &'a [GenericParam],
    ty: &'a str,
}

impl Emit for TypeAliasEmitter<'_> {
    /// Equation:
    ///   emit(TypeAlias) = vis "type" name generics "=" ty ";"
    fn emit(&self, _ir: &ModelIR, pad: &str) -> String {
        format!(
            "{}{}type {}{} = {};\n",
            pad,
            self.vis.to_token(),
            self.name,
            fmt_generics(self.generics),
            self.ty
        )
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
    ///       if child is Use     -> emit(Use)          // use declarations first
    ///       if child is Module  -> "pub mod <name>;"
    ///       else                -> emit(child)
    fn emit(&self, ir: &ModelIR, pad: &str) -> String {
        let mut out = String::new();
        let id = self.0;
        if id.index() >= ir.module_graph.vertex_count() {
            return out;
        }

        // Collect children, partitioned: Use nodes first, then the rest.
        let mut use_nodes: Vec<NodeId> = Vec::new();
        let mut other_nodes: Vec<NodeId> = Vec::new();

        for (child_id, edge) in ir.module_graph.neighbours(id) {
            if *edge != EdgeKind::Contains {
                continue;
            }
            let child = ir.node(child_id);
            if matches!(&child.kind, NodeKind::Use { .. }) {
                use_nodes.push(child_id);
            } else {
                other_nodes.push(child_id);
            }
        }

        // Emit use declarations at the top of the file.
        for uid in &use_nodes {
            out.push_str(&dispatch(ir, *uid, pad));
        }
        if !use_nodes.is_empty() {
            out.push('\n');
        }

        // Emit all other children.
        for child_id in &other_nodes {
            let child = ir.node(*child_id);
            match &child.kind {
                NodeKind::Module { path, .. } => {
                    let name = path.rsplit("::").next().unwrap_or(path.as_str());
                    out.push_str(&format!("{}pub mod {};\n", pad, name));
                }
                _ => {
                    let src = dispatch(ir, *child_id, pad);
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
    name: &'a str,
    vis: &'a Visibility,
    generics: &'a [GenericParam],
    fields: &'a [Field],
    derives: &'a [String],
}

impl Emit for StructEmitter<'_> {
    /// Equation:
    ///   emit(Struct) = [#[derive(...)]] vis "struct" name generics "{" fields "}"
    fn emit(&self, _ir: &ModelIR, pad: &str) -> String {
        let mut s = String::new();
        if !self.derives.is_empty() {
            s.push_str(&format!("{}#[derive({})]\n", pad, self.derives.join(", ")));
        }
        s.push_str(&format!(
            "{}{}struct {}{} {{\n",
            pad,
            self.vis.to_token(),
            self.name,
            fmt_generics(self.generics)
        ));
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
    name: &'a str,
    vis: &'a Visibility,
    generics: &'a [GenericParam],
    methods: &'a [TraitMethod],
}

impl Emit for TraitEmitter<'_> {
    /// Equation:
    ///   emit(Trait) = vis "trait" name generics "{" methods "}"
    fn emit(&self, ir: &ModelIR, pad: &str) -> String {
        let inner = format!("{}    ", pad);
        let mut s = format!(
            "{}{}trait {}{} {{\n",
            pad,
            self.vis.to_token(),
            self.name,
            fmt_generics(self.generics)
        );
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
    id: NodeId,
    for_struct: &'a str,
    for_trait: &'a Option<String>,
    generics: &'a [GenericParam],
}

impl Emit for ImplEmitter<'_> {
    /// Equation:
    ///   emit(Impl) = "impl" generics [trait "for"] struct "{" methods "}"
    ///   methods    = children via module_graph Contains edges
    fn emit(&self, ir: &ModelIR, pad: &str) -> String {
        let header = match self.for_trait {
            Some(tr) => format!(
                "{}impl{} {} for {} {{\n",
                pad,
                fmt_generics(self.generics),
                tr,
                self.for_struct
            ),
            None => format!(
                "{}impl{} {} {{\n",
                pad,
                fmt_generics(self.generics),
                self.for_struct
            ),
        };
        let mut s = header;
        let inner = format!("{}    ", pad);
        if self.id.index() < ir.module_graph.vertex_count() {
            for (child_id, edge) in ir.module_graph.neighbours(self.id) {
                if *edge != EdgeKind::Contains {
                    continue;
                }
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
    name: &'a str,
    vis: &'a Visibility,
    generics: &'a [GenericParam],
    params: &'a [Param],
    ret: &'a str,
    body: &'a Body,
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
        let sig = format!(
            "{}{}fn {}{}{}{}",
            pad,
            self.vis.to_token(),
            self.name,
            fmt_generics(self.generics),
            fmt_params(self.params),
            ret_part
        );
        let inner = format!("{}    ", pad);
        match self.body {
            Body::None => format!("{};\n", sig),
            Body::Blocks(bb) => format!("{} {{\n{}{}}}\n", sig, emit_blocks(bb, &inner), pad),
            Body::Raw(src) => format!("{} {{\n{}{}}}\n", sig, indent_raw(src, &inner), pad),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// UseEmitter
// ═══════════════════════════════════════════════════════════════════════════

struct UseEmitter<'a> {
    path: &'a str,
    alias: &'a Option<String>,
}

impl Emit for UseEmitter<'_> {
    /// Equation:
    ///   emit(Use) = "use" path ["as" alias] ";"
    fn emit(&self, _ir: &ModelIR, pad: &str) -> String {
        match self.alias {
            Some(a) => format!("{}use {} as {};\n", pad, self.path, a),
            None    => format!("{}use {};\n", pad, self.path),
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
    fn emit(&self, _ir: &ModelIR, pad: &str) -> String {
        format!("{}// type alias: {}\n", pad, self.name)
    }
}
