//! Node identity and kind for ModelIR.
//!
//! Variables:
//!   NodeId       = dense u32 index into the node arena
//!   NodeKind     = discriminated union of all IR item types
//!   GenericParam = (name, bounds)
//!   Field        = (name, ty, vis)
//!   BodyNode     = one statement/expression in a function body CFG
//!   Terminator   = how a basic block exits
//!
//! Equation:
//!   node_arena[NodeId] -> NodeKind

use serde::{Deserialize, Serialize};

/// Opaque dense node index. All graphs share the same NodeId space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

impl NodeId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// `pub`, `pub(crate)`, `pub(super)`, or private.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Visibility {
    Public,
    PubCrate,
    PubSuper,
    PubIn(String),
    Private,
}

impl Visibility {
    pub fn to_token(&self) -> &str {
        match self {
            Visibility::Public => "pub ",
            Visibility::PubCrate => "pub(crate) ",
            Visibility::PubSuper => "pub(super) ",
            Visibility::PubIn(_) => "pub(in ...) ",
            Visibility::Private => "",
        }
    }
}

/// A generic parameter: `T`, `T: Clone + Debug`, `'a`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenericParam {
    pub name: String,
    pub bounds: Vec<String>, // e.g. ["Clone", "Debug"]
    pub is_lifetime: bool,
}

/// A struct or tuple field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Field {
    pub name: Option<String>, // None = tuple field
    pub ty: String,
    pub vis: Visibility,
}

/// A function parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub ty: String,
    pub is_self: bool,
    pub mutable: bool,
}

/// A single statement or expression inside a function body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Stmt {
    /// `let <pat>: <ty> = <expr>;`
    Let { pat: String, ty: Option<String>, init: Option<String> },
    /// A bare expression statement.
    Expr(String),
    /// A return expression.  `return <expr>;`
    Return(Option<String>),
    /// Raw source verbatim (escape hatch for complex bodies).
    Raw(String),
}

/// How a basic block exits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Terminator {
    /// Falls through to block index `target`.
    Goto(u32),
    /// `if <cond> { goto true_bb } else { goto false_bb }`
    Branch { cond: String, true_bb: u32, false_bb: u32 },
    /// Function returns (value already in last Stmt::Return).
    Return,
    /// Unreachable / not yet assigned.
    None,
}

/// A basic block: list of statements + terminator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BasicBlock {
    pub stmts: Vec<Stmt>,
    pub terminator: Terminator,
}

/// Inline body for a function or method.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Body {
    /// No body (trait declaration, extern).
    None,
    /// Structured basic-block CFG.
    Blocks(Vec<BasicBlock>),
    /// Raw source verbatim (escape hatch).
    Raw(String),
}

/// A trait method signature (with optional default body).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraitMethod {
    pub name: String,
    pub vis: Visibility,
    pub generics: Vec<GenericParam>,
    pub params: Vec<Param>,
    pub ret: String, // "()" if unit
    pub body: Body,
}

/// Every item in the IR is one of these kinds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    Crate { name: String, edition: String },
    Module { path: String, file: String },
    Struct { name: String, vis: Visibility, generics: Vec<GenericParam>, fields: Vec<Field> },
    Trait { name: String, vis: Visibility, generics: Vec<GenericParam>, methods: Vec<TraitMethod> },
    Impl { for_struct: String, for_trait: Option<String>, generics: Vec<GenericParam> },
    Function { name: String, vis: Visibility, generics: Vec<GenericParam>, params: Vec<Param>, ret: String, body: Body },
    Method { name: String, vis: Visibility, generics: Vec<GenericParam>, params: Vec<Param>, ret: String, body: Body },
    TypeRef { name: String },
    TypeAlias {
        name: String,
        vis: Visibility,
        generics: Vec<GenericParam>,
        /// The right-hand side, e.g. "std::result::Result<T, String>"
        ty: String,
    },
}

/// A node in the arena: identity + kind + source span (optional).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    /// Original source span for round-trip emit, e.g. "src/lib.rs:12:5".
    pub span: Option<String>,
}
