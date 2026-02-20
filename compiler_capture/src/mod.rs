//! Integration layer connecting the kernel to external capture frontends.
//!
//! This module now exposes a layered architecture:
//! - [`frontends`] host compiler/language specific collectors.
//! - [`capture`] provides multi-crate orchestration and deduplication tools.
//! - [`transform`] converts captured items into the graph kernel representation.

/// Capture pipeline infrastructure and orchestration.
pub mod capture;
/// Compiler/language-specific frontends.
pub mod frontends;
/// Linux fact capture helpers.
pub mod linux;
/// Compatibility shims when compiler_capture is built standalone.
pub mod compat;
/// Small capability helpers (standalone).
pub mod capability;
/// Multi-target capture utilities for Cargo projects.
pub mod multi_capture;
/// Project helpers for invoking cargo/rustc.
pub mod project;
/// Transformation utilities that normalize captured items into the kernel graph.
pub mod transform;
/// Workspace metadata for capture outputs.
pub mod workspace;

pub mod kernel_root;
pub mod graph;

use std::collections::HashMap;

#[cfg(feature = "compiler_capture_crate")]
pub mod rename {
    pub mod core {
        pub mod symbol_id {
            pub use crate::compat::symbol_id::*;
        }
        pub mod oracle {
            pub use crate::compat::oracle::*;
        }
        pub use crate::compat::oracle::{NullOracle, StructuralEditOracle};
        pub use crate::compat::symbol_id::{normalize_symbol_id, normalize_symbol_id_with_crate};
    }
}
#[cfg(not(feature = "compiler_capture_crate"))]
pub mod rename {
    pub use crate::rename::core;
}
#[cfg(not(feature = "compiler_capture_crate"))]
pub use crate::{fs, rename as rename_mod, state};

use crate::compiler_capture::graph::GraphDelta;
pub use frontends::rustc::{RustcFrontend, RustcFrontendError};

// Allow `crate::compiler_capture::*` when built as a standalone crate.
pub mod compiler_capture {
    pub use crate::*;
}

/// Core trait implemented by language/compiler frontends.
pub trait FrontendExtractor {
    /// Frontend-specific configuration type (paths, flags, etc.).
    type Config;
    /// Error type exposed by the frontend.
    type Error;

    /// Extract metadata from a project configuration.
    fn extract(&mut self, config: Self::Config) -> Result<ExtractionResult, Self::Error>;

    /// Human readable identifier (e.g. `rustc`).
    fn name(&self) -> &str;

    /// Supported file extensions (e.g. `["rs"]`).
    fn supported_extensions(&self) -> &[&str];
}

/// Result of running an extractor for a single project or crate.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    /// Name of the crate that was captured.
    pub crate_name: String,
    /// Raw captured items awaiting normalization.
    pub items: Vec<CapturedItem>,
    /// Structured errors that happened during extraction.
    pub errors: Vec<ExtractionError>,
    /// Human-readable warnings emitted by the frontend.
    pub warnings: Vec<String>,
    /// Summary statistics for captured entities.
    pub stats: ExtractionStats,
    /// Optional graph deltas emitted by the capture.
    pub graph_deltas: Option<Vec<GraphDelta>>,
}

impl ExtractionResult {
    /// Creates an empty result for the provided crate name.
    pub fn empty(crate_name: impl Into<String>) -> Self {
        Self {
            crate_name: crate_name.into(),
            items: Vec::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            stats: ExtractionStats::default(),
            graph_deltas: None,
        }
    }
}

/// Statistics describing what an extractor produced.
#[derive(Debug, Clone, Default)]
pub struct ExtractionStats {
    /// Number of standalone function items captured.
    pub functions_captured: usize,
    /// Number of type definitions captured.
    pub types_captured: usize,
    /// Number of traits captured.
    pub traits_captured: usize,
    /// Number of impl blocks captured.
    pub impls_captured: usize,
    /// Number of modules captured.
    pub modules_captured: usize,
    /// Total item count processed by the frontend.
    pub total_items: usize,
    /// Duration of the extraction, in milliseconds.
    pub duration_ms: u64,
}

/// Generic captured item emitted before normalization to the graph IR.
#[derive(Debug, Clone)]
pub enum CapturedItem {
    /// Captured free or associated function.
    Function(FunctionCapture),
    /// Struct definition.
    Struct(StructCapture),
    /// Enum definition.
    Enum(EnumCapture),
    /// Trait definition.
    Trait(TraitCapture),
    /// Impl block.
    Impl(ImplCapture),
    /// Module boundary and children.
    Module(ModuleCapture),
    /// Type alias definition.
    TypeAlias(TypeAliasCapture),
    /// Const item.
    Const(ConstCapture),
    /// Static item.
    Static(StaticCapture),
}

/// Function metadata captured from a frontend.
#[derive(Debug, Clone, Default)]
pub struct FunctionCapture {
    /// User-visible function name.
    pub name: String,
    /// Fully qualified path.
    pub path: String,
    /// Optional signature representation.
    pub signature: Option<String>,
    /// Arbitrary metadata serialized as string key/values.
    pub metadata: HashMap<String, String>,
}

/// Struct metadata captured from a frontend.
#[derive(Debug, Clone, Default)]
pub struct StructCapture {
    /// Struct name.
    pub name: String,
    /// Fully qualified path.
    pub path: String,
    /// Field names in declaration order.
    pub fields: Vec<String>,
    /// Additional metadata for the struct.
    pub metadata: HashMap<String, String>,
}

/// Enum metadata captured from a frontend.
#[derive(Debug, Clone, Default)]
pub struct EnumCapture {
    /// Enum name.
    pub name: String,
    /// Fully qualified path.
    pub path: String,
    /// Enum variants.
    pub variants: Vec<String>,
    /// Additional metadata for the enum.
    pub metadata: HashMap<String, String>,
}

/// Trait metadata captured from a frontend.
#[derive(Debug, Clone, Default)]
pub struct TraitCapture {
    /// Trait name.
    pub name: String,
    /// Fully qualified path.
    pub path: String,
    /// Associated methods captured for the trait.
    pub methods: Vec<String>,
    /// Supertraits.
    pub supertraits: Vec<String>,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, String>,
}

/// Impl metadata captured from a frontend.
#[derive(Debug, Clone, Default)]
pub struct ImplCapture {
    /// Impl label (normally `<Type as Trait>`).
    pub name: String,
    /// Fully qualified path.
    pub path: String,
    /// Target type name.
    pub target: String,
    /// Referenced trait, if any.
    pub trait_ref: Option<String>,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, String>,
}

/// Module metadata captured from a frontend.
#[derive(Debug, Clone, Default)]
pub struct ModuleCapture {
    /// Module name.
    pub name: String,
    /// Fully qualified path.
    pub path: String,
    /// Child items associated with this module.
    pub children: Vec<String>,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, String>,
}

/// Type alias metadata captured from a frontend.
#[derive(Debug, Clone, Default)]
pub struct TypeAliasCapture {
    /// Alias name.
    pub name: String,
    /// Fully qualified path.
    pub path: String,
    /// Rendered aliased type.
    pub aliased_type: String,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, String>,
}

/// Const metadata captured from a frontend.
#[derive(Debug, Clone, Default)]
pub struct ConstCapture {
    /// Const identifier.
    pub name: String,
    /// Fully qualified path.
    pub path: String,
    /// Optional rendered value.
    pub value_repr: Option<String>,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, String>,
}

/// Static metadata captured from a frontend.
#[derive(Debug, Clone, Default)]
pub struct StaticCapture {
    /// Static identifier.
    pub name: String,
    /// Fully qualified path.
    pub path: String,
    /// Optional rendered value.
    pub value_repr: Option<String>,
    /// Whether the static is mutable.
    pub mutable: bool,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, String>,
}

/// A structured error emitted during extraction.
/// A structured error emitted during extraction.
#[derive(Debug, Clone)]
pub struct ExtractionError {
    /// Stable error code.
    pub code: String,
    /// Human readable explanation.
    pub message: String,
    /// Optional location where the error occurred.
    pub location: Option<SourceLocation>,
}

impl ExtractionError {
    /// Creates a new error with the provided code and message.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            location: None,
        }
    }
}

/// Describes the location within a source file where an error happened.
/// Describes the location within a source file where an error happened.
#[derive(Debug, Clone)]
pub struct SourceLocation {
    /// File path.
    pub file: String,
    /// 1-based line number.
    pub line: usize,
    /// 1-based column number.
    pub column: usize,
}
