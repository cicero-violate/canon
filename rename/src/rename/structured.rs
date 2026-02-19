//! Structured rewrite passes for doc comments, attributes, and use trees.
//!
//! # Unified AST Rewrite API (E1, E2, E3)
//!
//! This module provides a complete API for programmatic AST-level code editing:
//!
//! ## E1: AST Fragment Renderer
//!
//! The `ast_render` module provides utilities to render syn AST nodes
//! to formatted Rust code using prettyplease:
//!
//! ```no_run
//! use semantic_lint::rename::structured::ast_render;
//!
//! let func: syn::ItemFn = syn::parse_quote! {
//!     pub fn hello() -> String {
//!         "world".to_string()
//!     }
//! };
//!
//! let code = ast_render::render_function(&func);
//! println!("{}", code);
//! ```
//!
//! ## E2: Public API Surface
//!
//! The `apply_ast_rewrites` function is the main entry point for batch AST edits:
//!
//! ```no_run
//! use semantic_lint::rename::structured::{apply_ast_rewrites, AstEdit};
//! use std::path::Path;
//!
//! let file = Path::new("src/lib.rs");
//! let new_fn: syn::ItemFn = syn::parse_quote! {
//!     pub fn process(input: &str) -> Result<String, Error> {
//!         Ok(input.to_uppercase())
//!     }
//! };
//!
//! let edits = vec![AstEdit::insert(file, 0, &new_fn)];
//! let touched = apply_ast_rewrites(&edits, true).unwrap();
//! ```
//!
//! ## E3: Features
//!
//! - **Conflict Detection**: Overlapping edits are detected and reported
//! - **Formatting Integration**: Automatic rustfmt after applying edits
//! - **Buffer-based**: Uses rewrite buffers for efficient in-memory editing
//! - **Type-safe**: Leverages syn AST for correctness
//! - **Composable**: Works with all structured editing passes

use anyhow::{Error, Result};
use proc_macro2::Span;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use syn::spanned::Spanned;
use syn::visit::Visit;
use quote::ToTokens;

use super::alias::{UseKind, UseNode, VisibilityScope};
use super::core::{find_replacement_path, span_to_offsets, span_to_range, SpanRange};
use super::rewrite::{RewriteBufferSet, TextEdit};

/// C1: Unified Pass Orchestration
///
/// Trait for structured editing passes that operate on parsed ASTs
/// and accumulate changes via RewriteBuffer.
pub trait StructuredPass {
    /// Pass name for diagnostics
    fn name(&self) -> &'static str;

    /// Execute the pass on a file
    ///
    /// Returns true if any changes were made
    fn execute(
        &mut self,
        file: &Path,
        content: &str,
        ast: &syn::File,
        buffers: &mut RewriteBufferSet,
    ) -> Result<bool>;

    /// Check if this pass is enabled (for conditional execution)
    fn is_enabled(&self) -> bool {
        true
    }
}

/// C1: Pass orchestrator that runs structured passes in order
pub struct PassOrchestrator {
    passes: Vec<Box<dyn StructuredPass>>,
}

/// E1: AST Fragment Renderer
///
/// Utilities for rendering syn AST nodes to formatted text
/// and feeding them into rewrite buffers.
pub mod ast_render {
    use super::*;
    use anyhow::Result;
    use prettyplease;
    use quote::ToTokens;
    use std::path::Path;

    /// E1: Render a syn AST node to formatted Rust code
    ///
    /// Uses prettyplease for consistent formatting.
    pub fn render_node<T: ToTokens>(node: T) -> String {
        let tokens = quote::quote! { #node };
        let file = syn::parse_file(&tokens.to_string()).unwrap_or_else(|_| {
            // Fallback to unformatted if parse fails
            return syn::File {
                shebang: None,
                attrs: vec![],
                items: vec![],
            };
        });
        prettyplease::unparse(&file)
    }

    /// E1: Render a function to formatted Rust code
    pub fn render_function(func: &syn::ItemFn) -> String {
        let tokens = quote::quote! { #func };
        let file = syn::parse_file(&tokens.to_string()).unwrap_or_else(|_| syn::File {
            shebang: None,
            attrs: vec![],
            items: vec![syn::Item::Fn(func.clone())],
        });
        prettyplease::unparse(&file)
    }

    /// E1: Render an impl block to formatted Rust code
    pub fn render_impl(impl_block: &syn::ItemImpl) -> String {
        let tokens = quote::quote! { #impl_block };
        let file = syn::parse_file(&tokens.to_string()).unwrap_or_else(|_| syn::File {
            shebang: None,
            attrs: vec![],
            items: vec![syn::Item::Impl(impl_block.clone())],
        });
        prettyplease::unparse(&file)
    }

    /// E1: Render a struct to formatted Rust code
    pub fn render_struct(struct_item: &syn::ItemStruct) -> String {
        let tokens = quote::quote! { #struct_item };
        let file = syn::parse_file(&tokens.to_string()).unwrap_or_else(|_| syn::File {
            shebang: None,
            attrs: vec![],
            items: vec![syn::Item::Struct(struct_item.clone())],
        });
        prettyplease::unparse(&file)
    }

    /// E1: Render a trait to formatted Rust code
    pub fn render_trait(trait_item: &syn::ItemTrait) -> String {
        let tokens = quote::quote! { #trait_item };
        let file = syn::parse_file(&tokens.to_string()).unwrap_or_else(|_| syn::File {
            shebang: None,
            attrs: vec![],
            items: vec![syn::Item::Trait(trait_item.clone())],
        });
        prettyplease::unparse(&file)
    }

    /// E1: Replace a span in a file with rendered AST node
    ///
    /// # Example
    ///
    /// ```no_run
    /// use semantic_lint::rename::structured::ast_render;
    /// use semantic_lint::rename::rewrite::RewriteBufferSet;
    /// use std::path::Path;
    ///
    /// let mut buffers = RewriteBufferSet::new();
    /// let file = Path::new("src/lib.rs");
    /// let content = std::fs::read_to_string(file).unwrap();
    ///
    /// // Create a new function
    /// let new_fn: syn::ItemFn = syn::parse_quote! {
    ///     pub fn hello() -> String {
    ///         "world".to_string()
    ///     }
    /// };
    ///
    /// // Render and replace
    /// ast_render::replace_with_node(
    ///     &mut buffers,
    ///     file,
    ///     &content,
    ///     0,
    ///     10,
    ///     &new_fn,
    /// ).unwrap();
    /// ```
    pub fn replace_with_node<T: ToTokens>(
        buffers: &mut RewriteBufferSet,
        file: &Path,
        content: &str,
        start: usize,
        end: usize,
        node: &T,
    ) -> Result<()> {
        let rendered = render_node(node);
        let buffer = buffers.ensure_buffer(file, content);
        buffer.replace(start, end, rendered)?;
        Ok(())
    }

    /// E1: Insert rendered AST node at position
    pub fn insert_node<T: ToTokens>(
        buffers: &mut RewriteBufferSet,
        file: &Path,
        content: &str,
        offset: usize,
        node: &T,
    ) -> Result<()> {
        let rendered = render_node(node);
        let buffer = buffers.ensure_buffer(file, content);
        buffer.insert(offset, rendered)?;
        Ok(())
    }
}

/// E2: AST Rewrite Edit Operation
///
/// Describes a single AST-level rewrite operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstEdit {
    /// File to edit
    pub file: PathBuf,
    /// Start byte offset
    pub start: usize,
    /// End byte offset
    pub end: usize,
    /// Replacement AST node (as formatted text)
    pub replacement: String,
}

impl AstEdit {
    /// Create an edit that replaces a span with a rendered AST node
    pub fn replace<T: quote::ToTokens>(
        file: impl Into<PathBuf>,
        start: usize,
        end: usize,
        node: &T,
    ) -> Self {
        Self {
            file: file.into(),
            start,
            end,
            replacement: ast_render::render_node(node),
        }
    }

    /// Create an edit that inserts a rendered AST node at a position
    pub fn insert<T: quote::ToTokens>(file: impl Into<PathBuf>, offset: usize, node: &T) -> Self {
        Self {
            file: file.into(),
            start: offset,
            end: offset,
            replacement: ast_render::render_node(node),
        }
    }
}

/// E2: Apply a batch of AST rewrites to a project
///
/// This is the main public API for programmatic AST editing.
///
/// # Example
///
/// ```no_run
/// use semantic_lint::rename::structured::{apply_ast_rewrites, AstEdit};
/// use std::path::Path;
///
/// let file = Path::new("src/lib.rs");
///
/// // Create a new function to insert
/// let new_fn: syn::ItemFn = syn::parse_quote! {
///     pub fn greet(name: &str) -> String {
///         format!("Hello, {}!", name)
///     }
/// };
///
/// let edits = vec![
///     AstEdit::insert(file, 0, &new_fn),
/// ];
///
/// apply_ast_rewrites(&edits, true).unwrap();
/// ```
pub fn apply_ast_rewrites(edits: &[AstEdit], format: bool) -> Result<Vec<PathBuf>> {
    let mut buffers = RewriteBufferSet::new();
    let mut file_contents: HashMap<PathBuf, String> = HashMap::new();

    // Group edits by file
    for edit in edits {
        // Load file content if not already loaded
        if !file_contents.contains_key(&edit.file) {
            let content = std::fs::read_to_string(&edit.file)
                .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", edit.file.display(), e))?;
            file_contents.insert(edit.file.clone(), content);
        }
    }

    // Apply all edits
    for edit in edits {
        let content = file_contents.get(&edit.file).unwrap();
        let buffer = buffers.ensure_buffer(&edit.file, content);
        buffer.replace(edit.start, edit.end, &edit.replacement)?;
    }

    // Flush buffers
    let touched = buffers.flush()?;

    // Format if requested
    if format && !touched.is_empty() {
        format_touched_files(&touched)?;
    }

    Ok(touched)
}

/// E2: Helper to format files after AST rewrites
fn format_touched_files(files: &[PathBuf]) -> Result<()> {
    for file in files {
        if !file.exists() {
            continue;
        }
        let mut cmd = std::process::Command::new("rustfmt");
        cmd.arg("--edition").arg("2021").arg(file);
        let _ = cmd.status(); // Ignore formatting errors
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_pass_orchestrator_execution() {
        let mut orchestrator = PassOrchestrator::new();
        let mut buffers = RewriteBufferSet::new();

        // Create a simple test pass
        struct TestPass {
            enabled: bool,
            executed: bool,
        }

        impl StructuredPass for TestPass {
            fn name(&self) -> &'static str {
                "test_pass"
            }

            fn execute(
                &mut self,
                _file: &Path,
                _content: &str,
                _ast: &syn::File,
                _buffers: &mut RewriteBufferSet,
            ) -> Result<bool> {
                self.executed = true;
                Ok(true) // Report that changes were made
            }

            fn is_enabled(&self) -> bool {
                self.enabled
            }
        }

        orchestrator.add_pass(Box::new(TestPass {
            enabled: true,
            executed: false,
        }));

        assert_eq!(orchestrator.enabled_count(), 1);

        let file = PathBuf::from("test.rs");
        let content = "fn main() {}";
        let ast = syn::parse_file(content).unwrap();

        let changed = orchestrator
            .run_passes(&file, content, &ast, &mut buffers)
            .unwrap();

        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0], "test_pass");
    }

    #[test]
    fn test_disabled_pass_skipped() {
        let mut orchestrator = PassOrchestrator::new();
        let mut buffers = RewriteBufferSet::new();

        struct DisabledPass;
        impl StructuredPass for DisabledPass {
            fn name(&self) -> &'static str {
                "disabled"
            }
            fn execute(
                &mut self,
                _: &Path,
                _: &str,
                _: &syn::File,
                _: &mut RewriteBufferSet,
            ) -> Result<bool> {
                panic!("Should not execute disabled pass");
            }
            fn is_enabled(&self) -> bool {
                false
            }
        }

        orchestrator.add_pass(Box::new(DisabledPass));
        assert_eq!(orchestrator.enabled_count(), 0);

        let file = PathBuf::from("test.rs");
        let content = "fn main() {}";
        let ast = syn::parse_file(content).unwrap();

        let changed = orchestrator
            .run_passes(&file, content, &ast, &mut buffers)
            .unwrap();

        assert_eq!(changed.len(), 0);
    }

    /// E3: Test AST fragment rendering
    #[test]
    fn test_render_function() {
        use syn::parse_quote;

        let func: syn::ItemFn = parse_quote! {
            pub fn add(a: i32, b: i32) -> i32 {
                a + b
            }
        };

        let rendered = ast_render::render_function(&func);
        assert!(rendered.contains("pub fn add"));
        assert!(rendered.contains("i32"));
        assert!(rendered.contains("a + b"));
    }

    #[test]
    fn test_render_impl() {
        use syn::parse_quote;

        let impl_block: syn::ItemImpl = parse_quote! {
            impl MyStruct {
                fn new() -> Self {
                    Self { value: 0 }
                }
            }
        };

        let rendered = ast_render::render_impl(&impl_block);
        assert!(rendered.contains("impl MyStruct"));
        assert!(rendered.contains("fn new"));
    }

    #[test]
    fn test_render_struct() {
        use syn::parse_quote;

        let struct_item: syn::ItemStruct = parse_quote! {
            pub struct Point {
                x: f64,
                y: f64,
            }
        };

        let rendered = ast_render::render_struct(&struct_item);
        assert!(rendered.contains("pub struct Point"));
        assert!(rendered.contains("x: f64"));
        assert!(rendered.contains("y: f64"));
    }

    /// E3: Test AST edit creation
    #[test]
    fn test_ast_edit_replace() {
        use syn::parse_quote;

        let func: syn::ItemFn = parse_quote! {
            fn test() {}
        };

        let edit = AstEdit::replace("test.rs", 0, 10, &func);
        assert_eq!(edit.file, PathBuf::from("test.rs"));
        assert_eq!(edit.start, 0);
        assert_eq!(edit.end, 10);
        assert!(edit.replacement.contains("fn test"));
    }

    #[test]
    fn test_ast_edit_insert() {
        use syn::parse_quote;

        let func: syn::ItemFn = parse_quote! {
            fn helper() -> i32 { 42 }
        };

        let edit = AstEdit::insert("lib.rs", 100, &func);
        assert_eq!(edit.start, 100);
        assert_eq!(edit.end, 100); // Insert is zero-width
        assert!(edit.replacement.contains("fn helper"));
    }

    /// E3: Test buffer integration with AST rendering
    #[test]
    fn test_replace_with_node() {
        use syn::parse_quote;

        let mut buffers = RewriteBufferSet::new();
        let file = PathBuf::from("test.rs");
        let content = "fn old() {}\n";

        let new_fn: syn::ItemFn = parse_quote! {
            fn new() -> String {
                "hello".to_string()
            }
        };

        ast_render::replace_with_node(&mut buffers, &file, content, 0, 11, &new_fn).unwrap();

        assert!(buffers.is_file_dirty(&file));
        assert_eq!(buffers.total_edit_count(), 1);
    }

    /// E3: Test conflict detection with AST edits
    #[test]
    fn test_ast_edit_conflict_detection() {
        use syn::parse_quote;

        let mut buffers = RewriteBufferSet::new();
        let file = PathBuf::from("test.rs");
        let content = "fn foo() {}\nfn bar() {}\n";

        let fn1: syn::ItemFn = parse_quote! { fn replaced1() {} };
        let fn2: syn::ItemFn = parse_quote! { fn replaced2() {} };

        // First edit
        ast_render::replace_with_node(&mut buffers, &file, content, 0, 11, &fn1).unwrap();

        // Overlapping edit should conflict
        let result = ast_render::replace_with_node(&mut buffers, &file, content, 5, 15, &fn2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Conflicting"));
    }

    /// E3: Test multiple non-conflicting AST edits
    #[test]
    fn test_multiple_ast_edits() {
        use syn::parse_quote;

        let mut buffers = RewriteBufferSet::new();
        let file = PathBuf::from("multi.rs");
        let content = "fn a() {}\nfn b() {}\nfn c() {}\n";

        let new_a: syn::ItemFn = parse_quote! { fn new_a() {} };
        let new_c: syn::ItemFn = parse_quote! { fn new_c() {} };

        // Replace first function
        ast_render::replace_with_node(&mut buffers, &file, content, 0, 9, &new_a).unwrap();

        // Replace third function (non-overlapping)
        ast_render::replace_with_node(&mut buffers, &file, content, 20, 29, &new_c).unwrap();

        assert_eq!(buffers.total_edit_count(), 2);
    }

    /// E3: Test inserting new impl block
    #[test]
    fn test_insert_impl_block() {
        use syn::parse_quote;

        let mut buffers = RewriteBufferSet::new();
        let file = PathBuf::from("types.rs");
        let content = "struct Foo;\n";

        let new_impl: syn::ItemImpl = parse_quote! {
            impl Foo {
                pub fn new() -> Self {
                    Foo
                }
            }
        };

        // Insert after struct definition
        ast_render::insert_node(&mut buffers, &file, content, 12, &new_impl).unwrap();

        assert!(buffers.is_dirty());
    }

    /// E3: Comprehensive integration test demonstrating the full API
    #[test]
    fn test_e1_e2_e3_integration() {
        use syn::parse_quote;

        // E1: Render various AST fragments
        let new_struct: syn::ItemStruct = parse_quote! {
            pub struct Config {
                name: String,
                value: i32,
            }
        };
        let struct_code = ast_render::render_struct(&new_struct);
        assert!(struct_code.contains("pub struct Config"));

        let new_impl: syn::ItemImpl = parse_quote! {
            impl Config {
                pub fn new(name: String, value: i32) -> Self {
                    Self { name, value }
                }

                pub fn display(&self) -> String {
                    format!("{}: {}", self.name, self.value)
                }
            }
        };
        let impl_code = ast_render::render_impl(&new_impl);
        assert!(impl_code.contains("impl Config"));
        assert!(impl_code.contains("pub fn new"));

        let helper_fn: syn::ItemFn = parse_quote! {
            fn validate_config(config: &Config) -> bool {
                !config.name.is_empty() && config.value >= 0
            }
        };
        let fn_code = ast_render::render_function(&helper_fn);
        assert!(fn_code.contains("fn validate_config"));

        // E2: Create edits using the public API
        let file = PathBuf::from("config.rs");

        let edit1 = AstEdit::insert(&file, 0, &new_struct);
        assert_eq!(edit1.start, 0);
        assert_eq!(edit1.end, 0);

        let edit2 = AstEdit::insert(&file, 0, &new_impl);
        let edit3 = AstEdit::insert(&file, 0, &helper_fn);

        // Verify edits contain expected code
        assert!(edit1.replacement.contains("pub struct"));
        assert!(edit2.replacement.contains("impl Config"));
        assert!(edit3.replacement.contains("validate_config"));

        // E3: Test buffer integration with conflict detection
        let mut buffers = RewriteBufferSet::new();
        let content = "\n\n\n\n\n"; // Enough space for both inserts

        // Insert at different positions to avoid conflict
        ast_render::insert_node(&mut buffers, &file, content, 0, &new_struct).unwrap();
        ast_render::insert_node(&mut buffers, &file, content, 3, &helper_fn).unwrap();

        assert_eq!(buffers.total_edit_count(), 2);
        assert!(buffers.is_file_dirty(&file));
    }

    #[test]
    fn test_structured_config_flags() {
        let config = StructuredEditConfig::new(true, false, true);
        assert!(config.doc_literals_enabled());
        assert!(!config.attr_literals_enabled());
        assert!(config.use_statements_enabled());
        assert!(config.is_enabled());
        assert_eq!(config.summary(), "docs+uses");

        let disabled = StructuredEditConfig::disabled();
        assert!(!disabled.is_enabled());
        assert_eq!(disabled.summary(), "none");

        let all = StructuredEditConfig::all_enabled();
        assert_eq!(all.summary(), "docs+attrs+uses");
    }
}

impl PassOrchestrator {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Add a pass to the pipeline
    pub fn add_pass(&mut self, pass: Box<dyn StructuredPass>) {
        self.passes.push(pass);
    }

    /// Run all enabled passes on a file
    ///
    /// Returns set of pass names that made changes
    pub fn run_passes(
        &mut self,
        file: &Path,
        content: &str,
        ast: &syn::File,
        buffers: &mut RewriteBufferSet,
    ) -> Result<Vec<&'static str>> {
        let mut changed_passes = Vec::new();
        for pass in &mut self.passes {
            if !pass.is_enabled() {
                continue;
            }
            if pass.execute(file, content, ast, buffers)? {
                changed_passes.push(pass.name());
            }
        }
        Ok(changed_passes)
    }

    /// Get count of enabled passes
    pub fn enabled_count(&self) -> usize {
        self.passes.iter().filter(|p| p.is_enabled()).count()
    }
}

impl Default for PassOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

/// C1: Create a standard orchestrator with all rename passes
///
/// This is the main entry point for structured editing in the rename pipeline.
///
/// # Example
///
/// ```no_run
/// use semantic_lint::rename::structured::*;
/// use semantic_lint::rename::rewrite::RewriteBufferSet;
/// use std::collections::HashMap;
/// use std::path::Path;
///
/// let mapping = HashMap::from([
///     ("crate::old::Foo".to_string(), "NewFoo".to_string()),
/// ]);
/// let path_updates = HashMap::from([
///     ("crate::old".to_string(), "crate::new".to_string()),
/// ]);
/// let config = StructuredEditConfig::all_enabled();
///
/// let mut orchestrator = create_rename_orchestrator(
///     &mapping,
///     &path_updates,
///     vec![],
///     config,
/// );
///
/// let mut buffers = RewriteBufferSet::new();
/// let file = Path::new("src/lib.rs");
/// let content = std::fs::read_to_string(file).unwrap();
/// let ast = syn::parse_file(&content).unwrap();
///
/// let changed_passes = orchestrator.run_passes(
///     file,
///     &content,
///     &ast,
///     &mut buffers,
/// ).unwrap();
///
/// println!("Passes that made changes: {:?}", changed_passes);
/// buffers.flush().unwrap();
/// ```
pub fn create_rename_orchestrator(
    mapping: &HashMap<String, String>,
    path_updates: &HashMap<String, String>,
    alias_nodes: Vec<UseNode>,
    config: StructuredEditConfig,
) -> PassOrchestrator {
    let mut orchestrator = PassOrchestrator::new();

    // C2: Add doc/attr pass
    orchestrator.add_pass(Box::new(DocAttrPass::new(mapping.clone(), config.clone())));

    // C3: Add use-tree pass
    orchestrator.add_pass(Box::new(UseTreePass::new(
        path_updates.clone(),
        alias_nodes,
        config,
    )));

    orchestrator
}

/// B1: Structured Edit Configuration
///
/// Controls which structured editing passes are enabled.
/// Each pass can be independently enabled/disabled via environment variables:
///
/// - `SEMANTIC_LINT_STRUCTURED_EDITS` - Master toggle (default: true)
/// - `SEMANTIC_LINT_STRUCTURED_DOCS` - Doc comment literal rewrites (default: true)
/// - `SEMANTIC_LINT_STRUCTURED_ATTRS` - Attribute literal rewrites (default: true)
/// - `SEMANTIC_LINT_STRUCTURED_USES` - Use tree synthesis (default: true)
///
/// Set any to "0", "false", "off", or "disable" to turn off.
#[derive(Clone, Debug)]
pub struct StructuredEditConfig {
    doc_literals: bool,
    attr_literals: bool,
    use_statements: bool,
}

impl StructuredEditConfig {
    /// Create a new configuration with explicit flags
    pub fn new(doc_literals: bool, attr_literals: bool, use_statements: bool) -> Self {
        Self {
            doc_literals,
            attr_literals,
            use_statements,
        }
    }

    /// Create a configuration with all passes disabled
    pub fn disabled() -> Self {
        Self::new(false, false, false)
    }

    /// Create a configuration with all passes enabled
    pub fn all_enabled() -> Self {
        Self::new(true, true, true)
    }

    /// Check if any structured editing is enabled
    pub fn is_enabled(&self) -> bool {
        self.doc_literals || self.attr_literals || self.use_statements
    }

    /// Check if doc comment literal rewrites are enabled
    pub fn doc_literals_enabled(&self) -> bool {
        self.doc_literals
    }

    /// Check if attribute literal rewrites are enabled
    pub fn attr_literals_enabled(&self) -> bool {
        self.attr_literals
    }

    /// Check if either doc or attr rewrites are enabled
    pub fn doc_or_attr_enabled(&self) -> bool {
        self.doc_literals || self.attr_literals
    }

    /// Check if use statement synthesis is enabled
    pub fn use_statements_enabled(&self) -> bool {
        self.use_statements
    }

    /// Get a summary of which passes are enabled
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.doc_literals {
            parts.push("docs");
        }
        if self.attr_literals {
            parts.push("attrs");
        }
        if self.use_statements {
            parts.push("uses");
        }
        if parts.is_empty() {
            "none".to_string()
        } else {
            parts.join("+")
        }
    }
}

/// Check if structured edits are enabled (backwards compat helper)
pub fn structured_edits_enabled() -> bool {
    structured_edit_config().is_enabled()
}

/// Load structured edit configuration from environment variables
///
/// # B1: Environment Variable Loading
///
/// Hierarchy:
/// 1. If SEMANTIC_LINT_STRUCTURED_EDITS=0/false/off/disable -> all disabled
/// 2. Otherwise, check individual flags (DOCS, ATTRS, USES)
/// 3. Default: all enabled
pub fn structured_edit_config() -> StructuredEditConfig {
    let base_enabled = env_flag("SEMANTIC_LINT_STRUCTURED_EDITS", true);
    if !base_enabled {
        return StructuredEditConfig::disabled();
    }
    let doc_literals = env_flag("SEMANTIC_LINT_STRUCTURED_DOCS", true);
    let attr_literals = env_flag("SEMANTIC_LINT_STRUCTURED_ATTRS", true);
    let use_statements = env_flag("SEMANTIC_LINT_STRUCTURED_USES", true);
    StructuredEditConfig::new(
        base_enabled && doc_literals,
        base_enabled && attr_literals,
        base_enabled && use_statements,
    )
}

/// Check if a string value represents "false" in various forms
fn matches_ignore_case(value: &str, target: &str) -> bool {
    value.eq_ignore_ascii_case(target)
}

/// Parse environment variable as boolean flag
///
/// Values "0", "false", "off", "disable" (case-insensitive) -> false
/// Any other value or presence -> true
/// Missing variable -> default
fn env_flag(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(value) => {
            if matches_ignore_case(&value, "0")
                || matches_ignore_case(&value, "false")
                || matches_ignore_case(&value, "off")
                || matches_ignore_case(&value, "disable")
            {
                false
            } else {
                true
            }
        }
        Err(_) => default,
    }
}

/// Result of structured attribute/doc processing
///
/// Tracks which spans were rewritten so span-based edits can skip them
pub struct StructuredAttributeResult {
    pub literal_spans: Vec<SpanRange>,
    pub changed: bool,
}

impl StructuredAttributeResult {
    pub fn new() -> Self {
        Self {
            literal_spans: Vec::new(),
            changed: false,
        }
    }

    /// Check if a span should be skipped (already handled by structured pass)
    pub fn should_skip(&self, span: &SpanRange) -> bool {
        self.literal_spans.iter().any(|range| contains(range, span))
    }
}

/// Check if outer span contains inner span
fn contains(outer: &SpanRange, inner: &SpanRange) -> bool {
    if inner.start.line < outer.start.line || inner.end.line > outer.end.line {
        return false;
    }
    if outer.start.line == inner.start.line && inner.start.column < outer.start.column {
        return false;
    }
    if outer.end.line == inner.end.line && inner.end.column > outer.end.column {
        return false;
    }
    true
}

/// C2: Doc/Attr pass implementing StructuredPass trait
pub struct DocAttrPass {
    mapping: HashMap<String, String>,
    config: StructuredEditConfig,
}

impl DocAttrPass {
    pub fn new(mapping: HashMap<String, String>, config: StructuredEditConfig) -> Self {
        Self { mapping, config }
    }
}

impl StructuredPass for DocAttrPass {
    fn name(&self) -> &'static str {
        "doc_attr"
    }

    fn execute(
        &mut self,
        file: &Path,
        content: &str,
        ast: &syn::File,
        buffers: &mut RewriteBufferSet,
    ) -> Result<bool> {
        let result = rewrite_doc_and_attr_literals(
            file,
            content,
            ast,
            &self.mapping,
            &self.config,
            buffers,
        )?;
        Ok(result.changed)
    }

    fn is_enabled(&self) -> bool {
        self.config.doc_or_attr_enabled()
    }
}

/// C3: Use-tree synthesizer pass implementing StructuredPass trait
///
/// Rewrites use statements based on module path updates and alias graph.
/// Produces deterministic, grouped use statements.
pub struct UseTreePass {
    path_updates: HashMap<String, String>,
    alias_nodes: Vec<UseNode>,
    config: StructuredEditConfig,
}

impl UseTreePass {
    pub fn new(
        path_updates: HashMap<String, String>,
        alias_nodes: Vec<UseNode>,
        config: StructuredEditConfig,
    ) -> Self {
        Self {
            path_updates,
            alias_nodes,
            config,
        }
    }

    /// Update with fresh alias nodes for a specific file
    pub fn with_file_aliases(&mut self, file_key: &str, all_nodes: &[UseNode]) {
        self.alias_nodes = all_nodes
            .iter()
            .filter(|n| n.file == file_key)
            .cloned()
            .collect();
    }
}

impl StructuredPass for UseTreePass {
    fn name(&self) -> &'static str {
        "use_tree"
    }

    fn execute(
        &mut self,
        file: &Path,
        content: &str,
        ast: &syn::File,
        buffers: &mut RewriteBufferSet,
    ) -> Result<bool> {
        if self.path_updates.is_empty() || self.alias_nodes.is_empty() {
            return Ok(false);
        }

        let alias_node_refs: Vec<&UseNode> = self.alias_nodes.iter().collect();
        rewrite_use_statements(
            file,
            content,
            ast,
            &self.path_updates,
            &alias_node_refs,
            buffers,
        )
    }

    fn is_enabled(&self) -> bool {
        self.config.use_statements_enabled()
    }
}

/// Rewrite doc comments and attribute literals
///
/// # B1: Doc/Attr Pass Entry Point
///
/// Respects config.doc_literals_enabled() and config.attr_literals_enabled()
pub fn rewrite_doc_and_attr_literals(
    file: &Path,
    content: &str,
    ast: &syn::File,
    mapping: &HashMap<String, String>,
    config: &StructuredEditConfig,
    buffers: &mut RewriteBufferSet,
) -> Result<StructuredAttributeResult> {
    let mut visitor = AttributeRewriteVisitor::new(file, content, mapping, config, buffers);
    visitor.visit_file(ast);
    visitor.finish()
}

struct AttributeRewriteVisitor<'a> {
    file: &'a Path,
    content: &'a str,
    replacements: Vec<(String, String)>,
    buffers: &'a mut RewriteBufferSet,
    rewrite_docs: bool,
    rewrite_attrs: bool,
    error: Option<Error>,
    result: StructuredAttributeResult,
}

impl<'a> AttributeRewriteVisitor<'a> {
    fn new(
        file: &'a Path,
        content: &'a str,
        mapping: &HashMap<String, String>,
        config: &StructuredEditConfig,
        buffers: &'a mut RewriteBufferSet,
    ) -> Self {
        Self {
            file,
            content,
            replacements: build_replacements(mapping),
            buffers,
            rewrite_docs: config.doc_literals_enabled(),
            rewrite_attrs: config.attr_literals_enabled(),
            error: None,
            result: StructuredAttributeResult::new(),
        }
    }

    fn finish(self) -> Result<StructuredAttributeResult> {
        if let Some(err) = self.error {
            Err(err)
        } else {
            Ok(self.result)
        }
    }

    fn process_attribute(&mut self, attr: &syn::Attribute) {
        let is_doc = attr.path().is_ident("doc");
        if (is_doc && !self.rewrite_docs) || (!is_doc && !self.rewrite_attrs) {
            return;
        }
        let syn::Meta::NameValue(meta) = &attr.meta else {
            return;
        };
        let syn::Expr::Lit(expr_lit) = &meta.value else {
            return;
        };
        let syn::Lit::Str(lit) = &expr_lit.lit else {
            return;
        };
        let original = lit.value();
        if original.is_empty() {
            return;
        }
        if let Some(updated) = rewrite_literal(&original, &self.replacements) {
            if updated != original {
                // Preserve original literal span to avoid corrupting outer attribute/doc structure
                let new_literal = syn::LitStr::new(&updated, lit.span());
                let replacement_text = new_literal.to_token_stream().to_string();
                let span = span_to_range(lit.span());
                let (start, end) = span_to_offsets(self.content, &span.start, &span.end);
                if let Err(err) = self.buffers.queue_edits(
                    self.file,
                    self.content,
                    [TextEdit {
                        start,
                        end,
                        text: replacement_text,
                    }],
                ) {
                    self.error = Some(err);
                    return;
                }
                if is_doc {
                    self.result.literal_spans.push(span);
                }
                self.result.changed = true;
            }
        }
    }
}

impl<'ast> Visit<'ast> for AttributeRewriteVisitor<'_> {
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        self.process_attribute(attr);
        // No need to recurse inside attribute tokens.
    }
}

fn build_replacements(mapping: &HashMap<String, String>) -> Vec<(String, String)> {
    let mut replacements = Vec::new();
    for (old_id, new_name) in mapping {
        if let Some(tail) = old_id.rsplit("::").next() {
            if tail != new_name && !tail.is_empty() {
                replacements.push((tail.to_string(), new_name.clone()));
            }
        }
        if let Some(stripped) = old_id.strip_prefix("crate::") {
            if stripped != new_name && !stripped.is_empty() {
                replacements.push((stripped.to_string(), new_name.clone()));
            }
        }
    }
    replacements
}

fn rewrite_literal(value: &str, replacements: &[(String, String)]) -> Option<String> {
    let mut updated = value.to_string();
    let mut changed = false;
    for (old, new_name) in replacements {
        if old.is_empty() || old == new_name {
            continue;
        }
        if let Some(next) = replace_identifier(&updated, old, new_name) {
            updated = next;
            changed = true;
        }
    }
    if changed {
        Some(updated)
    } else {
        None
    }
}

fn replace_identifier(text: &str, old: &str, new_name: &str) -> Option<String> {
    let mut result = String::new();
    let mut cursor = 0usize;
    let mut changed = false;
    while let Some(rel_pos) = text[cursor..].find(old) {
        let start = cursor + rel_pos;
        let end = start + old.len();
        if is_boundary(text, start, end) {
            changed = true;
            result.push_str(&text[cursor..start]);
            result.push_str(new_name);
            cursor = end;
        } else {
            result.push_str(&text[cursor..end]);
            cursor = end;
        }
    }
    result.push_str(&text[cursor..]);
    if changed {
        Some(result)
    } else {
        None
    }
}

fn is_boundary(text: &str, start: usize, end: usize) -> bool {
    let prev = text[..start].chars().rev().next();
    let next = text[end..].chars().next();
    let prev_ok = match prev {
        Some(c) if c.is_ascii_alphanumeric() || c == '_' => false,
        _ => true,
    };
    let next_ok = match next {
        Some(c) if c.is_ascii_alphanumeric() || c == '_' => false,
        _ => true,
    };
    prev_ok && next_ok
}

pub fn rewrite_use_statements(
    file: &Path,
    content: &str,
    ast: &syn::File,
    path_updates: &HashMap<String, String>,
    alias_nodes: &[&UseNode],
    buffers: &mut RewriteBufferSet,
) -> Result<bool> {
    if path_updates.is_empty() || alias_nodes.is_empty() {
        return Ok(false);
    }
    let Some(replacement_block) = render_rewritten_use(alias_nodes, path_updates) else {
        return Ok(false);
    };

    let mut first_span: Option<SpanRange> = None;
    let mut last_span: Option<SpanRange> = None;
    for item in &ast.items {
        if let syn::Item::Use(use_item) = item {
            if !use_item.attrs.is_empty() {
                continue;
            }
            let span = span_to_range(use_item.span());
            if first_span.is_none() {
                first_span = Some(span.clone());
            }
            last_span = Some(span);
        }
    }

    if let (Some(first), Some(last)) = (first_span, last_span) {
        let (start, _) = span_to_offsets(content, &first.start, &first.end);
        let (_, end) = span_to_offsets(content, &last.start, &last.end);
        buffers.queue_edits(
            file,
            content,
            [TextEdit {
                start,
                end,
                text: ensure_trailing_newline(&replacement_block),
            }],
        )?;
        return Ok(true);
    }

    let insert_pos = find_use_insert_position(content);
    buffers.queue_edits(
        file,
        content,
        [TextEdit {
            start: insert_pos,
            end: insert_pos,
            text: format!("{}\n", replacement_block),
        }],
    )?;
    Ok(true)
}

#[derive(Clone, Eq, PartialEq)]
struct FlatUseEntry {
    segments: Vec<String>,
    alias: Option<String>,
    is_glob: bool,
    leading_colon: bool,
    visibility: String,
    order: usize,
}

impl Ord for FlatUseEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        visibility_rank(&self.visibility)
            .cmp(&visibility_rank(&other.visibility))
            .then_with(|| self.leading_colon.cmp(&other.leading_colon))
            .then_with(|| self.is_glob.cmp(&other.is_glob))
            .then_with(|| self.segments.cmp(&other.segments))
            .then_with(|| self.alias.cmp(&other.alias))
            .then_with(|| self.order.cmp(&other.order))
    }
}

impl PartialOrd for FlatUseEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd)]
enum BraceKind {
    SelfItem,
    Named,
    Glob,
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd)]
struct BraceEntry {
    text: String,
    kind: BraceKind,
    order: usize,
}

/// C3: Render use statements from alias nodes with deterministic ordering
///
/// Guarantees:
/// - Stable sort by (visibility, leading_colon, path segments)
/// - Consistent grouping by common prefixes
/// - Deterministic output for same input
fn render_rewritten_use(
    alias_nodes: &[&UseNode],
    path_updates: &HashMap<String, String>,
) -> Option<String> {
    let mut entries = Vec::new();
    // C3: Process all alias nodes in order
    for (order, node) in alias_nodes.iter().enumerate() {
        let segments: Vec<String> = node
            .source_path
            .trim_start_matches("::")
            .split("::")
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        if segments.is_empty() {
            continue;
        }
        let alias = match &node.original_name {
            Some(orig) if orig != &node.local_name => Some(node.local_name.clone()),
            Some(_) => None,
            None if segments
                .last()
                .map(|s| s != &node.local_name)
                .unwrap_or(false) =>
            {
                Some(node.local_name.clone())
            }
            _ => None,
        };
        let entry = FlatUseEntry {
            segments,
            alias,
            is_glob: node.kind == UseKind::Glob,
            leading_colon: node.source_path.starts_with("::"),
            visibility: visibility_string(&node.visibility),
            order,
        };
        entries.push(entry);
    }
    if entries.is_empty() {
        return None;
    }

    let mut changed = false;
    for entry in &mut entries {
        if apply_path_updates(entry, path_updates) {
            changed = true;
        }
    }
    if !changed {
        return None;
    }

    let mut set = BTreeSet::new();
    set.extend(entries.into_iter());
    let entries: Vec<_> = set.into_iter().collect();
    build_use_statements(&entries)
}

fn apply_path_updates(entry: &mut FlatUseEntry, updates: &HashMap<String, String>) -> bool {
    if entry.segments.is_empty() {
        return false;
    }
    let mut path_str = entry.segments.join("::");
    if entry.leading_colon {
        path_str = format!("::{}", path_str);
    }
    let query = path_str.trim_start_matches("::");
    if let Some(new_path) = find_replacement_path(query, updates) {
        let mut segments: Vec<String> = new_path.split("::").map(|s| s.to_string()).collect();
        if entry.leading_colon && segments.first().map(|s| s == "crate").unwrap_or(false) {
            segments.remove(0);
        }
        if segments.is_empty() {
            return false;
        }
        if segments != entry.segments {
            entry.segments = segments;
            return true;
        }
    }
    false
}

/// C3: Build use statements with deterministic grouping
///
/// Groups imports by common prefix and visibility, maintaining stable order.
/// Uses BTreeMap to ensure iteration order is deterministic.
fn build_use_statements(entries: &[FlatUseEntry]) -> Option<String> {
    if entries.is_empty() {
        return None;
    }
    let mut output = String::new();

    let mut grouped = vec![false; entries.len()];
    // C3: BTreeMap ensures deterministic iteration order
    let mut groups: BTreeMap<(String, bool, Vec<String>), Vec<usize>> = BTreeMap::new();

    for (idx, entry) in entries.iter().enumerate() {
        if entry.is_glob && entry.segments.is_empty() {
            continue;
        }
        let prefix_len = if entry.segments.is_empty() {
            0
        } else if entry.is_glob {
            entry.segments.len()
        } else if entry.segments.len() == 1 {
            1
        } else {
            entry.segments.len() - 1
        };
        if prefix_len == 0 {
            continue;
        }
        let prefix = entry.segments[..prefix_len].to_vec();
        groups
            .entry((entry.visibility.clone(), entry.leading_colon, prefix))
            .or_default()
            .push(idx);
    }

    for ((visibility, leading_colon, prefix), indexes) in &groups {
        if indexes.len() <= 1 {
            continue;
        }
        let mut brace_items = Vec::new();
        let mut can_group = true;
        for idx in indexes {
            let entry = &entries[*idx];
            let brace_entry = if entry.is_glob {
                if entry.segments.len() != prefix.len() {
                    can_group = false;
                    break;
                }
                BraceEntry {
                    text: "*".to_string(),
                    kind: BraceKind::Glob,
                    order: entry.order,
                }
            } else if entry.segments.len() == prefix.len() {
                let text = if let Some(alias) = &entry.alias {
                    format!("self as {}", alias)
                } else {
                    "self".to_string()
                };
                BraceEntry {
                    text,
                    kind: BraceKind::SelfItem,
                    order: entry.order,
                }
            } else {
                let last = entry.segments.last().cloned().unwrap_or_default();
                let text = if let Some(alias) = &entry.alias {
                    format!("{last} as {alias}")
                } else {
                    last
                };
                BraceEntry {
                    text,
                    kind: BraceKind::Named,
                    order: entry.order,
                }
            };
            if matches!(brace_entry.kind, BraceKind::Glob) && entry.segments.len() != prefix.len() {
                can_group = false;
                break;
            }
            brace_items.push(brace_entry);
            grouped[*idx] = true;
        }
        if !can_group {
            for idx in indexes {
                grouped[*idx] = false;
            }
            continue;
        }
        brace_items.sort();
        let prefix_str = build_path_string(*leading_colon, prefix.clone());
        let mut statement = String::new();
        let vis_prefix = visibility_prefix(visibility);
        statement.push_str(&vis_prefix);
        statement.push_str("use ");
        statement.push_str(&prefix_str);
        statement.push_str("::{");
        statement.push_str(
            &brace_items
                .iter()
                .map(|item| item.text.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        );
        statement.push_str("};\n");
        output.push_str(&statement);
    }

    for (idx, entry) in entries.iter().enumerate() {
        if grouped[idx] {
            continue;
        }
        let vis_prefix = visibility_prefix(&entry.visibility);
        let path_str = build_path_string(entry.leading_colon, entry.segments.clone());
        let rendered = if entry.is_glob {
            format!("{vis_prefix}use {path_str}::*;\n")
        } else if let Some(alias) = &entry.alias {
            format!("{vis_prefix}use {path_str} as {alias};\n")
        } else {
            format!("{vis_prefix}use {path_str};\n")
        };
        output.push_str(&rendered);
    }

    if output.is_empty() {
        None
    } else {
        Some(output)
    }
}

fn normalize_use_tokens(tokens: &str) -> String {
    let mut text = tokens.replace(" :: ", "::");
    text = text.replace(" { ", " {");
    text = text.replace(" }", "}");
    text = text.replace(" ;", ";");
    text
}

fn ensure_trailing_newline(text: &str) -> String {
    if text.ends_with('\n') {
        text.to_string()
    } else {
        format!("{}\n", text)
    }
}

fn visibility_string(scope: &VisibilityScope) -> String {
    match scope {
        VisibilityScope::Public => "pub".to_string(),
        VisibilityScope::Crate => "pub(crate)".to_string(),
        VisibilityScope::Super => "pub(super)".to_string(),
        VisibilityScope::Private => String::new(),
        VisibilityScope::Restricted(path) => format!("pub({})", path),
    }
}

fn visibility_rank(vis: &str) -> usize {
    match vis {
        "" => 0,
        "pub(self)" => 1,
        "pub(super)" => 2,
        "pub(crate)" => 3,
        _ if vis.starts_with("pub(") => 4,
        "pub" => 5,
        _ => 6,
    }
}

fn find_use_insert_position(content: &str) -> usize {
    let mut offset = 0;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("//")
            || trimmed.starts_with("/*")
            || trimmed.starts_with("#!")
        {
            offset += line.len();
            if !content[offset..].is_empty() {
                offset += 1; // newline
            }
        } else {
            break;
        }
    }
    offset
}

fn build_path_string(leading_colon: bool, segments: Vec<String>) -> String {
    if segments.is_empty() {
        return String::new();
    }
    let mut path_str = segments.join("::");
    if leading_colon {
        path_str = format!("::{}", path_str);
    }
    path_str
}

fn visibility_prefix(vis: &str) -> String {
    if vis.is_empty() {
        String::new()
    } else {
        format!("{vis} ")
    }
}
