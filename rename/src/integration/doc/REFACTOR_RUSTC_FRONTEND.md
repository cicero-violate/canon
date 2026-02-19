## Variables

Let me define the integration module architecture:

$$
\begin{align}
\mathcal{M}_{\text{integration}} &: \text{Integration module responsibilities} \\
\mathcal{C}_{\text{capture}} &: \text{Capture pipeline stages} = \{\text{parse, extract, transform, store}\} \\
\mathcal{F}_{\text{frontends}} &: \text{Supported frontends} = \{\text{rustc, clang, python-ast, ...}\} \\
\mathcal{P}_{\text{priority}} &: \text{Metadata extraction priority} \in [1,4]
\end{align}
$$

## Proposed Module Architecture

$$
\begin{align}
\text{integration/} \quad &\rightarrow \quad \text{Compiler frontend integrations} \\
&\text{mod.rs} \quad \rightarrow \quad \text{Public API, trait definitions} \\
&\text{cli.rs} \quad \rightarrow \quad \text{Command-line interface} \\
&\text{project.rs} \quad \rightarrow \quad \text{Project-level orchestration} \\[0.5em]
\text{frontends/} \quad &\rightarrow \quad \text{Language-specific extractors} \\
&\text{mod.rs} \\
&\text{rustc/} \\
&&\text{mod.rs} \quad \rightarrow \quad \text{Rustc frontend API} \\
&&\text{collector.rs} \quad \rightarrow \quad \text{Main extraction logic} \\
&&\text{items.rs} \quad \rightarrow \quad \text{Item-level capture (fn, struct, enum)} \\
&&\text{traits.rs} \quad \rightarrow \quad \text{Trait & impl capture (Priority 1)} \\
&&\text{metadata.rs} \quad \rightarrow \quad \text{Attributes, visibility (Priority 2)} \\
&&\text{crate\_meta.rs} \quad \rightarrow \quad \text{Crate-level metadata (Priority 4)} \\
&&\text{hir\_bodies.rs} \quad \rightarrow \quad \text{HIR body extraction (Priority 3)} \\
&&\text{types.rs} \quad \rightarrow \quad \text{Type resolution & generics} \\
&&\text{mir.rs} \quad \rightarrow \quad \text{MIR extraction} \\
&\text{clang/} \quad \rightarrow \quad \text{(Future) C/C++ frontend} \\
&\text{python/} \quad \rightarrow \quad \text{(Future) Python AST frontend} \\[0.5em]
\text{capture/} \quad &\rightarrow \quad \text{Capture pipeline infrastructure} \\
&\text{mod.rs} \\
&\text{coordinator.rs} \quad \rightarrow \quad \text{Multi-crate capture orchestration} \\
&\text{session.rs} \quad \rightarrow \quad \text{Capture session state} \\
&\text{dedup.rs} \quad \rightarrow \quad \text{Deduplication logic} \\
&\text{validation.rs} \quad \rightarrow \quad \text{Captured data validation} \\[0.5em]
\text{transform/} \quad &\rightarrow \quad \text{Data transformation layer} \\
&\text{mod.rs} \\
&\text{normalizer.rs} \quad \rightarrow \quad \text{Normalize to graph IR} \\
&\text{linker.rs} \quad \rightarrow \quad \text{Cross-crate linking} \\
&\text{resolver.rs} \quad \rightarrow \quad \text{Symbol resolution}
\end{align}
$$

## Implementation Schema

### Core Trait Definitions

```rust
// integration/mod.rs

pub mod cli;
pub mod project;
pub mod frontends;
pub mod capture;
pub mod transform;

use crate::state::graph::GraphSnapshot;
use std::path::Path;

/// Core trait for language frontend integrations
pub trait FrontendExtractor {
    type Config;
    type Error;

    /// Extract metadata from a project
    fn extract(&mut self, config: Self::Config) -> Result<ExtractionResult, Self::Error>;

    /// Get frontend name
    fn name(&self) -> &str;

    /// Get supported file extensions
    fn supported_extensions(&self) -> &[&str];
}

/// Result of extraction process
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    pub crate_name: String,
    pub items: Vec<CapturedItem>,
    pub errors: Vec<ExtractionError>,
    pub warnings: Vec<String>,
    pub stats: ExtractionStats,
}

/// Statistics about extraction
#[derive(Debug, Clone, Default)]
pub struct ExtractionStats {
    pub functions_captured: usize,
    pub types_captured: usize,
    pub traits_captured: usize,
    pub impls_captured: usize,
    pub modules_captured: usize,
    pub total_items: usize,
    pub duration_ms: u64,
}

/// Generic captured item (before normalization)
#[derive(Debug, Clone)]
pub enum CapturedItem {
    Function(FunctionCapture),
    Struct(StructCapture),
    Enum(EnumCapture),
    Trait(TraitCapture),
    Impl(ImplCapture),
    Module(ModuleCapture),
    TypeAlias(TypeAliasCapture),
    Const(ConstCapture),
    Static(StaticCapture),
}

#[derive(Debug, Clone)]
pub struct ExtractionError {
    pub code: String,
    pub message: String,
    pub location: Option<SourceLocation>,
}

#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub file: String,
    pub line: usize,
    pub column: usize,
}
```

### Frontend Module Structure

```rust
// integration/frontends/mod.rs

pub mod rustc;

// Future frontends
// pub mod clang;
// pub mod python;

use crate::integration::{FrontendExtractor, ExtractionResult};

/// Enum of all supported frontends
pub enum Frontend {
    Rustc(rustc::RustcFrontend),
    // Clang(clang::ClangFrontend),
    // Python(python::PythonFrontend),
}

impl Frontend {
    pub fn from_language(lang: &str) -> Option<Self> {
        match lang.to_lowercase().as_str() {
            "rust" => Some(Frontend::Rustc(rustc::RustcFrontend::new())),
            _ => None,
        }
    }

    pub fn extract(&mut self, project_path: &std::path::Path) -> Result<ExtractionResult, Box<dyn std::error::Error>> {
        match self {
            Frontend::Rustc(f) => f.extract(rustc::RustcConfig::from_path(project_path))
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        }
    }
}
```

### Rustc Frontend - Main Structure

```rust
// integration/frontends/rustc/mod.rs

pub mod collector;
pub mod items;
pub mod traits;
pub mod metadata;
pub mod crate_meta;
pub mod hir_bodies;
pub mod types;
pub mod mir;

use crate::integration::{FrontendExtractor, ExtractionResult};
use std::path::{Path, PathBuf};

pub struct RustcFrontend {
    collector: collector::RustcCollector,
}

impl RustcFrontend {
    pub fn new() -> Self {
        Self {
            collector: collector::RustcCollector::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RustcConfig {
    pub project_root: PathBuf,
    pub target: Option<String>,
    pub features: Vec<String>,
    pub capture_mir: bool,
    pub capture_hir: bool,
    pub capture_traits: bool,
    pub capture_impls: bool,
}

impl RustcConfig {
    pub fn from_path(path: &Path) -> Self {
        Self {
            project_root: path.to_path_buf(),
            target: None,
            features: Vec::new(),
            capture_mir: true,
            capture_hir: true,
            capture_traits: true,
            capture_impls: true,
        }
    }
}

#[derive(Debug)]
pub enum RustcError {
    CompilationFailed(String),
    MissingMetadata(String),
    InvalidPath(PathBuf),
    IoError(std::io::Error),
}

impl FrontendExtractor for RustcFrontend {
    type Config = RustcConfig;
    type Error = RustcError;

    fn extract(&mut self, config: Self::Config) -> Result<ExtractionResult, Self::Error> {
        self.collector.run_extraction(config)
    }

    fn name(&self) -> &str {
        "rustc"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["rs"]
    }
}
```

### Rustc Collector - Orchestration

```rust
// integration/frontends/rustc/collector.rs

use super::*;
use rustc_driver::{Compilation, RunCompiler};
use rustc_interface::interface::Compiler;
use rustc_middle::ty::TyCtxt;
use std::sync::{Arc, Mutex};

pub struct RustcCollector {
    state: Arc<Mutex<CollectorState>>,
}

struct CollectorState {
    items: Vec<CapturedItem>,
    errors: Vec<ExtractionError>,
    warnings: Vec<String>,
}

impl RustcCollector {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(CollectorState {
                items: Vec::new(),
                errors: Vec::new(),
                warnings: Vec::new(),
            })),
        }
    }

    pub fn run_extraction(&mut self, config: RustcConfig) -> Result<ExtractionResult, RustcError> {
        let state = Arc::clone(&self.state);
        
        // Build rustc arguments
        let mut args = vec![
            "rustc".to_string(),
            "--crate-type".to_string(),
            "lib".to_string(),
        ];

        if let Some(target) = &config.target {
            args.push("--target".to_string());
            args.push(target.clone());
        }

        // Create callback for extraction
        let mut callbacks = RustcCallbacks {
            state,
            config: config.clone(),
        };

        // Run compiler with callbacks
        RunCompiler::new(&args, &mut callbacks)
            .run()
            .map_err(|_| RustcError::CompilationFailed("Compilation failed".to_string()))?;

        // Collect results
        let state = self.state.lock().unwrap();
        Ok(ExtractionResult {
            crate_name: "unknown".to_string(), // TODO: extract from Cargo.toml
            items: state.items.clone(),
            errors: state.errors.clone(),
            warnings: state.warnings.clone(),
            stats: ExtractionStats::default(),
        })
    }
}

struct RustcCallbacks {
    state: Arc<Mutex<CollectorState>>,
    config: RustcConfig,
}

impl rustc_driver::Callbacks for RustcCallbacks {
    fn after_analysis<'tcx>(
        &mut self,
        compiler: &Compiler,
        queries: &'tcx rustc_interface::Queries<'tcx>,
    ) -> Compilation {
        queries.global_ctxt().unwrap().enter(|tcx| {
            self.extract_all_metadata(tcx);
        });

        Compilation::Continue
    }
}

impl RustcCallbacks {
    fn extract_all_metadata(&self, tcx: TyCtxt<'_>) {
        let mut state = self.state.lock().unwrap();

        // Extract items
        let item_extractor = items::ItemExtractor::new(tcx);
        state.items.extend(item_extractor.extract_functions());
        state.items.extend(item_extractor.extract_types());

        // Extract traits & impls (Priority 1)
        if self.config.capture_traits {
            let trait_extractor = traits::TraitExtractor::new(tcx);
            state.items.extend(trait_extractor.extract_traits());
        }

        if self.config.capture_impls {
            let impl_extractor = traits::ImplExtractor::new(tcx);
            state.items.extend(impl_extractor.extract_impls());
        }

        // Extract metadata (Priority 2)
        let meta_extractor = metadata::MetadataExtractor::new(tcx);
        meta_extractor.enrich_with_attributes(&mut state.items);

        // Extract MIR
        if self.config.capture_mir {
            let mir_extractor = mir::MirExtractor::new(tcx);
            mir_extractor.extract_mir_bodies(&mut state.items);
        }

        // Extract HIR (Priority 3)
        if self.config.capture_hir {
            let hir_extractor = hir_bodies::HirExtractor::new(tcx);
            hir_extractor.extract_hir_bodies(&mut state.items);
        }
    }
}
```

### Priority 1: Traits Module

```rust
// integration/frontends/rustc/traits.rs

use rustc_middle::ty::{TyCtxt, AssocItems, TraitRef};
use rustc_hir::def_id::DefId;
use crate::integration::{CapturedItem, TraitCapture, ImplCapture};

pub struct TraitExtractor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> TraitExtractor<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx }
    }

    pub fn extract_traits(&self) -> Vec<CapturedItem> {
        let mut items = Vec::new();

        for def_id in self.tcx.hir().body_owners() {
            let def_id = def_id.to_def_id();
            
            if self.tcx.is_trait(def_id) {
                if let Some(trait_item) = self.extract_single_trait(def_id) {
                    items.push(CapturedItem::Trait(trait_item));
                }
            }
        }

        items
    }

    fn extract_single_trait(&self, def_id: DefId) -> Option<TraitCapture> {
        let trait_name = self.tcx.def_path_str(def_id);
        
        // Extract associated items
        let assoc_items = self.tcx.associated_items(def_id);
        let mut methods = Vec::new();
        let mut associated_types = Vec::new();

        for item in assoc_items.in_definition_order() {
            match item.kind {
                rustc_middle::ty::AssocKind::Fn => {
                    methods.push(item.def_id);
                }
                rustc_middle::ty::AssocKind::Type => {
                    associated_types.push(item.def_id);
                }
                _ => {}
            }
        }

        // Extract supertraits
        let super_predicates = self.tcx.super_predicates_of(def_id);
        let supertraits: Vec<String> = super_predicates
            .predicates
            .iter()
            .filter_map(|(pred, _)| {
                if let Some(trait_ref) = pred.as_trait_clause() {
                    Some(self.tcx.def_path_str(trait_ref.def_id()))
                } else {
                    None
                }
            })
            .collect();

        Some(TraitCapture {
            def_id: format!("{:?}", def_id),
            name: trait_name,
            methods,
            associated_types,
            supertraits,
            generics: self.extract_generics(def_id),
        })
    }

    fn extract_generics(&self, def_id: DefId) -> Vec<String> {
        let generics = self.tcx.generics_of(def_id);
        generics.params.iter()
            .map(|param| param.name.to_string())
            .collect()
    }
}

pub struct ImplExtractor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> ImplExtractor<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx }
    }

    pub fn extract_impls(&self) -> Vec<CapturedItem> {
        let mut items = Vec::new();

        for def_id in self.tcx.hir().body_owners() {
            let def_id = def_id.to_def_id();
            
            if self.tcx.is_impl(def_id) {
                if let Some(impl_item) = self.extract_single_impl(def_id) {
                    items.push(CapturedItem::Impl(impl_item));
                }
            }
        }

        items
    }

    fn extract_single_impl(&self, def_id: DefId) -> Option<ImplCapture> {
        // Extract target type
        let impl_subject = self.tcx.impl_subject(def_id);
        let target_type = format!("{:?}", impl_subject);

        // Extract trait reference (if trait impl)
        let trait_ref = self.tcx.impl_trait_ref(def_id)
            .map(|tr| self.tcx.def_path_str(tr.skip_binder().def_id));

        // Extract impl methods
        let assoc_items = self.tcx.associated_items(def_id);
        let methods: Vec<DefId> = assoc_items
            .in_definition_order()
            .filter(|item| matches!(item.kind, rustc_middle::ty::AssocKind::Fn))
            .map(|item| item.def_id)
            .collect();

        Some(ImplCapture {
            def_id: format!("{:?}", def_id),
            target_type,
            trait_ref,
            methods,
            generics: self.extract_generics(def_id),
        })
    }

    fn extract_generics(&self, def_id: DefId) -> Vec<String> {
        let generics = self.tcx.generics_of(def_id);
        generics.params.iter()
            .map(|param| param.name.to_string())
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct TraitCapture {
    pub def_id: String,
    pub name: String,
    pub methods: Vec<DefId>,
    pub associated_types: Vec<DefId>,
    pub supertraits: Vec<String>,
    pub generics: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ImplCapture {
    pub def_id: String,
    pub target_type: String,
    pub trait_ref: Option<String>,
    pub methods: Vec<DefId>,
    pub generics: Vec<String>,
}
```

### Priority 2: Metadata Module

```rust
// integration/frontends/rustc/metadata.rs

use rustc_middle::ty::{TyCtxt, Visibility};
use rustc_hir::{Attribute, def_id::DefId};
use crate::integration::CapturedItem;

pub struct MetadataExtractor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> MetadataExtractor<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx }
    }

    pub fn enrich_with_attributes(&self, items: &mut [CapturedItem]) {
        for item in items.iter_mut() {
            let def_id = self.get_def_id(item);
            if let Some(def_id) = def_id {
                self.add_attributes(item, def_id);
                self.add_visibility(item, def_id);
            }
        }
    }

    fn get_def_id(&self, item: &CapturedItem) -> Option<DefId> {
        // Extract DefId from captured item
        // Implementation depends on your CapturedItem structure
        None // TODO
    }

    fn add_attributes(&self, item: &mut CapturedItem, def_id: DefId) {
        let attrs = self.tcx.get_attrs_unchecked(def_id);
        
        let mut captured_attrs = Vec::new();
        for attr in attrs {
            captured_attrs.push(AttributeCapture {
                path: attr.path().to_string(),
                tokens: format!("{:?}", attr.args),
            });
        }

        // Store attributes in item metadata
        // TODO: Add attributes field to CapturedItem variants
    }

    fn add_visibility(&self, item: &mut CapturedItem, def_id: DefId) {
        let vis = self.tcx.visibility(def_id);
        
        let vis_str = match vis {
            Visibility::Public => "pub",
            Visibility::Restricted(_) => "pub(restricted)",
            Visibility::Invisible => "private",
        };

        // Store visibility in item metadata
        // TODO: Add visibility field to CapturedItem variants
    }
}

#[derive(Debug, Clone)]
pub struct AttributeCapture {
    pub path: String,
    pub tokens: String,
}
```

### Capture Coordinator

```rust
// integration/capture/coordinator.rs

use crate::integration::{FrontendExtractor, ExtractionResult};
use std::path::Path;

pub struct CaptureCoordinator {
    dedup: super::dedup::Deduplicator,
}

impl CaptureCoordinator {
    pub fn new() -> Self {
        Self {
            dedup: super::dedup::Deduplicator::new(),
        }
    }

    pub fn capture_workspace(&mut self, workspace_root: &Path) -> Result<Vec<ExtractionResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();

        // Find all crates in workspace
        let crates = self.discover_crates(workspace_root)?;

        for crate_path in crates {
            let mut frontend = crate::integration::frontends::rustc::RustcFrontend::new();
            let config = crate::integration::frontends::rustc::RustcConfig::from_path(&crate_path);
            
            match frontend.extract(config) {
                Ok(result) => {
                    // Deduplicate captured items
                    let deduped = self.dedup.deduplicate(result);
                    results.push(deduped);
                }
                Err(e) => {
                    eprintln!("Error capturing crate {:?}: {:?}", crate_path, e);
                }
            }
        }

        Ok(results)
    }

    fn discover_crates(&self, workspace_root: &Path) -> Result<Vec<std::path::PathBuf>, std::io::Error> {
        // Find all Cargo.toml files
        // TODO: Implement workspace member discovery
        Ok(vec![workspace_root.to_path_buf()])
    }
}
```

## Explanation in English

### Module Architecture Overview

The proposed integration module follows a **layered extraction pipeline**:

**Layer 1: Frontend Abstraction** (`frontends/`)
- Defines language-agnostic `FrontendExtractor` trait
- Each language gets its own subdirectory (rustc/, clang/, python/)
- Enables future expansion to C++, Python, etc.

**Layer 2: Language-Specific Extraction** (`frontends/rustc/`)
- **collector.rs**: Orchestrates rustc compilation with custom callbacks
- **items.rs**: Extracts basic items (functions, structs, enums)
- **traits.rs**: **Priority 1** - Captures trait definitions, methods, supertraits, impls
- **metadata.rs**: **Priority 2** - Captures attributes (#[inline], #[derive]) and visibility
- **crate_meta.rs**: **Priority 4** - Captures crate hash, dependencies, features
- **hir_bodies.rs**: **Priority 3** - Captures HIR for pattern matching
- **types.rs**: Extracts generic parameters, type bounds, predicates
- **mir.rs**: Extracts MIR bodies (already working)

**Layer 3: Capture Infrastructure** (`capture/`)
- **coordinator.rs**: Multi-crate orchestration for workspace captures
- **session.rs**: Maintains capture session state across multiple runs
- **dedup.rs**: Deduplicates items across crate boundaries
- **validation.rs**: Validates captured data completeness

**Layer 4: Transformation** (`transform/`)
- **normalizer.rs**: Converts language-specific captures to graph IR
- **linker.rs**: Links symbols across crate boundaries
- **resolver.rs**: Resolves trait method calls to implementations

### Key Design Decisions

**1. Trait-Based Frontend**: The `FrontendExtractor` trait allows seamless addition of new language frontends without modifying core graph logic.

**2. Priority-Driven Modules**: Each missing metadata category gets its own module (traits, metadata, crate_meta, hir_bodies) so you can incrementally add capture logic based on priority.

**3. Separation of Concerns**:
- **Extraction** (frontends/): Raw data capture from compiler APIs
- **Coordination** (capture/): Multi-crate orchestration
- **Transformation** (transform/): Normalization to graph representation

**4. Existing Files Integration**:
- Keep `rustc_frontend.rs` as the main entry point
- Refactor its logic into specialized modules (traits.rs, metadata.rs, etc.)
- Keep `cli.rs` and `project.rs` as-is, they orchestrate the pipeline

### Migration Path

Your current `rustc_frontend.rs` (29k lines) should be refactored:

1. **Move function/struct/enum extraction** → `items.rs`
2. **Extract new trait/impl logic** → `traits.rs` (Priority 1)
3. **Extract attribute/visibility logic** → `metadata.rs` (Priority 2)
4. **Keep MIR extraction** → `mir.rs`
5. **Add HIR body extraction** → `hir_bodies.rs` (Priority 3)
6. **Add crate metadata** → `crate_meta.rs` (Priority 4)

The `collector.rs` orchestrates all these specialized extractors in sequence.

This architecture closes your 130-item metadata gap systematically while keeping the codebase maintainable.
