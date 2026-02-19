#![cfg(feature = "rustc_frontend")]

use super::context::FrontendMetadata;
use super::nodes::ensure_node;
use super::RustcFrontendError;
use crate::state::builder::KernelGraphBuilder;
use crate::state::graph::GraphSnapshot;
use crate::state::ids::NodeId;
use rustc_driver::{catch_with_exit_code, run_compiler, Callbacks, Compilation};
use rustc_hir::def::DefKind;
use rustc_interface::interface::Compiler;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::crate_meta;
use super::items::{capture_adt, capture_const_static, capture_type_alias};
use super::mir::capture_function;
use super::traits::{capture_impl, capture_trait};
use super::types::capture_types_from_function;

/// Real rustc frontend that runs `rustc_driver` to capture MIR call graphs.
#[derive(Debug, Clone)]
pub struct RustcFrontend {
    edition: String,
    crate_type: String,
    rust_version: Option<String>,
    target_name: Option<String>,
    workspace_root: Option<PathBuf>,
    package_name: Option<String>,
    package_version: Option<String>,
    package_features: Vec<String>,
    cfg_flags: Vec<String>,
}

impl Default for RustcFrontend {
    fn default() -> Self {
        Self {
            edition: "2021".into(),
            crate_type: "lib".into(),
            rust_version: None,
            target_name: None,
            workspace_root: None,
            package_name: None,
            package_version: None,
            package_features: Vec::new(),
            cfg_flags: Vec::new(),
        }
    }
}

impl RustcFrontend {
    /// Creates a new frontend with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the crate type used when invoking rustc (e.g., `lib`, `bin`).
    pub fn with_crate_type(mut self, crate_type: impl Into<String>) -> Self {
        self.crate_type = crate_type.into();
        self
    }

    /// Sets the Rust edition supplied to rustc.
    pub fn with_edition(mut self, edition: impl Into<String>) -> Self {
        self.edition = edition.into();
        self
    }

    /// Sets the Rust toolchain version recorded in snapshot metadata.
    pub fn with_rust_version(mut self, rust_version: impl Into<String>) -> Self {
        self.rust_version = Some(rust_version.into());
        self
    }

    /// Sets the Cargo target name (lib/bin/test name).
    pub fn with_target_name(mut self, target_name: impl Into<String>) -> Self {
        self.target_name = Some(target_name.into());
        self
    }

    /// Sets the workspace root recorded for this capture.
    pub fn with_workspace_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.workspace_root = Some(root.into());
        self
    }

    /// Declares the Cargo package metadata.
    pub fn with_package_info(
        mut self,
        name: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        self.package_name = Some(name.into());
        self.package_version = Some(version.into());
        self
    }

    /// Declares the set of enabled Cargo features.
    pub fn with_package_features<I, S>(mut self, features: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.package_features = features.into_iter().map(|s| s.into()).collect();
        self
    }

    /// Declares the cfg flags passed to rustc.
    pub fn with_cfg_flags<I, S>(mut self, cfgs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.cfg_flags = cfgs.into_iter().map(|s| s.into()).collect();
        self
    }

    /// Captures a [`GraphSnapshot`] from the provided Rust source file.
    pub fn capture_snapshot<P: AsRef<Path>>(
        &self,
        entry: P,
        extra_args: &[String],
        env_vars: &[(String, String)],
    ) -> Result<GraphSnapshot, RustcFrontendError> {
        let entry = fs::canonicalize(entry)?;
        let mut args = vec![
            "rustc".to_string(),
            entry.display().to_string(),
            format!("--crate-type={}", self.crate_type),
            format!("--edition={}", self.edition),
        ];
        if let Some(name) = entry.file_stem().and_then(|s| s.to_str()) {
            args.push(format!("--crate-name={}", name.replace('-', "_")));
        }
        let sysroot = determine_sysroot().map_err(RustcFrontendError::Sysroot)?;
        args.push("--sysroot".into());
        args.push(sysroot.to_string_lossy().into());
        args.extend(extra_args.iter().cloned());

        let metadata = FrontendMetadata {
            edition: self.edition.clone(),
            rust_version: self.rust_version.clone(),
            crate_type: self.crate_type.clone(),
            target_triple: std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()),
            target_name: self.target_name.clone(),
            workspace_root: self
                .workspace_root
                .as_ref()
                .map(|p| p.display().to_string()),
            package_name: self.package_name.clone(),
            package_version: self.package_version.clone(),
            package_features: self.package_features.clone(),
            cfg_flags: {
                let mut cfgs = self.cfg_flags.clone();
                cfgs.extend(extract_cfg_flags(extra_args));
                cfgs.sort();
                cfgs.dedup();
                cfgs
            },
        };
        let mut callbacks = SnapshotCallbacks::new(metadata);

        for (key, value) in env_vars {
            std::env::set_var(key, value);
        }

        let exit_code = catch_with_exit_code(|| run_compiler(&args, &mut callbacks));

        for (key, _) in env_vars {
            std::env::remove_var(key);
        }

        if exit_code != 0 {
            return Err(RustcFrontendError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("rustc exited with code {}", exit_code),
            )));
        }

        callbacks
            .into_snapshot()
            .ok_or(RustcFrontendError::MissingSnapshot)
    }
}

struct SnapshotCallbacks {
    snapshot: Option<GraphSnapshot>,
    metadata: FrontendMetadata,
}

impl SnapshotCallbacks {
    fn new(metadata: FrontendMetadata) -> Self {
        Self {
            snapshot: None,
            metadata,
        }
    }

    fn into_snapshot(self) -> Option<GraphSnapshot> {
        self.snapshot
    }
}

impl Callbacks for SnapshotCallbacks {
    fn after_analysis<'tcx>(&mut self, _compiler: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        self.snapshot = Some(build_snapshot_from_tcx(tcx, &self.metadata));
        Compilation::Stop
    }
}

fn extract_cfg_flags(extra_args: &[String]) -> Vec<String> {
    let mut cfgs = Vec::new();
    let mut iter = extra_args.iter().peekable();
    while let Some(arg) = iter.next() {
        if let Some(rest) = arg.strip_prefix("--cfg=") {
            cfgs.push(rest.to_string());
            continue;
        }
        if arg == "--cfg" {
            if let Some(next) = iter.next() {
                cfgs.push(next.to_string());
            }
            continue;
        }
    }
    cfgs
}

fn build_snapshot_from_tcx<'tcx>(tcx: TyCtxt<'tcx>, metadata: &FrontendMetadata) -> GraphSnapshot {
    let mut builder = KernelGraphBuilder::new();
    let mut cache: HashMap<DefId, NodeId> = HashMap::new();

    crate_meta::capture_crate_metadata(&mut builder, tcx, metadata);

    let crate_items = tcx.hir_crate_items(());
    for local_def_id in crate_items.definitions() {
        let def_id = local_def_id.to_def_id();
        let def_kind = tcx.def_kind(def_id);

        match def_kind {
            DefKind::Struct | DefKind::Enum | DefKind::Union => {
                capture_adt(&mut builder, tcx, def_id, &mut cache, metadata);
            }
            DefKind::Trait => {
                capture_trait(&mut builder, tcx, def_id, &mut cache, metadata);
            }
            DefKind::Impl { of_trait } => {
                capture_impl(&mut builder, tcx, def_id, &mut cache, metadata, of_trait);
            }
            DefKind::TyAlias => {
                capture_type_alias(&mut builder, tcx, def_id, &mut cache, metadata);
            }
            DefKind::Static { .. } | DefKind::Const => {
                capture_const_static(&mut builder, tcx, def_id, &mut cache, metadata);
            }
            _ => {}
        }
    }

    for &local_def in tcx.mir_keys(()).iter() {
        let def_id = local_def.to_def_id();
        if !is_supported_def(tcx.def_kind(def_id)) || !tcx.is_mir_available(def_id) {
            continue;
        }

        let caller_id = ensure_node(&mut builder, tcx, def_id, &mut cache, metadata);
        capture_function(
            &mut builder,
            tcx,
            local_def,
            caller_id,
            &mut cache,
            metadata,
        );
        capture_types_from_function(&mut builder, tcx, local_def, &mut cache, metadata);
    }

    builder.finalize()
}

fn is_supported_def(kind: DefKind) -> bool {
    matches!(kind, DefKind::Fn | DefKind::AssocFn)
}

fn determine_sysroot() -> Result<PathBuf, std::io::Error> {
    if let Ok(sysroot) = std::env::var("SYSROOT") {
        return Ok(PathBuf::from(sysroot));
    }

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let output = Command::new(rustc).args(["--print", "sysroot"]).output()?;
    Ok(PathBuf::from(
        String::from_utf8_lossy(&output.stdout).trim(),
    ))
}
