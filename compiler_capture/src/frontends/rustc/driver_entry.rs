//! Entry point for the `capture_driver` subprocess.
//! Reads CaptureRequest from stdin, runs rustc, writes Vec<GraphDelta> to stdout.

#![cfg(feature = "rustc_frontend")]

use super::frontend_context::FrontendMetadata;
use crate::compiler_capture::graph::{DeltaCollector, GraphDelta, NodeId};
use rustc_driver::{catch_with_exit_code, run_compiler, Callbacks, Compilation};
use rustc_hir::def::DefKind;
use rustc_interface::interface::Compiler;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::process::ExitCode;

#[derive(Serialize, Deserialize)]
pub struct CaptureRequest {
    pub entry: String,
    pub args: Vec<String>,
    pub env_vars: Vec<(String, String)>,
    pub metadata: FrontendMetadata,
}

pub fn run_capture_driver() -> anyhow::Result<()> {
    // Read request from stdin
    let mut stdin = String::new();
    io::stdin().read_to_string(&mut stdin)?;
    let req: CaptureRequest = serde_json::from_str(&stdin)?;

    // Set env vars
    for (k, v) in &req.env_vars {
        unsafe { std::env::set_var(k, v); }
    }

    // Run rustc and collect deltas
    let mut callbacks = DriverCallbacks::new(req.metadata);
    let exit = catch_with_exit_code(|| run_compiler(&req.args, &mut callbacks));

    // Unset env vars
    for (k, _) in &req.env_vars {
        unsafe { std::env::remove_var(k); }
    }

    if exit != ExitCode::SUCCESS {
        anyhow::bail!("rustc exited with {:?}", exit);
    }

    let deltas = callbacks.deltas.unwrap_or_default();
    let json = serde_json::to_string(&deltas)?;
    io::stdout().write_all(json.as_bytes())?;
    Ok(())
}

struct DriverCallbacks {
    deltas: Option<Vec<GraphDelta>>,
    metadata: FrontendMetadata,
}

impl DriverCallbacks {
    fn new(metadata: FrontendMetadata) -> Self {
        Self { deltas: None, metadata }
    }
}

impl Callbacks for DriverCallbacks {
    fn after_analysis<'tcx>(&mut self, _compiler: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        self.deltas = Some(build_graph_deltas(tcx, &self.metadata));
        Compilation::Stop
    }
}

fn build_graph_deltas<'tcx>(tcx: TyCtxt<'tcx>, metadata: &FrontendMetadata) -> Vec<GraphDelta> {
    use super::super::super::frontends::rustc::{
        crate_metadata, item_capture::*, mir_capture::capture_function,
        node_builder::ensure_node, trait_capture::*, type_capture::capture_function_types,
    };
    let mut builder = DeltaCollector::new();
    let mut cache: HashMap<DefId, NodeId> = HashMap::new();
    crate_metadata::capture_crate_metadata(&mut builder, tcx, metadata);
    let crate_items = tcx.hir_crate_items(());
    for local_def_id in crate_items.definitions() {
        let def_id = local_def_id.to_def_id();
        match tcx.def_kind(def_id) {
            DefKind::Mod => {
                capture_mod(&mut builder, tcx, def_id, &mut cache, metadata);
            }
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
        if !matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn)
            || !tcx.is_mir_available(def_id)
        {
            continue;
        }
        let caller_id = ensure_node(&mut builder, tcx, def_id, &mut cache, metadata);
        capture_function(&mut builder, tcx, local_def, caller_id, &mut cache, metadata);
        capture_function_types(&mut builder, tcx, local_def, &mut cache, metadata);
    }
    builder.into_deltas()
}
