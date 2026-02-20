#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_span;

use database::graph_log::{GraphDelta, WireEdgeId, WireNodeId};
use rustc_driver::{catch_with_exit_code, run_compiler, Callbacks, Compilation};
use rustc_hir::def::DefKind;
use rustc_interface::interface::Compiler;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::io::{self, Read, Write};
use std::process::ExitCode;

#[derive(Serialize, Deserialize)]
struct CaptureRequest {
    entry: String,
    args: Vec<String>,
    env_vars: Vec<(String, String)>,
    metadata: Metadata,
}

#[derive(Serialize, Deserialize, Clone)]
struct Metadata {
    edition: String,
    rust_version: Option<String>,
    crate_type: String,
    target_triple: String,
    target_name: Option<String>,
    workspace_root: Option<String>,
    package_name: Option<String>,
    package_version: Option<String>,
    package_features: Vec<String>,
    cfg_flags: Vec<String>,
}

fn main() {
    let mut stdin = String::new();
    io::stdin().read_to_string(&mut stdin).expect("read stdin");
    let req: CaptureRequest = serde_json::from_str(&stdin).expect("parse request");

    for (k, v) in &req.env_vars {
        unsafe { std::env::set_var(k, v); }
    }

    let mut cb = DriverCb { deltas: None, meta: req.metadata.clone() };
    let exit = catch_with_exit_code(|| run_compiler(&req.args, &mut cb));

    for (k, _) in &req.env_vars {
        unsafe { std::env::remove_var(k); }
    }

    if exit != ExitCode::SUCCESS {
        eprintln!("rustc exited with {:?}", exit);
        std::process::exit(1);
    }

    let deltas = cb.deltas.unwrap_or_default();
    let json = serde_json::to_string(&deltas).expect("serialize deltas");
    io::stdout().write_all(json.as_bytes()).expect("write stdout");
}

struct DriverCb {
    deltas: Option<Vec<GraphDelta>>,
    meta: Metadata,
}

impl Callbacks for DriverCb {
    fn after_analysis<'tcx>(&mut self, _: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        self.deltas = Some(collect(tcx, &self.meta));
        Compilation::Stop
    }
}

fn collect<'tcx>(tcx: TyCtxt<'tcx>, meta: &Metadata) -> Vec<GraphDelta> {
    let mut out: Vec<GraphDelta> = Vec::new();
    let mut cache: HashMap<DefId, WireNodeId> = HashMap::new();

    let crate_name = tcx.crate_name(rustc_span::def_id::LOCAL_CRATE).to_string();
    let crate_key = format!("{}::", crate_name);
    let crate_id = WireNodeId::from_key(&crate_key);
    out.push(GraphDelta::AddNode(database::graph_log::WireNode {
        id: crate_id.clone(),
        key: crate_key.clone(),
        label: "crate".into(),
        metadata: {
            let mut m = BTreeMap::new();
            m.insert("kind".into(), "crate".into());
            if let Some(p) = &meta.package_name { m.insert("package".into(), p.clone()); }
            if let Some(v) = &meta.package_version { m.insert("version".into(), v.clone()); }
            m
        },
    }));

    // All defined items
    for local_def_id in tcx.hir_crate_items(()).definitions() {
        let def_id = local_def_id.to_def_id();
        let def_kind = tcx.def_kind(def_id);
        let label = format!("{:?}", def_kind).to_lowercase();
        let path = tcx.def_path_str(def_id);
        let node_id = WireNodeId::from_key(&path);
        cache.insert(def_id, node_id.clone());

        let mut metadata = BTreeMap::new();
        metadata.insert("kind".into(), label.clone());
        metadata.insert("path".into(), path.clone());
        if let Some(tn) = &meta.target_name {
            metadata.insert("target_name".into(), tn.clone());
        }
        if let Some(local) = def_id.as_local() {
            let vis = tcx.visibility(local);
            metadata.insert("visibility".into(), format!("{:?}", vis).to_lowercase());
        }

        out.push(GraphDelta::AddNode(database::graph_log::WireNode {
            id: node_id.clone(),
            key: path.clone(),
            label,
            metadata,
        }));
        out.push(GraphDelta::AddEdge(database::graph_log::WireEdge {
            id: WireEdgeId::from_components(&crate_id, &node_id, "contains"),
            from: crate_id.clone(),
            to: node_id,
            kind: "contains".into(),
            metadata: BTreeMap::new(),
        }));
    }

    // Call edges from MIR
    for &local_def in tcx.mir_keys(()).iter() {
        let caller_def_id = local_def.to_def_id();
        if !matches!(tcx.def_kind(caller_def_id), DefKind::Fn | DefKind::AssocFn) { continue; }
        if !tcx.is_mir_available(caller_def_id) { continue; }
        let body = tcx.optimized_mir(local_def);
        let caller_path = tcx.def_path_str(caller_def_id);
        let caller_id = WireNodeId::from_key(&caller_path);

        let is_unsafe = body.basic_blocks.iter().any(|bb| {
            bb.statements.iter().any(|s| {
                matches!(s.kind, rustc_middle::mir::StatementKind::Intrinsic(_))
            })
        });
        {
            let mut metadata = BTreeMap::new();
            metadata.insert("kind".into(), "fn".into());
            metadata.insert("path".into(), caller_path.clone());
            metadata.insert("is_unsafe".into(), is_unsafe.to_string());
            if let Some(tn) = &meta.target_name {
                metadata.insert("target_name".into(), tn.clone());
            }
            out.push(GraphDelta::AddNode(database::graph_log::WireNode {
                id: caller_id.clone(),
                key: caller_path.clone(),
                label: "fn".into(),
                metadata,
            }));
        }

        for bb in body.basic_blocks.iter() {
            if let Some(term) = &bb.terminator {
                if let rustc_middle::mir::TerminatorKind::Call { func, .. } = &term.kind {
                    if let rustc_middle::mir::Operand::Constant(box_const) = func {
                        if let rustc_middle::ty::TyKind::FnDef(callee_def_id, _) =
                            box_const.const_.ty().kind()
                        {
                            let callee_path = tcx.def_path_str(*callee_def_id);
                            let callee_id = WireNodeId::from_key(&callee_path);
                            out.push(GraphDelta::AddEdge(database::graph_log::WireEdge {
                                id: WireEdgeId::from_components(&caller_id, &callee_id, "call"),
                                from: caller_id.clone(),
                                to: callee_id,
                                kind: "call".into(),
                                metadata: BTreeMap::new(),
                            }));
                        }
                    }
                }
            }
        }
    }

    out
}
