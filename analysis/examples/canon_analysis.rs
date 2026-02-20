//! Real analysis of the `canon` crate via compiler_capture.
//!
//! Real analysis of the `canon` crate via compiler_capture.
//!
//! Run with:
//!   cargo run --example canon_analysis -p analysis --features rustc_frontend
//!
//! What this does:
//!   1. Ingests canon/src via RustcFrontend -> GraphDelta stream
//!   2. Materializes a GraphSnapshot (nodes = symbols, edges = calls/contains/refs)
//!   3. Runs all analysis domains over the real graph

#![feature(rustc_private)]

use analysis::report;
use compiler_capture::frontends::rustc::RustcFrontend;
use compiler_capture::multi_capture::capture_project;
use compiler_capture::project::CargoProject;
use database::graph_log::{GraphDelta, WireEdge, WireNode};
use std::collections::HashSet;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/canon");
    println!("=== canon analysis ===");
    println!("ingesting: {}\n", project_path.display());

    // ── 1. Ingest via compiler_capture ───────────────────────────────────────
    let project = CargoProject::from_entry(project_path)?;
    let frontend = RustcFrontend::new();
    let artifacts = capture_project(&frontend, &project, &[])?;

    // ── 2. Materialize GraphSnapshot from deltas ──────────────────────────────
    let mut nodes: Vec<WireNode> = Vec::new();
    let mut edges: Vec<WireEdge> = Vec::new();
    let mut seen_nodes = HashSet::new();
    let mut seen_edges = HashSet::new();
    for delta in &artifacts.graph_deltas {
        match delta {
            GraphDelta::AddNode(n) => {
                if seen_nodes.insert(n.id.clone()) {
                    nodes.push(n.clone());
                }
            }
            GraphDelta::AddEdge(e) => {
                if seen_edges.insert(e.id.clone()) {
                    edges.push(e.clone());
                }
            }
        }
    }
    println!("captured: {} nodes, {} edges\n", nodes.len(), edges.len());

    // ── 3. Run all analysis domains via report module ────────────────────────
    let report = report::run_all(&nodes, &edges);
    report.print();

    Ok(())
}
