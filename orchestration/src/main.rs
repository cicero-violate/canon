//! Orchestration — JSON ModelIR → analyze → emit Rust source.
//!
//! Usage:
//!   orchestration <model_ir.json> <output_dir>
//!
//! Pipeline:
//!   1. load:    read JSON -> deserialize ModelIR
//!   2. edges:   populate graph builders from ir.edge_hints
//!   3. derive:  build all five CSR graphs
//!   4. solve:   run all five solvers
//!   5. emit:    walk module_graph -> write .rs files to output_dir

use anyhow::{Context, Result};
use std::path::PathBuf;
use model::ir::model_ir::ModelIR;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let json_path = args.next().map(PathBuf::from)
        .context("usage: orchestration <model_ir.json> <output_dir>")?;
    let out_dir = args.next().map(PathBuf::from)
        .context("usage: orchestration <model_ir.json> <output_dir>")?;

    run_pipeline(json_path, out_dir)
}

fn run_pipeline(json_path: PathBuf, out_dir: PathBuf) -> Result<()> {
    // ── Stage 1: load ModelIR from JSON ─────────────────────────────────────
    println!("Loading {:?}", json_path);
    let json = std::fs::read_to_string(&json_path)
        .with_context(|| format!("cannot read {:?}", json_path))?;
    let mut ir: ModelIR = serde_json::from_str(&json)
        .with_context(|| format!("cannot parse ModelIR from {:?}", json_path))?;
    println!("  nodes: {}", ir.nodes.len());

    // ── Stage 2: derive constraint graphs ───────────────────────────────────
    println!("Deriving constraint graphs...");
    analyzer::analyze(&mut ir)
        .context("analysis failed")?;

    // ── Stage 3: project → emit to disk ─────────────────────────────────────
    println!("Emitting source...");
    let plan = projection::project(&ir)
        .context("project failed")?;
    projection::emit_to_disk(&plan, &out_dir)
        .context("emit failed")?;

    println!("Emitted {} file(s) to {:?}", plan.files.len(), out_dir);

    // ── Stage 4: write back annotated JSON snapshot ──────────────────────────
    let snap_path = out_dir.join("model_ir_solved.json");
    let snap = serde_json::to_string_pretty(&ir)
        .context("json serialize failed")?;
    std::fs::create_dir_all(&out_dir)?;
    std::fs::write(&snap_path, snap)
        .context("json write failed")?;
    println!("Snapshot written to {:?}", snap_path);

    println!("Pipeline complete.");
    Ok(())
}
