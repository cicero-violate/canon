use std::fs;

use crate::cli::Command;
use crate::diff::diff_ir;
use crate::io_utils::*;
use crate::version_gate::enforce_version_gate;

use canon::{
    generate_schema, validate_ir, render_impl_function,
    runtime::{TickExecutionMode, TickExecutor},
};

pub fn execute_command(cmd: Command) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        Command::Schema { pretty } => {
            println!("{}", generate_schema(pretty)?);
        }

        Command::Validate { path } => {
            let ir = load_ir(&path)?;
            validate_ir(&ir)?;
            enforce_version_gate(&ir)?;
            println!("Validation passed: `{}`.", path.display());
        }

        Command::Lint { ir } => {
            let ir_doc = load_ir(&ir)?;
            validate_ir(&ir_doc)?;
            println!("Lint passed.");
        }

        Command::RenderFn { ir, fn_id } => {
            let ir_doc = load_ir(&ir)?;
            let function = ir_doc
                .functions
                .iter()
                .find(|f| f.id == fn_id)
                .ok_or("unknown function")?;
            println!("{}", render_impl_function(function));
        }

        Command::DiffIr { before, after } => {
            let a = load_ir(&before)?;
            let b = load_ir(&after)?;
            println!("{}", diff_ir(&a, &b));
        }

        Command::Materialize { ir, layout, out_dir } => {
            let ir_doc = load_ir_or_semantic(&ir)?;
            validate_ir(&ir_doc)?;
            enforce_version_gate(&ir_doc)?;
            let layout_doc = load_layout(resolve_layout(layout, &ir))?;
            canon::materialize(&ir_doc, &layout_doc, Some(&out_dir));
            println!("Materialized into `{}`.", out_dir.display());
        }

        Command::Ingest { .. } => {
            // Destructure so the compiler enforces all fields are handled.
            let Command::Ingest { src, semantic_out, layout_out } = cmd else {
                unreachable!()
            };
            let layout_map = canon::ingest::ingest_workspace(
                &canon::ingest::IngestOptions::new(src),
            )?;
            let semantic_json = serde_json::to_string_pretty(&layout_map.semantic)?;
            fs::write(&semantic_out, &semantic_json)?;
            println!("Semantic IR written to `{}`.", semantic_out.display());
            let layout_path = layout_out.unwrap_or_else(|| {
                let mut p = semantic_out.clone();
                p.set_extension("layout.json");
                p
            });
            let layout_json = serde_json::to_string_pretty(&layout_map.layout)?;
            fs::write(&layout_path, &layout_json)?;
            println!("Layout written to `{}`.", layout_path.display());
        }

        Command::ObserveEvents { .. } => {
            return Err("ObserveEvents command not yet reconnected after refactor.".into());
        }

        Command::ExecuteTick { ir, tick, parallel } => {
            let ir_doc = load_ir(&ir)?;
            validate_ir(&ir_doc)?;
            let exec = TickExecutor::new(&ir_doc);
            let mode = if parallel {
                TickExecutionMode::ParallelVerified
            } else {
                TickExecutionMode::Sequential
            };
            let result = exec.execute_tick_with_mode(&tick, mode)?;
            println!("Tick emitted {} deltas.", result.emitted_deltas.len());
        }

        Command::ExportDot { ir, layout, output } => {
            let ir_doc = load_ir(&ir)?;
            let layout_doc = load_layout(resolve_layout(layout, &ir))?;
            let dot = canon::dot_export::export_dot(&ir_doc, &layout_doc);
            fs::write(output, dot)?;
        }

        Command::VerifyDot { ir, layout, original } => {
            let ir_doc = load_ir(&ir)?;
            let layout_doc = load_layout(resolve_layout(layout, &ir))?;
            let dot = fs::read_to_string(original)?;
            canon::verify_dot(&ir_doc, &layout_doc, &dot)?;
            println!("DOT verified.");
        }
    }

    Ok(())
}
