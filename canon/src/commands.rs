use std::fs;

use crate::auto_accept_dsl_proposal;
use crate::cli::Command;
use crate::diff::diff_ir;
use crate::ingest::{self, IngestOptions};
use crate::layout::LayoutGraph;
use crate::materialize::materialize;
use crate::render_impl_function;
use crate::runtime::{TickExecutionMode, TickExecutor};
use crate::schema::generate_schema;
use crate::validate::validate_ir;
use crate::verify_dot;
use crate::version_gate::enforce_version_gate;
use crate::write_file_tree;
use crate::dot_export;
use crate::io_utils::load_ir;

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

        Command::Materialize {
            ir,
            layout,
            out_dir,
        } => {
            let ir_doc = load_ir_or_semantic(&ir)?;
            validate_ir(&ir_doc)?;
            enforce_version_gate(&ir_doc)?;
            let layout_doc = load_layout(resolve_layout(layout, &ir))?;
            let output = materialize(&ir_doc, &layout_doc, Some(&out_dir));
            write_file_tree(&output.tree, &out_dir)?;
            println!("Materialized into `{}`.", out_dir.display());
        }

        Command::SubmitDsl {
            dsl,
            ir,
            layout,
            output_ir,
            materialize_dir,
        } => {
            let ir_doc = load_ir(&ir)?;
            let layout_doc = if let Some(layout_path) = layout {
                load_layout(layout_path)?
            } else {
                LayoutGraph::default()
            };
            let dsl_source = fs::read_to_string(&dsl)?;
            let acceptance = auto_accept_dsl_proposal(&ir_doc, &layout_doc, &dsl_source)?;
            let evolved_json = serde_json::to_string_pretty(&acceptance.ir)?;
            fs::write(&output_ir, evolved_json)?;
            let output = materialize(&acceptance.ir, &acceptance.layout, Some(&materialize_dir));
            write_file_tree(&output.tree, &materialize_dir)?;
            println!(
                "DSL proposal accepted; evolved IR written to `{}`.",
                output_ir.display()
            );
        }

        Command::Ingest { .. } => {
            let Command::Ingest {
                src,
                semantic_out,
                layout_out,
            } = cmd
            else {
                unreachable!()
            };
            let layout_map = ingest::ingest_workspace(&IngestOptions::new(src))?;
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
            let mut ir_doc = load_ir(&ir)?;
            validate_ir(&ir_doc)?;
            let mut exec = TickExecutor::new(&mut ir_doc);
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
            let dot = dot_export::export_dot(&ir_doc, &layout_doc);
            fs::write(output, dot)?;
        }

        Command::VerifyDot {
            ir,
            layout,
            original,
        } => {
            let ir_doc = load_ir(&ir)?;
            let layout_doc = load_layout(resolve_layout(layout, &ir))?;
            let dot = fs::read_to_string(original)?;
            verify_dot(&ir_doc, &layout_doc, &dot)?;
            println!("DOT verified.");
        }
    }

    Ok(())
}
