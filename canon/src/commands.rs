use std::fs;

use crate::agent::call::AgentCallOutput;
use crate::agent::refactor::RefactorProposal;
use crate::agent::{RewardLedger, run_meta_tick, run_pipeline};
use crate::agent::runner::{RunnerConfig, run_agent};
use crate::cli::Command;
use crate::decision::auto_dsl::auto_accept_dsl_proposal;
use crate::diff::diff_ir;
use crate::dot_export::{self, verify_dot};
use crate::ingest::{self, IngestOptions};
use crate::io_utils::{
    load_capability_graph, load_ir, load_ir_or_semantic, load_layout, resolve_layout,
    save_capability_graph,
};
use crate::layout::LayoutGraph;
use crate::materialize::{materialize, write_file_tree};
use crate::materialize::render_fn::render_impl_function;
use crate::runtime::{TickExecutionMode, TickExecutor};
use crate::schema::generate_schema;
use crate::validate::validate_ir;
use crate::version_gate::enforce_version_gate;

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
            let output = materialize(&ir_doc, &layout_doc, Some(&out_dir));
            write_file_tree(&output.tree, &out_dir)?;
            println!("Materialized into `{}`.", out_dir.display());
        }

        Command::Ingest { src, semantic_out, layout_out } => {
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
            return Err("ObserveEvents not yet reconnected after refactor.".into());
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

        Command::VerifyDot { ir, layout, original } => {
            let ir_doc = load_ir(&ir)?;
            let layout_doc = load_layout(resolve_layout(layout, &ir))?;
            let dot = fs::read_to_string(original)?;
            verify_dot(&ir_doc, &layout_doc, &dot)?;
            println!("DOT verified.");
        }

        Command::SubmitDsl { dsl, ir, layout, output_ir, materialize_dir } => {
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

        // --- Agent layer commands (Item 3) ---

        Command::RunPipeline { ir, layout, outputs, proposal, output_ir } => {
            let ir_doc = load_ir(&ir)?;
            let layout_path = resolve_layout(layout, &ir);
            let layout_doc = load_layout(layout_path)?;
            let stage_outputs: Vec<AgentCallOutput> =
                serde_json::from_slice(&fs::read(&outputs)?)?;
            let prop: RefactorProposal =
                serde_json::from_slice(&fs::read(&proposal)?)?;

            match run_pipeline(&ir_doc, &layout_doc, prop, &stage_outputs) {
                Ok(result) => {
                    let json = serde_json::to_string_pretty(&result.ir)?;
                    fs::write(&output_ir, json)?;
                    println!(
                        "Pipeline OK — reward={:.4}  admission={}",
                        result.reward, result.admission_id
                    );
                    println!("Mutated IR written to: {}", output_ir.display());
                }
                Err(e) => {
                    eprintln!("Pipeline failed: {e}");
                    std::process::exit(1);
                }
            }
        }

        Command::MetaTick { graph, ledger, output_graph } => {
            let cap_graph = load_capability_graph(&graph)?;
            let ledger_doc: RewardLedger =
                serde_json::from_slice(&fs::read(&ledger)?)?;

            match run_meta_tick(&cap_graph, &ledger_doc) {
                Ok(result) => {
                    println!("Meta-tick OK");
                    println!(
                        "  entropy: {:.4} → {:.4}",
                        result.entropy_before, result.entropy_after
                    );
                    println!("  applied mutations : {}", result.applied.len());
                    println!("  rejected mutations: {}", result.rejected.len());
                    for m in &result.applied {
                        println!("    + {m:?}");
                    }
                    for (m, e) in &result.rejected {
                        println!("    ✗ {m:?}  reason: {e}");
                    }
                    save_capability_graph(&result.graph, &output_graph)?;
                    println!("Graph written to: {}", output_graph.display());
                }
                Err(e) => {
                    eprintln!("Meta-tick failed: {e}");
                    std::process::exit(1);
                }
            }
        }

        Command::ShowLedger { ledger } => {
            let ledger_doc: RewardLedger =
                serde_json::from_slice(&fs::read(&ledger)?)?;
            let ranked = ledger_doc.ranked_nodes();
            if ranked.is_empty() {
                println!("Ledger is empty — no pipeline runs recorded.");
            } else {
                println!("{:<30} {:>10} {:>10}", "node_id", "ema_reward", "run_count");
                println!("{}", "-".repeat(54));
                for entry in ranked {
                    println!(
                        "{:<30} {:>10.4} {:>10}",
                        entry.node_id, entry.ema_reward, entry.run_count
                    );
                }
                println!("aggregate reward: {:.4}", ledger_doc.aggregate_reward());
            }
        }

        Command::ShowGraph { graph } => {
            let cap_graph = load_capability_graph(&graph)?;
            println!(
                "Capability graph: {} nodes, {} edges",
                cap_graph.nodes.len(),
                cap_graph.edges.len()
            );
            println!("Entropy H(G) = {:.4}", cap_graph.entropy());
            println!();
            println!("{:<20} {:<12} {}", "id", "kind", "label");
            println!("{}", "-".repeat(60));
            for node in &cap_graph.nodes {
                println!(
                    "{:<20} {:<12} {}",
                    node.id,
                    format!("{:?}", node.kind),
                    node.label
                );
            }
            if !cap_graph.edges.is_empty() {
                println!();
                println!("{:<20} → {:<20} {:>8}", "from", "to", "conf");
                println!("{}", "-".repeat(54));
                for edge in &cap_graph.edges {
                    println!(
                        "{:<20} → {:<20} {:>8.3}",
                        edge.from, edge.to, edge.proof_confidence
                    );
                }
            }
        }

        Command::RunAgent {
            ir,
            layout,
            graph,
            proposal,
            ir_out,
            ledger_out,
            graph_out,
            max_ticks,
            meta_tick_interval,
            policy_update_interval,
        } => {
            let mut ir_doc = load_ir(&ir)?;
            let layout_path = resolve_layout(layout, &ir);
            let mut layout_doc = load_layout(layout_path)?;
            let mut cap_graph = load_capability_graph(&graph)?;
            let seed_proposal: RefactorProposal =
                serde_json::from_slice(&fs::read(&proposal)?)?;

            let mut config = RunnerConfig::new(graph_out, ledger_out, ir_out);
            config.max_ticks = max_ticks;
            config.meta_tick_interval = meta_tick_interval;
            config.policy_update_interval = policy_update_interval;

            let stats = run_agent(
                &mut ir_doc,
                &mut layout_doc,
                &mut cap_graph,
                seed_proposal,
                &config,
            )?;

            println!("Agent run complete — {} ticks", stats.len());
            println!("{:<6} {:>8} {:>8} {:>10} {:>6} {:>6}",
                "tick", "nodes", "errors", "reward", "meta", "policy");
            println!("{}", "-".repeat(52));
            for s in &stats {
                println!(
                    "{:<6} {:>8} {:>8} {:>10} {:>6} {:>6}",
                    s.tick_number,
                    s.nodes_called,
                    s.llm_errors,
                    s.pipeline_reward
                        .map(|r| format!("{r:.4}"))
                        .unwrap_or_else(|| "-".to_string()),
                    if s.meta_tick_fired { "Y" } else { "-" },
                    if s.policy_updated { "Y" } else { "-" },
                );
            }
        }
    }
    Ok(())
}
