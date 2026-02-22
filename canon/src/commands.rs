use crate::agent::bootstrap::{seed_capability_graph, seed_refactor_proposal};
use crate::agent::call::AgentCallOutput;
use crate::agent::io::{load_capability_graph, save_capability_graph};
use crate::agent::observe::{analyze_ir, ir_observation_to_json};
use crate::agent::refactor::RefactorProposal;
use crate::agent::runner::{run_agent, RunnerConfig};
use crate::agent::ws_server;
use crate::agent::{evolve_capability_graph, run_refactor_pipeline, NodeRewardLedger};
use crate::cli::Command;
use crate::decision::auto_dsl::apply_dsl_proposal;
use crate::diagnose::trace_root_causes;
use crate::diff::diff_ir;
use crate::dot_export::{self, verify_dot};
use crate::gpu::codegen::generate_shader;
use crate::gpu::dispatch::GpuExecutor;
use crate::gpu::fusion::analyze_fusion_candidates;
use crate::ingest::{self, IngestOptions};
use crate::io_utils::{
    load_ir_from_file, load_ir_or_semantic_graph, load_layout, resolve_layout,
};
use crate::layout::FileTopology;
use crate::materialize::render_fn::render_impl_function;
use crate::materialize::{materialize, write_file_tree};
use crate::runtime::bytecode::FunctionBytecode;
use crate::runtime::system_interpreter::SystemInterpreter;
use crate::runtime::{TickExecutionMode, ExecutionEngine};
use crate::schema::generate_schema;
use crate::semantic_builder::SemanticIrBuilder;
use crate::storage::builder::StateWriter;
use crate::validate::validate_ir;
use crate::version_gate::enforce_version_gate;
use kernel::kernel::{Kernel as MemoryEngine, MemoryEngineConfig};
use std::fs;
pub async fn execute_command(cmd: Command) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        Command::Schema { pretty } => {
            println!("{}", generate_schema(pretty) ?);
        }
        Command::Validate { path } => {
            let ir = load_ir_from_file(&path)?;
            validate_ir(&ir)?;
            enforce_version_gate(&ir)?;
            println!("Validation passed: `{}`.", path.display());
        }
        Command::Lint { ir } => {
            let ir_doc = load_ir_from_file(&ir)?;
            validate_ir(&ir_doc)?;
            println!("Lint passed.");
        }
        Command::Diagnose { ir } => {
            let ir_doc = load_ir_or_semantic_graph(&ir)?;
            match validate_ir(&ir_doc) {
                Ok(()) => {
                    println!("No violations. IR is structurally sound.");
                }
                Err(errors) => {
                    let briefs = trace_root_causes(&errors, &ir_doc);
                    println!(
                        "{} violation(s) → {} root cause(s):\n", errors.violations()
                        .len(), briefs.len()
                    );
                    for (i, b) in briefs.iter().enumerate() {
                        println!(
                            "── Root Cause {} ─────────────────────────────",
                            i + 1
                        );
                        println!("Rule:      {:?}", b.rule);
                        println!("Count:     {} violation(s)", b.violation_count);
                        println!("Class:     {}", b.defect_class);
                        println!("IR Field:  {}", b.ir_field);
                        println!("Fix Site:  {}", b.fix_site);
                        println!("Examples:");
                        for ex in &b.examples {
                            println!("  - {}", ex);
                        }
                        println!("Brief:\n  {}\n", b.brief);
                        if let Some(report) = &b.structured_report {
                            println!("{}", report);
                            println!();
                        }
                    }
                }
            }
        }
        Command::RenderFn { ir, fn_id } => {
            let ir_doc = load_ir_from_file(&ir)?;
            let function = ir_doc
                .functions
                .iter()
                .find(|f| f.id == fn_id)
                .ok_or("unknown function")?;
            println!("{}", render_impl_function(function));
        }
        Command::DiffIr { before, after } => {
            let a = load_ir_from_file(&before)?;
            let b = load_ir_from_file(&after)?;
            println!("{}", diff_ir(& a, & b));
        }
        Command::Materialize { ir, layout, out_dir } => {
            let ir_doc = load_ir_or_semantic_graph(&ir)?;
            validate_ir(&ir_doc)?;
            enforce_version_gate(&ir_doc)?;
            let layout_doc = load_layout(resolve_layout(layout, &ir))?;
            let output = materialize(&ir_doc, &layout_doc, Some(&out_dir));
            write_file_tree(&output.tree, &out_dir)?;
            println!("Materialized into `{}`.", out_dir.display());
        }
        Command::Ingest { src, semantic_out, layout_out } => {
            let layout_map = ingest::ingest_workspace(&IngestOptions::new(src.clone()))?;
            if let Some(parent) = semantic_out.parent() {
                fs::create_dir_all(parent)?;
            }
            let semantic_json = serde_json::to_string_pretty(&layout_map.semantic)?;
            fs::write(&semantic_out, &semantic_json)?;
            println!("Semantic IR written to `{}`.", semantic_out.display());
            let layout_path = layout_out
                .unwrap_or_else(|| {
                    let mut p = semantic_out.clone();
                    p.set_extension("layout.json");
                    p
                });
            let layout_json = serde_json::to_string_pretty(&layout_map.layout)?;
            fs::write(&layout_path, &layout_json)?;
            println!("Layout written to `{}`.", layout_path.display());
            let workspace_name = src
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("workspace");
            let semantic_ir = SemanticIrBuilder::new(workspace_name)
                .build(layout_map.semantic);
            let mut tlog_path = semantic_out.clone();
            tlog_path.set_extension("tlog");
            if let Some(parent) = tlog_path.parent() {
                fs::create_dir_all(parent)?;
            }
            let engine = MemoryEngine::new(MemoryEngineConfig {
                tlog_path: tlog_path.clone(),
            })?;
            let builder = StateWriter::new(&engine);
            builder.write_ir_to_disk(&semantic_ir)?;
            let mut checkpoint_path = semantic_out.clone();
            checkpoint_path.set_extension("bin");
            engine.checkpoint(&checkpoint_path)?;
            println!(
                "MemoryEngine snapshot written to `{}`.", checkpoint_path.display()
            );
        }
        Command::ObserveEvents { ir, output } => {
            let ir_doc = load_ir_from_file(&ir)?;
            let obs = analyze_ir(&ir_doc);
            let payload = ir_observation_to_json(&obs);
            let json = serde_json::to_string_pretty(&payload)?;
            fs::write(&output, &json)?;
            println!(
                "Observation written to `{}` ({} modules, {} functions, {} call edges).",
                output.display(), obs.totals.modules, obs.totals.functions, obs.totals
                .call_edges,
            );
        }
        Command::ExecuteTick { ir, tick, parallel } => {
            let mut ir_doc = load_ir_from_file(&ir)?;
            validate_ir(&ir_doc)?;
            let mut exec = ExecutionEngine::new(&mut ir_doc);
            let mode = if parallel {
                TickExecutionMode::ParallelVerified
            } else {
                TickExecutionMode::Sequential
            };
            let result = exec.execute_tick_with_mode(&tick, mode)?;
            println!("Tick emitted {} deltas.", result.emitted_deltas.len());
        }
        Command::ExportDot { ir, layout, output } => {
            let ir_doc = load_ir_from_file(&ir)?;
            let layout_doc = load_layout(resolve_layout(layout, &ir))?;
            let dot = dot_export::export_dot(&ir_doc, &layout_doc);
            fs::write(output, dot)?;
        }
        Command::VerifyDot { ir, layout, original } => {
            let ir_doc = load_ir_from_file(&ir)?;
            let layout_doc = load_layout(resolve_layout(layout, &ir))?;
            let dot = fs::read_to_string(original)?;
            verify_dot(&ir_doc, &layout_doc, &dot)?;
            println!("DOT verified.");
        }
        Command::SubmitDsl { dsl, ir, layout, output_ir, materialize_dir } => {
            let ir_doc = load_ir_from_file(&ir)?;
            let layout_doc = if let Some(layout_path) = layout {
                load_layout(layout_path)?
            } else {
                FileTopology::default()
            };
            let dsl_source = fs::read_to_string(&dsl)?;
            let acceptance = apply_dsl_proposal(&ir_doc, &layout_doc, &dsl_source)?;
            let evolved_json = serde_json::to_string_pretty(&acceptance.ir)?;
            fs::write(&output_ir, evolved_json)?;
            let output = materialize(
                &acceptance.ir,
                &acceptance.layout,
                Some(&materialize_dir),
            );
            write_file_tree(&output.tree, &materialize_dir)?;
            println!(
                "DSL proposal accepted; evolved IR written to `{}`.", output_ir.display()
            );
        }
        Command::RunPipeline { ir, layout, outputs, proposal, output_ir } => {
            let ir_doc = load_ir_from_file(&ir)?;
            let layout_path = resolve_layout(layout, &ir);
            let layout_doc = load_layout(layout_path)?;
            let stage_outputs: Vec<AgentCallOutput> = serde_json::from_slice(
                &fs::read(&outputs)?,
            )?;
            let prop: RefactorProposal = serde_json::from_slice(&fs::read(&proposal)?)?;
            match run_refactor_pipeline(&ir_doc, &layout_doc, prop, &stage_outputs) {
                Ok(result) => {
                    let json = serde_json::to_string_pretty(&result.ir)?;
                    fs::write(&output_ir, json)?;
                    println!(
                        "Pipeline OK — reward={:.4}  admission={}", result.reward,
                        result.admission_id
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
            let ledger_doc: NodeRewardLedger = serde_json::from_slice(
                &fs::read(&ledger)?,
            )?;
            match evolve_capability_graph(&cap_graph, &ledger_doc) {
                Ok(result) => {
                    println!("Meta-tick OK");
                    println!(
                        "  entropy: {:.4} → {:.4}", result.entropy_before, result
                        .entropy_after
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
            let ledger_doc: NodeRewardLedger = serde_json::from_slice(
                &fs::read(&ledger)?,
            )?;
            let ranked = ledger_doc.ranked_nodes();
            if ranked.is_empty() {
                println!("Ledger is empty — no pipeline runs recorded.");
            } else {
                println!("{:<30} {:>10} {:>10}", "node_id", "ema_reward", "run_count");
                println!("{}", "-".repeat(54));
                for entry in ranked {
                    println!(
                        "{:<30} {:>10.4} {:>10}", entry.node_id, entry.ema_reward, entry
                        .run_count
                    );
                }
                println!("aggregate reward: {:.4}", ledger_doc.aggregate_reward());
            }
        }
        Command::ShowGraph { graph } => {
            let cap_graph = load_capability_graph(&graph)?;
            println!(
                "Capability graph: {} nodes, {} edges", cap_graph.nodes.len(), cap_graph
                .edges.len()
            );
            println!("Entropy H(G) = {:.4}", cap_graph.entropy());
            println!();
            println!("{:<20} {:<12} {}", "id", "kind", "label");
            println!("{}", "-".repeat(60));
            for node in &cap_graph.nodes {
                println!(
                    "{:<20} {:<12} {}", node.id, format!("{:?}", node.kind), node.label
                );
            }
            if !cap_graph.edges.is_empty() {
                println!();
                println!("{:<20} → {:<20} {:>8}", "from", "to", "conf");
                println!("{}", "-".repeat(54));
                for edge in &cap_graph.edges {
                    println!(
                        "{:<20} → {:<20} {:>8.3}", edge.from, edge.to, edge
                        .proof_confidence
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
            let mut ir_doc = load_ir_or_semantic_graph(&ir)?;
            let layout_path = resolve_layout(layout, &ir);
            let mut layout_doc = load_layout(layout_path)?;
            let mut cap_graph = load_capability_graph(&graph)?;
            let seed_proposal: RefactorProposal = serde_json::from_slice(
                &fs::read(&proposal)?,
            )?;
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
                    &ws_server::spawn("127.0.0.1:8787".parse()?),
                )
                .await?;
            println!("Agent run complete — {} ticks", stats.len());
            println!(
                "{:<6} {:>8} {:>8} {:>10} {:>6} {:>6}", "tick", "nodes", "errors",
                "reward", "meta", "policy"
            );
            println!("{}", "-".repeat(52));
            for s in &stats {
                println!(
                    "{:<6} {:>8} {:>8} {:>10} {:>6} {:>6}", s.tick_number, s
                    .nodes_called, s.llm_errors, s.pipeline_reward.map(| r |
                    format!("{r:.4}")).unwrap_or_else(|| "-".to_string()), if s
                    .meta_tick_fired { "Y" } else { "-" }, if s.policy_updated { "Y" }
                    else { "-" },
                );
            }
        }
        Command::GpuAnalyze { ir, execute } => {
            let ir_doc = load_ir_from_file(&ir)?;
            let candidates = analyze_fusion_candidates(&ir_doc);
            if candidates.is_empty() {
                println!("No GPU fusion candidates found.");
            } else {
                println!("{} fusion candidate(s):", candidates.len());
                for c in &candidates {
                    println!(
                        "  {} → {}  (fn: {} → {})", c.producer_gpu, c.consumer_gpu, c
                        .producer_function, c.consumer_function,
                    );
                }
                if execute {
                    let first = &candidates[0];
                    let gpu_fn = ir_doc
                        .gpu_functions
                        .iter()
                        .find(|g| g.id == first.producer_gpu)
                        .ok_or("producer gpu function not found")?;
                    let canon_fn = ir_doc
                        .functions
                        .iter()
                        .find(|f| f.id == gpu_fn.function)
                        .ok_or("backing function not found")?;
                    let bytecode = FunctionBytecode::from_function(canon_fn)?;
                    let program = generate_shader(gpu_fn, &bytecode)
                        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
                    println!(
                        "Shader generated ({} lanes). Attempting GPU execution...",
                        program.lanes
                    );
                    let executor = GpuExecutor::new().await?;
                    let inputs = vec![
                        0f32; gpu_fn.inputs.iter().map(| p | p.lanes as usize).sum()
                    ];
                    let mut outputs = vec![
                        0f32; gpu_fn.outputs.iter().map(| p | p.lanes as usize).sum()
                    ];
                    executor.execute(&program, &inputs, &mut outputs).await?;
                    println!("GPU execution OK. Output[0] = {:?}", outputs.first());
                }
            }
        }
        Command::ExecuteGraph { ir, graph_id, checkpoint } => {
            let ir_doc = load_ir_from_file(&ir)?;
            let engine = MemoryEngine::new(MemoryEngineConfig {
                tlog_path: checkpoint.with_extension("tlog"),
            })?;
            let interp = SystemInterpreter::new(&ir_doc, &engine);
            let result = interp.execute_graph(&graph_id, Default::default())?;
            println!(
                "ExecuteGraph OK — graph={} nodes_executed={} deltas_emitted={}",
                graph_id, result.node_results.len(), result.emitted_deltas.len(),
            );
        }
        Command::BootstrapGraph { graph_out, proposal_out, ir } => {
            let ir_doc = load_ir_or_semantic_graph(&ir)?;
            let target_id = ir_doc
                .modules
                .first()
                .map(|m| m.id.as_str())
                .unwrap_or("module_0")
                .to_string();
            let graph = seed_capability_graph();
            let proposal = seed_refactor_proposal(&target_id);
            let violations = graph.validate_edges();
            if !violations.is_empty() {
                for v in &violations {
                    eprintln!("graph violation: {v}");
                }
                return Err("bootstrap graph has edge violations".into());
            }
            save_capability_graph(&graph, &graph_out)?;
            let proposal_json = serde_json::to_string_pretty(&proposal)?;
            fs::write(&proposal_out, proposal_json)?;
            println!("Bootstrap complete.");
            println!(
                "  graph:    {} nodes, {} edges → {}", graph.nodes.len(), graph.edges
                .len(), graph_out.display()
            );
            println!(
                "  proposal: target={} kind={:?} → {}", target_id, crate
                ::agent::refactor::RefactorKind::SplitModule, proposal_out.display()
            );
            println!();
            println!("To start the agent loop:");
            println!(
                "  canon run-agent --ir <IR> --graph {} --proposal {} \
                 --ir-out out.ir.json --ledger-out ledger.json \
                 --graph-out {} --max-ticks 10",
                graph_out.display(), proposal_out.display(), graph_out.display(),
            );
        }
    }
    Ok(())
}
