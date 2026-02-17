use std::collections::BTreeMap;
use std::env;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use canon::ir::{CanonicalIr, SystemGraph};
use canon::runtime::value::ScalarValue;
use canon::runtime::{SystemExecutionEvent, SystemExecutionResult, SystemInterpreter, Value};
use hex;
use memory_engine::primitives::PageID;

use crate::shell_flow::{ShellOptions, parse_shell_mode};
use crate::state_io::{gate_and_commit, load_state, resolve_fixture_path};

#[derive(Debug)]
pub struct Intent {
    pub lhs: i32,
    pub rhs: i32,
}

pub enum CliMode {
    Execute { verify_only: bool, intent: Intent },
    Shell(ShellOptions),
}

pub fn parse_cli_mode() -> Result<CliMode> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if let Some(first) = args.first() {
        if first == "shell" {
            args.remove(0);
            let options = parse_shell_mode(args)?;
            return Ok(CliMode::Shell(options));
        }
    }

    let verify = if let Some(idx) = args.iter().position(|arg| arg == "--verify") {
        args.remove(idx);
        true
    } else {
        false
    };
    let lhs = args.get(0).and_then(|a| a.parse().ok()).unwrap_or(2);
    let rhs = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(3);
    Ok(CliMode::Execute {
        verify_only: verify,
        intent: Intent { lhs, rhs },
    })
}

pub fn run_system_flow(repo_root: &Path, verify_only: bool, intent: Intent) -> Result<()> {
    let ir_path = resolve_fixture_path(repo_root)?;
    let ir = load_ir(&ir_path)?;

    println!("Loaded fixture {}", ir_path.display());

    let mut state = load_state(repo_root)?;
    if verify_only {
        println!("Verified root hash {}", hex::encode(state.root_hash()));
        return Ok(());
    }

    println!("Intent: lhs={}, rhs={}", intent.lhs, intent.rhs);

    let config = memory_engine::MemoryEngineConfig {
        tlog_path: repo_root.join("canon_store").join("engine.wal"),
    };
    let engine = memory_engine::MemoryEngine::new(config)?;
    let interpreter = SystemInterpreter::new(&ir, &engine);
    let inputs = build_initial_inputs(&intent);
    let graph = select_system_graph(&ir)?;
    let execution = interpreter.execute_graph(&graph.id, inputs)?;
    report_execution(&execution)?;
    let deltas: Vec<memory_engine::delta::Delta> = execution
        .emitted_deltas
        .iter()
        .map(|dv| {
            memory_engine::delta::Delta::new_dense(
                memory_engine::primitives::DeltaID(
                    dv.delta_id.parse().expect("delta_id must be a valid u64"),
                ),
                PageID(0),
                memory_engine::epoch::Epoch(0),
                dv.payload_hash.clone().into(),
                vec![],
                memory_engine::delta::Source(dv.delta_id.clone()),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    gate_and_commit(repo_root, &mut state, &graph.id, &deltas, None)?;
    Ok(())
}

fn load_ir(path: &Path) -> Result<CanonicalIr> {
    let data = std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let ir = serde_json::from_slice(&data)?;
    Ok(ir)
}

fn select_system_graph<'a>(ir: &'a CanonicalIr) -> Result<&'a SystemGraph> {
    ir.system_graphs
        .first()
        .ok_or_else(|| anyhow!("no system graphs defined in Canonical IR"))
}

fn build_initial_inputs(intent: &Intent) -> BTreeMap<String, Value> {
    let mut inputs = BTreeMap::new();
    inputs.insert("Lhs".into(), Value::Scalar(ScalarValue::I32(intent.lhs)));
    inputs.insert("Rhs".into(), Value::Scalar(ScalarValue::I32(intent.rhs)));
    inputs
}

fn report_execution(result: &SystemExecutionResult) -> Result<()> {
    println!(
        "Executed system graph `{}` across {} nodes ({} deltas).",
        result.graph_id,
        result.execution_order.len(),
        result.emitted_deltas.len()
    );

    for event in &result.events {
        match event {
            SystemExecutionEvent::Validation { check, detail } => {
                println!("[validation] {check}: {detail}");
            }
            SystemExecutionEvent::NodeExecuted { node_id, kind } => {
                println!("[node] {node_id} ({kind:?}) executed");
            }
            SystemExecutionEvent::ProofRecorded { node_id, proof_id } => {
                println!("[proof-event] {node_id} proof_id={proof_id}");
            }
            SystemExecutionEvent::DeltaRecorded { node_id, count } => {
                println!("[delta-event] {node_id} emitted {count} deltas");
            }
        }
    }

    if result.proof_artifacts.is_empty() {
        println!("No proofs recorded.");
    } else {
        for proof in &result.proof_artifacts {
            println!(
                "[proof] node={} proof_id={} accepted={}",
                proof.node_id, proof.proof_id, proof.accepted
            );
        }
    }

    if result.delta_provenance.is_empty() {
        println!("No deltas recorded.");
    } else {
        for delta in &result.delta_provenance {
            let summary: Vec<String> = delta
                .deltas
                .iter()
                .map(|d| format!("delta_id={} hash={}", d.delta_id, d.payload_hash))
                .collect();
            println!("[delta] node={} {}", delta.node_id, summary.join(", "));
        }
    }

    if let Some(last_node) = result.execution_order.last() {
        if let Some(outputs) = result.node_results.get(last_node) {
            println!("Final outputs: {outputs:?}");
        }
    }

    if let Some(root) = result.state_root {
        println!("[engine] committed state root {}", hex::encode(root));
    } else {
        println!("[engine] no deltas committed via MemoryEngine");
    }

    if let Some(judgment) = &result.judgment_proof {
        println!(
            "[engine] judgment proof hash={} approved={} ts={}",
            hex::encode(judgment.hash),
            judgment.approved,
            judgment.timestamp
        );
    }

    if let Some(admission) = &result.admission_proof {
        println!(
            "[engine] admission hash={} epoch={} nonce={}",
            hex::encode(admission.judgment_proof_hash),
            admission.epoch,
            admission.nonce
        );
    }

    for (idx, commit) in result.commit_proofs.iter().enumerate() {
        println!(
            "[engine] commit[{idx}] delta_hash={} state_hash={}",
            hex::encode(commit.delta_hash),
            hex::encode(commit.state_hash)
        );
        if let Some(outcome) = result.outcome_proofs.get(idx) {
            println!(
                "           outcome success={} hash={}",
                outcome.success,
                hex::encode(outcome.commit_proof_hash)
            );
        }
        if let Some(event_hash) = result.event_hashes.get(idx) {
            println!("           event hash={}", hex::encode(event_hash));
        }
    }

    Ok(())
}
