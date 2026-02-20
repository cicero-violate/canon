//! Parallel execution and output/delta verification.

use std::collections::{BTreeMap, HashMap};

use crate::ir::FunctionId;
use crate::runtime::context::ExecutionContext;
use crate::runtime::executor::FunctionExecutor;
use crate::runtime::parallel::{execute_jobs, partition_independent_batches, ParallelJob, ParallelJobResult};
use crate::runtime::value::{DeltaValue, Value};
use crate::CanonicalIr;

use super::graph::gather_inputs;
use crate::runtime::error::RuntimeError;

pub(super) fn execute_parallel(
    ir: &CanonicalIr, execution_order: &[FunctionId], dependencies: &HashMap<FunctionId, Vec<FunctionId>>, initial_inputs: &BTreeMap<String, Value>,
) -> Result<(HashMap<FunctionId, BTreeMap<String, Value>>, Vec<DeltaValue>), RuntimeError> {
    let batches = partition_independent_batches(execution_order, dependencies);
    let mut results = HashMap::new();
    let mut deltas = Vec::new();

    for batch in batches {
        if batch.is_empty() {
            continue;
        }
        let mut jobs = Vec::new();
        for function_id in &batch {
            let inputs = gather_inputs(function_id, dependencies, &results, initial_inputs)?;
            jobs.push(ParallelJob { function: function_id.clone(), inputs });
        }

        let worker = |function_id: &FunctionId, inputs: BTreeMap<String, Value>| -> Result<ParallelJobResult, crate::runtime::executor::ExecutorError> {
            let mut local_context = ExecutionContext::new(initial_inputs.clone());
            let function_executor = FunctionExecutor::new(ir);
            let outputs = function_executor.execute_by_id(function_id, inputs, &mut local_context)?;
            Ok(ParallelJobResult { function: function_id.clone(), outputs, deltas: local_context.deltas().to_vec() })
        };

        let batch_results = execute_jobs(jobs, &worker).map_err(RuntimeError::Executor)?;
        let mut batch_delta_map: HashMap<FunctionId, Vec<DeltaValue>> = HashMap::new();
        for result in batch_results {
            batch_delta_map.insert(result.function.clone(), result.deltas);
            results.insert(result.function, result.outputs);
        }
        for function_id in &batch {
            if let Some(delta_list) = batch_delta_map.remove(function_id) {
                deltas.extend(delta_list);
            }
        }
    }

    Ok((results, deltas))
}

pub(super) fn verify_parallel_outputs(sequential: &HashMap<FunctionId, BTreeMap<String, Value>>, parallel: &HashMap<FunctionId, BTreeMap<String, Value>>) -> Result<(), RuntimeError> {
    if sequential.len() != parallel.len() {
        return Err(RuntimeError::ParallelMismatch { function: "<count mismatch>".into() });
    }
    for (function, seq_outputs) in sequential {
        match parallel.get(function) {
            Some(p_outputs) if p_outputs == seq_outputs => continue,
            Some(_) => {
                return Err(RuntimeError::ParallelMismatch { function: function.clone() });
            }
            None => {
                return Err(RuntimeError::ParallelMismatch { function: function.clone() });
            }
        }
    }
    Ok(())
}

pub(super) fn verify_parallel_deltas(sequential: &[DeltaValue], parallel: &[DeltaValue]) -> Result<(), RuntimeError> {
    if sequential.len() != parallel.len() {
        return Err(RuntimeError::ParallelDeltaMismatch { index: sequential.len().min(parallel.len()) });
    }
    for (idx, (seq, par)) in sequential.iter().zip(parallel.iter()).enumerate() {
        if seq != par {
            return Err(RuntimeError::ParallelDeltaMismatch { index: idx });
        }
    }
    Ok(())
}
