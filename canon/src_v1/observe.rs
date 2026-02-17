use crate::ir::{Delta, DeltaKind, DeltaPayload, ExecutionEvent, ExecutionRecord, PipelineStage};

pub fn execution_events_to_observe_deltas(
    execution: &ExecutionRecord,
    proof_id: &str,
) -> Vec<Delta> {
    execution
        .events
        .iter()
        .enumerate()
        .map(|(idx, event)| Delta {
            id: format!("{}_event_{idx}", execution.id),
            kind: DeltaKind::History,
            stage: PipelineStage::Observe,
            append_only: true,
            proof: proof_id.to_string(),
            description: describe_event(&execution.id, idx, event),
            related_function: None,
            payload: Some(DeltaPayload::AttachExecutionEvent {
                execution_id: execution.id.clone(),
                event: event.clone(),
            }),
            proof_object_hash: None,
        })
        .collect()
}

fn describe_event(execution_id: &str, idx: usize, event: &ExecutionEvent) -> String {
    match event {
        ExecutionEvent::Stdout { .. } => format!("Execution {execution_id} stdout #{idx}"),
        ExecutionEvent::Stderr { .. } => format!("Execution {execution_id} stderr #{idx}"),
        ExecutionEvent::Artifact { path, .. } => {
            format!("Execution {execution_id} artifact `{path}`")
        }
        ExecutionEvent::Error { code, .. } => {
            format!("Execution {execution_id} error `{code}`")
        }
    }
}
