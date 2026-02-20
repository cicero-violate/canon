use crate::ir::{
    StateChange, DeltaKind, ChangePayload, ExecutionEvent, ExecutionRecord, PipelineStage,
};
pub fn wrap_execution_events_as_deltas(
    execution: &ExecutionRecord,
    proof_id: &str,
) -> Vec<StateChange> {
    execution
        .events
        .iter()
        .enumerate()
        .map(|(idx, event)| StateChange {
            id: format!("{}_event_{idx}", execution.id),
            kind: DeltaKind::History,
            stage: PipelineStage::Observe,
            append_only: true,
            proof: proof_id.to_string(),
            description: describe_event(&execution.id, idx, event),
            related_function: None,
            payload: Some(ChangePayload::AttachExecutionEvent {
                execution_id: execution.id.clone(),
                event: event.clone(),
            }),
            proof_object_hash: None,
        })
        .collect()
}
fn describe_event(execution_id: &str, idx: usize, event: &ExecutionEvent) -> String {
    match event {
        ExecutionEvent::Stdout { .. } => {
            format!("Execution {execution_id} stdout #{idx}")
        }
        ExecutionEvent::Stderr { .. } => {
            format!("Execution {execution_id} stderr #{idx}")
        }
        ExecutionEvent::Artifact { path, .. } => {
            format!("Execution {execution_id} artifact `{path}`")
        }
        ExecutionEvent::Error { code, .. } => {
            format!("Execution {execution_id} error `{code}`")
        }
    }
}
