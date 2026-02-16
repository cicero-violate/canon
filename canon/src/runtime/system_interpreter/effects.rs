use std::collections::BTreeMap;

use crate::ir::{SystemNode, SystemNodeKind};
use crate::runtime::value::{ScalarValue, StructValue, Value};
use memory_engine::Delta;

use super::{ProofArtifact, SystemExecutionEvent, SystemInterpreter, SystemInterpreterError};

impl<'a> SystemInterpreter<'a> {
    pub(super) fn apply_node_effects(
        &self,
        node: &SystemNode,
        outputs: &BTreeMap<String, Value>,
        emitted_deltas: &[Delta],
        proofs: &mut Vec<ProofArtifact>,
        delta_records: &mut Vec<super::DeltaEmission>,
        events: &mut Vec<SystemExecutionEvent>,
    ) -> Result<(), SystemInterpreterError> {
        match node.kind {
            SystemNodeKind::Function => Ok(()),
            SystemNodeKind::Gate => {
                let proof = self.extract_gate_proof(node, outputs)?;
                if !proof.accepted {
                    return Err(SystemInterpreterError::GateRejected(node.id.clone()));
                }
                events.push(SystemExecutionEvent::ProofRecorded {
                    node_id: proof.node_id.clone(),
                    proof_id: proof.proof_id,
                });
                proofs.push(proof);
                Ok(())
            }
            SystemNodeKind::Persist => {
                self.ensure_struct_output(node, outputs, "Record")?;
                if emitted_deltas.is_empty() {
                    return Err(SystemInterpreterError::PersistWithoutDelta(node.id.clone()));
                }
                delta_records.push(super::DeltaEmission {
                    node_id: node.id.clone(),
                    deltas: emitted_deltas.to_vec(),
                });
                events.push(SystemExecutionEvent::DeltaRecorded {
                    node_id: node.id.clone(),
                    count: emitted_deltas.len(),
                });
                Ok(())
            }
            SystemNodeKind::Materialize => {
                self.ensure_struct_output(node, outputs, "Artifact")?;
                Ok(())
            }
        }
    }

    fn extract_gate_proof(
        &self,
        node: &SystemNode,
        outputs: &BTreeMap<String, Value>,
    ) -> Result<ProofArtifact, SystemInterpreterError> {
        let decision = self.ensure_struct_output(node, outputs, "Decision")?;
        let accepted_value = self.expect_struct_field(node, "Decision", decision, "Accepted")?;
        let proof_id_value = self.expect_struct_field(node, "Decision", decision, "ProofId")?;

        let accepted = match accepted_value {
            Value::Scalar(ScalarValue::Bool(value)) => *value,
            other => {
                return Err(SystemInterpreterError::OutputTypeMismatch {
                    node: node.id.clone(),
                    output: "Decision.Accepted".to_string(),
                    message: format!("expected bool, found {:?}", other.kind()),
                });
            }
        };

        let proof_id = match proof_id_value {
            Value::Scalar(ScalarValue::I32(value)) => *value,
            Value::Scalar(ScalarValue::U32(value)) => *value as i32,
            other => {
                return Err(SystemInterpreterError::OutputTypeMismatch {
                    node: node.id.clone(),
                    output: "Decision.ProofId".to_string(),
                    message: format!("expected integer, found {:?}", other.kind()),
                });
            }
        };

        Ok(ProofArtifact {
            node_id: node.id.clone(),
            proof_id,
            accepted,
        })
    }

    fn ensure_struct_output<'value>(
        &self,
        node: &SystemNode,
        outputs: &'value BTreeMap<String, Value>,
        name: &str,
    ) -> Result<&'value StructValue, SystemInterpreterError> {
        let value = self.expect_output(node, outputs, name)?;
        match value {
            Value::Struct(struct_value) => Ok(struct_value),
            other => Err(SystemInterpreterError::OutputTypeMismatch {
                node: node.id.clone(),
                output: name.to_string(),
                message: format!("expected struct, found {:?}", other.kind()),
            }),
        }
    }

    fn expect_output<'value>(
        &self,
        node: &SystemNode,
        outputs: &'value BTreeMap<String, Value>,
        name: &str,
    ) -> Result<&'value Value, SystemInterpreterError> {
        outputs
            .get(name)
            .ok_or_else(|| SystemInterpreterError::MissingOutput {
                node: node.id.clone(),
                output: name.to_string(),
            })
    }

    fn expect_struct_field<'value>(
        &self,
        node: &SystemNode,
        output_name: &str,
        value: &'value StructValue,
        field: &str,
    ) -> Result<&'value Value, SystemInterpreterError> {
        value
            .fields
            .get(field)
            .ok_or_else(|| SystemInterpreterError::MissingField {
                node: node.id.clone(),
                output: output_name.to_string(),
                field: field.to_string(),
            })
    }
}
