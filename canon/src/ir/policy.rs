use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::ids::{PolicyParameterId, ProofId, TickEpochId};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PolicyParameters {
    pub id: PolicyParameterId,
    pub version: u32,
    pub epoch: TickEpochId,
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub entropy_weight: f64,
    pub reward_baseline: f64,
    #[serde(default)]
    pub proof_id: Option<ProofId>,
}
