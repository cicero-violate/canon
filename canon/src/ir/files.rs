use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct FileNode {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct FileEdge {
    pub from: String,
    pub to: String,
}
