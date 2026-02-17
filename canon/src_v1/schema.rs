use schemars::schema_for;
use serde_json::Value;

use crate::ir::CanonicalIr;

pub fn generate_schema(pretty: bool) -> serde_json::Result<String> {
    let schema = schema_for!(CanonicalIr);
    if pretty {
        serde_json::to_string_pretty(&schema)
    } else {
        serde_json::to_string(&schema)
    }
}

pub fn schema_value() -> Value {
    serde_json::to_value(schema_for!(CanonicalIr)).expect("schema serialization must succeed")
}
