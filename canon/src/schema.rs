use schemars::schema_for;
use serde_json::Value;
use crate::ir::SystemState;
pub fn generate_schema(pretty: bool) -> serde_json::Result<String> {
    let schema = schema_for!(SystemState);
    if pretty {
        serde_json::to_string_pretty(&schema)
    } else {
        serde_json::to_string(&schema)
    }
}
pub fn schema_value() -> Value {
    serde_json::to_value(schema_for!(SystemState))
        .expect("schema serialization must succeed")
}
