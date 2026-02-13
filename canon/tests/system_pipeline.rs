use canon::runtime::ast::FunctionAst;
use canon::runtime::value::Value;
use serde_json::{Value as JsonValue, json};

#[test]
fn parse_system_pipeline_ast() {
    let fixture_path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/system_pipeline.json");
    let data = std::fs::read_to_string(fixture_path).unwrap();
    let ir: serde_json::Value = serde_json::from_str(&data).unwrap();
    let functions = ir.get("functions").and_then(JsonValue::as_array).unwrap();
    for function in functions {
        let function_id = function.get("id").and_then(JsonValue::as_str).unwrap();
        if function
            .get("metadata")
            .and_then(|m| m.get("ast"))
            .is_some()
        {
            let ast_value = function
                .get("metadata")
                .unwrap()
                .get("ast")
                .unwrap()
                .clone();
            serde_json::from_value::<FunctionAst>(ast_value.clone()).unwrap_or_else(|err| {
                panic!(
                    "failed to parse AST for {}: {}. json={}",
                    function_id, err, ast_value
                );
            });
        }
    }

    let literal = json!({"Scalar": {"I32": 1}});
    serde_json::from_value::<Value>(literal).unwrap();
}
