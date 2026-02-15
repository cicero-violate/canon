use super::super::error::{Violation, ViolationDetail};
use super::super::rules::CanonRule;
use crate::ir::CanonicalIr;
use serde_json::Value as JsonValue;
use std::collections::HashSet;

const ALLOWED_KINDS: &[&str] = &[
    "block", "let", "if", "match", "while", "return", "call", "call_stmt",
    "lit", "bin", "cmp", "logical", "unary", "field", "index", "method",
    "struct_lit", "tuple", "array", "ref", "ref_expr", "range", "cast",
    "question", "for", "loop", "break", "continue", "assign",
    "compound_assign", "closure", "if_expr", "match_expr", "input",
];

pub fn check_ast_node_kinds(ir: &CanonicalIr, violations: &mut Vec<Violation>) {
    let allowed: HashSet<&str> = ALLOWED_KINDS.iter().copied().collect();
    for function in &ir.functions {
        let Some(ast) = &function.metadata.ast else { continue };
        validate_ast_node(ast, function.id.as_str(), &allowed, violations);
    }
}

fn validate_ast_node(
    value: &JsonValue,
    function_id: &str,
    allowed: &HashSet<&str>,
    violations: &mut Vec<Violation>,
) {
    match value {
        JsonValue::Array(items) => {
            for item in items {
                validate_ast_node(item, function_id, allowed, violations);
            }
        }
        JsonValue::Object(map) => {
            if let Some(kind) = map.get("kind").and_then(|k| k.as_str()) {
                if !allowed.contains(kind) {
                    violations.push(Violation::structured(
                        CanonRule::FunctionAst,
                        function_id.to_string(),
                        ViolationDetail::UnknownAstNodeKind {
                            function_id: function_id.to_string(),
                            kind: kind.to_string(),
                        },
                    ));
                }
            }
            for val in map.values() {
                validate_ast_node(val, function_id, allowed, violations);
            }
        }
        _ => {}
    }
}
