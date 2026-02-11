use blake3::hash;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofObject {
    pub id: String,
    pub expression: Expression,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Expression {
    Atom(String),
    Equals(Box<Expression>, Box<Expression>),
    And(Vec<Expression>),
    Or(Vec<Expression>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofArtifact {
    pub proof: ProofObject,
    pub derived_hash: String,
}

pub fn evaluate_proof_object(object: &ProofObject) -> ProofResult {
    let reduced = reduce(&object.expression);
    let derived_hash = hash(&serde_json::to_vec(&reduced).unwrap())
        .to_hex()
        .to_string();
    ProofResult {
        valid: reduced == Expression::Atom("true".into()),
        derived_hash,
    }
}

fn reduce(expr: &Expression) -> Expression {
    match expr {
        Expression::Atom(_) => expr.clone(),
        Expression::Equals(lhs, rhs) => {
            if evaluate_normal_form(lhs) == evaluate_normal_form(rhs) {
                Expression::Atom("true".into())
            } else {
                Expression::Atom("false".into())
            }
        }
        Expression::And(items) => {
            if items
                .iter()
                .all(|item| reduce(item) == Expression::Atom("true".into()))
            {
                Expression::Atom("true".into())
            } else {
                Expression::Atom("false".into())
            }
        }
        Expression::Or(items) => {
            if items
                .iter()
                .any(|item| reduce(item) == Expression::Atom("true".into()))
            {
                Expression::Atom("true".into())
            } else {
                Expression::Atom("false".into())
            }
        }
    }
}

fn evaluate_normal_form(expr: &Expression) -> Expression {
    match expr {
        Expression::Atom(v) => Expression::Atom(v.clone()),
        Expression::Equals(lhs, rhs) => Expression::Equals(
            Box::new(evaluate_normal_form(lhs)),
            Box::new(evaluate_normal_form(rhs)),
        ),
        Expression::And(items) => Expression::And(
            items
                .iter()
                .map(evaluate_normal_form)
                .sorted_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)))
                .collect(),
        ),
        Expression::Or(items) => Expression::Or(
            items
                .iter()
                .map(evaluate_normal_form)
                .sorted_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)))
                .collect(),
        ),
    }
}

#[derive(Debug)]
pub struct ProofResult {
    pub valid: bool,
    pub derived_hash: String,
}
