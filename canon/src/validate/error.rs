use super::rules::CanonRule;
use std::fmt;

#[derive(Debug, Clone)]
pub struct Violation {
    rule: CanonRule,
    detail: String,
}

impl Violation {
    pub fn new(rule: CanonRule, detail: impl Into<String>) -> Self {
        Self {
            rule,
            detail: detail.into(),
        }
    }

    pub fn rule(&self) -> CanonRule {
        self.rule
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

#[derive(Debug)]
pub struct ValidationErrors {
    violations: Vec<Violation>,
}

impl ValidationErrors {
    pub fn new(violations: Vec<Violation>) -> Self {
        Self { violations }
    }

    pub fn violations(&self) -> &[Violation] {
        &self.violations
    }
}

impl fmt::Display for ValidationErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Canon validation failed with {} violation(s):",
            self.violations.len()
        )?;
        for v in &self.violations {
            writeln!(
                f,
                "- {} ({}) → {}",
                v.rule().code(),
                v.rule().text(),
                v.detail()
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationErrors {}
