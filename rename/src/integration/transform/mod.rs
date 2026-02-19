//! Transformation pipeline that converts captured items into graph snapshots.

/// Cross-crate linking stage.
pub mod linker;
/// Normalization stage that builds graph snapshots.
pub mod normalizer;
/// Symbol resolution stage.
pub mod resolver;

pub use normalizer::GraphNormalizer;
