mod cross_file;
mod editor;
mod graph_pipeline;
mod model_validation;
mod oracle;
mod refactor;
mod registry_builder;
mod use_imports;
mod use_path;
mod utils;

mod invariants;
mod ops;
mod propagate;

pub use editor::{ChangeReport, EditConflict, ProjectEditor};
pub use oracle::NullOracle;

pub(crate) use editor::QueuedOp;
