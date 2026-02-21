mod cross_file;

mod editor;

mod graph_pipeline;

mod invariants;

mod model_validation;

mod ops;

mod oracle;

mod propagate;

mod refactor;

mod registry_builder;

mod use_imports;

mod use_path;

mod utils;

pub use editor::{ChangeReport, EditConflict, ProjectEditor};


pub use oracle::NullOracle;


pub(crate) use editor::QueuedOp;
