use crate::core::collect::{add_file_module_symbol, collect_symbols};


use crate::core::paths::module_path_for_file;


use crate::core::symbol_id::normalize_symbol_id;


use crate::model::types::SymbolIndex;


use crate::state::NodeRegistry;


use anyhow::Result;


use std::collections::HashSet;


use std::path::{Path, PathBuf};
