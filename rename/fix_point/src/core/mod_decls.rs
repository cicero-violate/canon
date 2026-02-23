use anyhow::{Context, Result};


use std::collections::{HashMap, HashSet};


use std::path::{Path, PathBuf};


use crate::fs;


use super::paths::{module_child_path, module_path_for_file};


use crate::model::types::{FileRename, SymbolIndex};


#[derive(Debug)]
struct ModuleRenamePlan {
    old_name: String,
    new_name: String,
    old_parent: String,
    new_parent: String,
}
