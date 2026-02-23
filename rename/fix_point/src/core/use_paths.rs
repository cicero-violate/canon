use super::structured::EditSessionTracker;


use crate::alias::AliasGraph;


use crate::fs;


use crate::structured::StructuredEditOptions;


use crate::resolve::ResolverContext;


use crate::model::types::SymbolIndex;


use std::sync::Arc;


use anyhow::{Context, Result};


use std::collections::{HashMap, HashSet};


use std::path::PathBuf;
