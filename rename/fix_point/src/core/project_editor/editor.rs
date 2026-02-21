pub struct ChangeReport {
    pub touched_files: Vec<PathBuf>,
    pub conflicts: Vec<EditConflict>,
    pub file_moves: Vec<(PathBuf, PathBuf)>,
}


pub struct EditConflict {
    pub symbol_id: String,
    pub reason: String,
}


pub struct ProjectEditor {
    pub registry: NodeRegistry,
    pub changesets: HashMap<PathBuf, Vec<QueuedOp>>,
    pub oracle: Box<dyn StructuralEditOracle>,
    pub original_sources: HashMap<PathBuf, String>,
    project_root: PathBuf,
    model0: Option<GraphSnapshot>,
    span_lookup: Option<SpanLookup>,
    pending_file_moves: Vec<(PathBuf, PathBuf)>,
    pending_file_renames: Vec<FileRename>,
    pending_new_files: Vec<(PathBuf, String)>,
    last_touched_files: HashSet<PathBuf>,
}


pub(crate) struct QueuedOp {
    pub symbol_id: String,
    pub op: NodeOp,
}


fn build_span_lookup_from_snapshot(snapshot: &GraphSnapshot) -> Result<SpanLookup> {
    let mut lookup: SpanLookup = HashMap::new();
    for node in snapshot.nodes.iter() {
        let metadata = &node.metadata;
        let Some(source_file) = metadata.get("source_file") else { continue };
        let def_path = metadata
            .get("def_path")
            .map(|s| s.as_str())
            .unwrap_or(node.key.as_str());
        let symbol_id = normalize_symbol_id(def_path);
        let file_path = PathBuf::from(source_file);
        let canonical = std::fs::canonicalize(&file_path).unwrap_or(file_path);
        let Some(line) = parse_i64(metadata.get("line")) else { continue };
        let Some(col) = parse_i64(metadata.get("column")) else { continue };
        let Some(end_line) = parse_i64(metadata.get("span_end_line")) else { continue };
        let Some(end_col) = parse_i64(metadata.get("span_end_column")) else { continue };
        let span = SpanRange {
            start: LineColumn {
                line,
                column: col + 1,
            },
            end: LineColumn {
                line: end_line,
                column: end_col + 1,
            },
        };
        let byte_range = match (
            metadata.get("span_start_byte"),
            metadata.get("span_end_byte"),
        ) {
            (Some(start), Some(end)) => {
                match (start.parse::<usize>(), end.parse::<usize>()) {
                    (Ok(start), Ok(end)) => Some((start, end)),
                    _ => None,
                }
            }
            _ => None,
        };
        lookup
            .entry(canonical)
            .or_default()
            .insert(symbol_id, SpanOverride { span, byte_range });
    }
    Ok(lookup)
}


fn build_symbol_index(
    project_root: &Path,
    registry: &NodeRegistry,
) -> Result<SymbolIndex> {
    let mut symbol_table = SymbolIndex::default();
    let mut symbols = Vec::new();
    let mut symbol_set = HashSet::new();
    for (file, ast) in &registry.asts {
        let module_path = normalize_symbol_id(&module_path_for_file(project_root, file));
        add_file_module_symbol(
            &module_path,
            file,
            &mut symbol_table,
            &mut symbols,
            &mut symbol_set,
        );
        let _ = collect_symbols(
            ast,
            &module_path,
            file,
            &mut symbol_table,
            &mut symbols,
            &mut symbol_set,
        );
    }
    Ok(symbol_table)
}


fn find_project_root(registry: &NodeRegistry) -> Result<Option<PathBuf>> {
    let file = match registry.asts.keys().next() {
        Some(f) => f,
        None => return Ok(None),
    };
    let mut current = file.parent().unwrap_or_else(|| Path::new("/")).to_path_buf();
    loop {
        if current.join("Cargo.toml").exists() {
            return Ok(Some(current));
        }
        if !current.pop() {
            break;
        }
    }
    Ok(None)
}


fn find_project_root_sync(registry: &NodeRegistry) -> Option<PathBuf> {
    let file = registry.asts.keys().next()?;
    let mut cur = file.parent()?.to_path_buf();
    loop {
        if cur.join("Cargo.toml").exists() {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}


fn parse_i64(value: Option<&String>) -> Option<i64> {
    value.and_then(|v| v.parse::<i64>().ok())
}
