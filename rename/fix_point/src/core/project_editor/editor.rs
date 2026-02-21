#[derive(Debug, Clone)]
pub struct EditConflict {
    pub symbol_id: String,
    pub reason: String,
}


#[derive(Debug, Clone)]
pub struct EditConflict {
    pub symbol_id: String,
    pub reason: String,
}


#[derive(Debug, Clone)]
pub struct ChangeReport {
    pub touched_files: Vec<PathBuf>,
    pub conflicts: Vec<EditConflict>,
    pub file_moves: Vec<(PathBuf, PathBuf)>,
}


#[derive(Debug, Clone)]
pub struct ChangeReport {
    pub touched_files: Vec<PathBuf>,
    pub conflicts: Vec<EditConflict>,
    pub file_moves: Vec<(PathBuf, PathBuf)>,
}


#[derive(Clone)]
pub(crate) struct QueuedOp {
    pub symbol_id: String,
    pub op: NodeOp,
}


#[derive(Clone)]
pub(crate) struct QueuedOp {
    pub symbol_id: String,
    pub op: NodeOp,
}


#[derive(Debug, Clone)]
pub struct EditConflict {
    pub symbol_id: String,
    pub reason: String,
}


#[derive(Debug, Clone)]
pub struct EditConflict {
    pub symbol_id: String,
    pub reason: String,
}


#[derive(Debug, Clone)]
pub struct ChangeReport {
    pub touched_files: Vec<PathBuf>,
    pub conflicts: Vec<EditConflict>,
    pub file_moves: Vec<(PathBuf, PathBuf)>,
}


#[derive(Debug, Clone)]
pub struct ChangeReport {
    pub touched_files: Vec<PathBuf>,
    pub conflicts: Vec<EditConflict>,
    pub file_moves: Vec<(PathBuf, PathBuf)>,
}


#[derive(Clone)]
pub(crate) struct QueuedOp {
    pub symbol_id: String,
    pub op: NodeOp,
}


#[derive(Clone)]
pub(crate) struct QueuedOp {
    pub symbol_id: String,
    pub op: NodeOp,
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


impl ProjectEditor {
    pub fn load(project: &Path, oracle: Box<dyn StructuralEditOracle>) -> Result<Self> {
        let files = fs::collect_rs_files(project)?;
        let mut registry = NodeRegistry::new();
        let mut original_sources = HashMap::new();
        for file in files {
            let content = std::fs::read_to_string(&file)?;
            let source = Arc::new(content.clone());
            let ast = syn::parse_file(&content)
                .with_context(|| format!("Failed to parse {}", file.display()))?;
            let mut builder = NodeRegistryBuilder::new(
                project,
                &file,
                &mut registry,
                source.clone(),
                None,
            );
            builder.visit_file(&ast);
            registry.insert_ast(file.clone(), ast);
            registry.insert_source(file.clone(), source);
            original_sources.insert(file, content);
        }
        Ok(Self {
            registry,
            changesets: HashMap::new(),
            oracle,
            original_sources,
            project_root: project.to_path_buf(),
            model0: None,
            span_lookup: None,
            pending_file_moves: Vec::new(),
            pending_file_renames: Vec::new(),
            pending_new_files: Vec::new(),
            last_touched_files: HashSet::new(),
        })
    }
    pub fn load_with_rustc(project: &Path) -> Result<Self> {
        let cargo = CargoProject::from_entry(project)?;
        let frontend = RustcFrontend::new();
        let _artifacts = capture_project(&frontend, &cargo, &[])
            .with_context(|| format!("rustc capture failed for {}", project.display()))?;
        let workspace_root = cargo
            .metadata()
            .map(|m| m.workspace_root)
            .unwrap_or_else(|_| cargo.workspace_root().to_path_buf());
        if std::env::var("RENAME_DEBUG_PLAN").ok().as_deref() == Some("1") {
            eprintln!("[plan] workspace_root={}", workspace_root.display());
        }
        let state_dir = workspace_root.join(".rename");
        std::fs::create_dir_all(&state_dir)?;
        let tlog_path = state_dir.join("state.tlog");
        let engine = MemoryEngine::new(MemoryEngineConfig { tlog_path })?;
        let mut snapshot = engine.materialized_graph()?;
        validate_model0(&mut snapshot)?;
        let span_lookup = build_span_lookup_from_snapshot(&snapshot)?;
        let model0 = snapshot.clone();
        let oracle = Box::new(GraphSnapshotOracle::from_snapshot(snapshot));
        Self::load_with_span_lookup(project, oracle, span_lookup, Some(model0))
    }
    fn load_with_span_lookup(
        project: &Path,
        oracle: Box<dyn StructuralEditOracle>,
        span_lookup: SpanLookup,
        model0: Option<GraphSnapshot>,
    ) -> Result<Self> {
        let files = fs::collect_rs_files(project)?;
        let mut registry = NodeRegistry::new();
        let mut original_sources = HashMap::new();
        for file in files {
            let content = std::fs::read_to_string(&file)?;
            let source = Arc::new(content.clone());
            let ast = syn::parse_file(&content)
                .with_context(|| format!("Failed to parse {}", file.display()))?;
            let mut builder = NodeRegistryBuilder::new(
                project,
                &file,
                &mut registry,
                source.clone(),
                Some(&span_lookup),
            );
            builder.visit_file(&ast);
            registry.insert_ast(file.clone(), ast);
            registry.insert_source(file.clone(), source);
            original_sources.insert(file, content);
        }
        Ok(Self {
            registry,
            changesets: HashMap::new(),
            oracle,
            original_sources,
            project_root: project.to_path_buf(),
            model0,
            span_lookup: Some(span_lookup),
            pending_file_moves: Vec::new(),
            pending_file_renames: Vec::new(),
            pending_new_files: Vec::new(),
            last_touched_files: HashSet::new(),
        })
    }
    pub fn queue(&mut self, symbol_id: &str, op: NodeOp) -> Result<()> {
        let norm = normalize_symbol_id(symbol_id);
        let handle = match &op {
            NodeOp::ReplaceNode { handle, .. }
            | NodeOp::InsertBefore { handle, .. }
            | NodeOp::InsertAfter { handle, .. }
            | NodeOp::DeleteNode { handle }
            | NodeOp::MutateField { handle, .. }
            | NodeOp::MoveSymbol { handle, .. } => Some(handle),
            NodeOp::ReorderItems { .. } => None,
        };
        if let Some(handle) = handle {
            let exists = self.registry.handles.get(&norm);
            if exists.is_none() {
                self.registry.insert_handle(norm.clone(), handle.clone());
            }
        }
        let file = match &op {
            NodeOp::ReplaceNode { handle, .. }
            | NodeOp::InsertBefore { handle, .. }
            | NodeOp::InsertAfter { handle, .. }
            | NodeOp::DeleteNode { handle }
            | NodeOp::MutateField { handle, .. } => handle.file.clone(),
            NodeOp::ReorderItems { file, .. } => file.clone(),
            NodeOp::MoveSymbol { handle, .. } => handle.file.clone(),
        };
        self.changesets.entry(file).or_default().push(QueuedOp { symbol_id: norm, op });
        Ok(())
    }
    pub fn queue_by_id(
        &mut self,
        symbol_id: &str,
        mutation: FieldMutation,
    ) -> Result<()> {
        let norm = normalize_symbol_id(symbol_id);
        let handle = self
            .registry
            .handles
            .get(&norm)
            .cloned()
            .with_context(|| format!("no handle found for {symbol_id}"))?;
        let op = NodeOp::MutateField {
            handle,
            mutation,
        };
        self.queue(&norm, op)
    }
    pub fn apply(&mut self) -> Result<ChangeReport> {
        let only_moves = self
            .changesets
            .values()
            .all(|ops| ops.iter().all(|q| matches!(q.op, NodeOp::MoveSymbol { .. })));
        if only_moves {
            if let Some(model0) = self.model0.clone() {
                let moveset = self.build_moveset()?;
                let mut model1 = model0.clone();
                apply_moves_to_snapshot(&mut model1, &moveset)?;
                let plan = project_plan(&model1, &self.project_root)?;
                let report = emit_plan(
                    &mut self.registry,
                    plan,
                    &self.project_root,
                    false,
                )?;
                self.rebuild_registry_from_sources()?;
                let model2 = rebuild_graph_snapshot(&self.project_root)?;
                if let Err(err) = compare_snapshots(&model1, &model2) {
                    rollback_emission(&self.project_root, &report.written)?;
                    return Err(err);
                }
                self.model0 = Some(model2.clone());
                self.oracle = Box::new(GraphSnapshotOracle::from_snapshot(model2));
                let touched: HashSet<PathBuf> = report.written.iter().cloned().collect();
                self.last_touched_files = touched.clone();
                return Ok(ChangeReport {
                    touched_files: touched.into_iter().collect(),
                    conflicts: Vec::new(),
                    file_moves: Vec::new(),
                });
            }
        }
        let mut touched_files: HashSet<PathBuf> = HashSet::new();
        let mut rewrites = Vec::new();
        let mut conflicts = Vec::new();
        let mut file_renames = Vec::new();
        let moveset = self.build_moveset()?;
        let handle_snapshot = self.registry.handles.clone();
        for (_file, ops) in &self.changesets {
            for queued in ops {
                let prop = propagate(
                    &queued.op,
                    &queued.symbol_id,
                    &self.registry,
                    &*self.oracle,
                )?;
                rewrites.extend(prop.rewrites);
                conflicts.extend(prop.conflicts);
                file_renames.extend(prop.file_renames);
            }
        }
        let rewrite_touched = apply_rewrites(&mut self.registry, &rewrites)?;
        touched_files.extend(rewrite_touched);
        for (file, ops) in &self.changesets {
            for queued in ops {
                let changed = {
                    let ast = self
                        .registry
                        .asts
                        .get_mut(file)
                        .with_context(|| format!("missing AST for {}", file.display()))?;
                    let content = self
                        .registry
                        .sources
                        .get(file)
                        .map(|s| s.as_str())
                        .unwrap_or("");
                    apply_node_op(
                            ast,
                            content,
                            &handle_snapshot,
                            &queued.symbol_id,
                            &queued.op,
                        )
                        .with_context(|| {
                            format!("failed to apply {}", queued.symbol_id)
                        })?
                };
                if changed {
                    touched_files.insert(file.clone());
                }
            }
        }
        let ast_touched = touched_files.clone();
        let cross_file_touched = apply_cross_file_moves(
            &mut self.registry,
            &self.changesets,
        )?;
        touched_files.extend(cross_file_touched.clone());
        self.refresh_sources_from_asts(&ast_touched, &cross_file_touched)?;
        self.rebuild_registry_from_sources()?;
        self.run_refactor_pipeline(&moveset)?;
        let _ = build_symbol_index_and_occurrences(&self.registry)?;
        let use_path_touched = run_use_path_rewrite(
            &mut self.registry,
            &self.changesets,
        )?;
        touched_files.extend(use_path_touched);
        self.pending_new_files = collect_new_files(&self.registry, &self.changesets);
        let mut validation = self.validate()?;
        validation.extend(conflicts);
        self.pending_file_moves = file_renames
            .iter()
            .map(|r| (PathBuf::from(&r.from), PathBuf::from(&r.to)))
            .collect();
        self.pending_file_renames = file_renames.clone();
        self.last_touched_files = touched_files.clone();
        Ok(ChangeReport {
            touched_files: touched_files.into_iter().collect(),
            conflicts: validation,
            file_moves: self.pending_file_moves.clone(),
        })
    }
    fn rebuild_registry_from_sources(&mut self) -> Result<()> {
        let project_root = find_project_root(&self.registry)?
            .unwrap_or_else(|| PathBuf::from("."));
        let mut rebuilt = NodeRegistry::new();
        let span_lookup = self.span_lookup.as_ref();
        for (file, source) in self.registry.sources.iter() {
            let content = source.as_str();
            let ast = syn::parse_file(content)
                .with_context(|| format!("Failed to parse {}", file.display()))?;
            let mut builder = NodeRegistryBuilder::new(
                &project_root,
                file,
                &mut rebuilt,
                source.clone(),
                span_lookup,
            );
            builder.visit_file(&ast);
            rebuilt.insert_ast(file.clone(), ast);
            rebuilt.insert_source(file.clone(), source.clone());
        }
        self.registry = rebuilt;
        Ok(())
    }
    fn refresh_sources_from_asts(
        &mut self,
        files: &HashSet<PathBuf>,
        exclude: &HashSet<PathBuf>,
    ) -> Result<()> {
        for file in files {
            if exclude.contains(file) {
                continue;
            }
            if let Some(ast) = self.registry.asts.get(file) {
                let rendered = crate::structured::render_file(ast);
                self.registry.sources.insert(file.clone(), Arc::new(rendered));
            }
        }
        Ok(())
    }
    fn build_moveset(&self) -> Result<MoveSet> {
        let project_root = find_project_root(&self.registry)?
            .unwrap_or_else(|| PathBuf::from("."));
        let mut moveset = MoveSet::default();
        for (_file, ops) in &self.changesets {
            for queued in ops {
                let crate::structured::NodeOp::MoveSymbol {
                    handle,
                    new_module_path,
                    ..
                } = &queued.op else {
                    continue;
                };
                let old_module = normalize_symbol_id(
                    &module_path_for_file(&project_root, &handle.file),
                );
                let new_module = normalize_symbol_id(new_module_path);
                moveset
                    .entries
                    .insert(queued.symbol_id.clone(), (old_module, new_module));
            }
        }
        Ok(moveset)
    }
    fn run_refactor_pipeline(&mut self, moveset: &MoveSet) -> Result<()> {
        let touched1 = run_pass1_canonical_rewrite(&mut self.registry, moveset)?;
        if !touched1.is_empty() {
            self.rebuild_registry_from_sources()?;
        }
        let touched2 = run_pass2_scope_rehydration(&mut self.registry, moveset)?;
        if !touched2.is_empty() {
            self.rebuild_registry_from_sources()?;
        }
        let touched3 = run_pass3_orphan_cleanup(&mut self.registry)?;
        if !touched3.is_empty() {
            self.rebuild_registry_from_sources()?;
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<Vec<EditConflict>> {
        let mut conflicts = Vec::new();
        for (symbol_id, _handle) in &self.registry.handles {
            if self.oracle.is_macro_generated(symbol_id) {
                conflicts
                    .push(EditConflict {
                        symbol_id: symbol_id.to_string(),
                        reason: "symbol generated by macro".to_string(),
                    });
            }
        }
        for ops in self.changesets.values() {
            for queued in ops {
                if let NodeOp::MutateField { mutation, .. } = &queued.op {
                    if matches!(mutation, FieldMutation::ReplaceSignature(_)) {
                        let impacts = self.oracle.impact_of(&queued.symbol_id);
                        if !impacts.is_empty() {
                            conflicts
                                .push(EditConflict {
                                    symbol_id: queued.symbol_id.clone(),
                                    reason: "signature change may require updating call sites"
                                        .to_string(),
                                });
                        }
                    }
                }
            }
        }
        Ok(conflicts)
    }
    pub fn commit(&self) -> Result<Vec<PathBuf>> {
        let mut written = Vec::new();
        let targets: Vec<PathBuf> = if self.last_touched_files.is_empty() {
            self.changesets.keys().cloned().collect()
        } else {
            self.last_touched_files.iter().cloned().collect()
        };
        for file in targets {
            let ast = self
                .registry
                .asts
                .get(&file)
                .with_context(|| format!("missing AST for {}", file.display()))?;
            let rendered = crate::structured::render_file(ast);
            if let Some(parent) = file.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&file, rendered)?;
            written.push(file.clone());
        }
        for (from, to) in &self.pending_file_moves {
            if let Some(parent) = to.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::rename(from, to)?;
            written.push(to.clone());
        }
        if !self.pending_file_renames.is_empty() {
            if let Some(project_root) = find_project_root(&self.registry)? {
                let symbol_table = build_symbol_index(&project_root, &self.registry)?;
                let mut touched = HashSet::new();
                update_mod_declarations(
                    &project_root,
                    &symbol_table,
                    &self.pending_file_renames,
                    &mut touched,
                )?;
                written.extend(touched.into_iter());
            }
        }
        if !self.pending_new_files.is_empty() {
            if let Some(project_root) = find_project_root(&self.registry)? {
                let symbol_table = build_symbol_index(&project_root, &self.registry)?;
                let mut touched = HashSet::new();
                let mut synthetic_renames: Vec<FileRename> = Vec::new();
                for (new_path, new_module_id) in &self.pending_new_files {
                    if let Some(ast) = self.registry.asts.get(new_path) {
                        if let Some(parent) = new_path.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        let rendered = crate::structured::render_file(ast);
                        std::fs::write(new_path, &rendered)?;
                        written.push(new_path.clone());
                    }
                    let parts: Vec<&str> = new_module_id.split("::").collect();
                    if parts.len() >= 2 {
                        synthetic_renames
                            .push(FileRename {
                                from: String::new(),
                                to: new_path.to_string_lossy().to_string(),
                                is_directory_move: false,
                                old_module_id: format!(
                                    "crate::__new__::{}", parts.last().unwrap()
                                ),
                                new_module_id: new_module_id.clone(),
                            });
                    }
                }
                update_mod_declarations(
                    &project_root,
                    &symbol_table,
                    &synthetic_renames,
                    &mut touched,
                )?;
                written.extend(touched.into_iter());
            }
        }
        written.sort();
        written.dedup();
        Ok(written)
    }
    pub fn preview(&self) -> Result<String> {
        let mut output = String::new();
        let targets: Vec<PathBuf> = if self.last_touched_files.is_empty() {
            self.changesets.keys().cloned().collect()
        } else {
            self.last_touched_files.iter().cloned().collect()
        };
        for file in targets.into_iter().filter(|p| self.original_sources.contains_key(p))
        {
            let original = &self.original_sources[&file];
            let ast = self
                .registry
                .asts
                .get(&file)
                .with_context(|| format!("missing AST for {}", file.display()))?;
            let rendered = crate::structured::render_file(ast);
            if original != &rendered {
                let diff = similar::TextDiff::from_lines(original, &rendered)
                    .unified_diff()
                    .header(
                        &format!("{} (original)", file.display()),
                        &format!("{} (updated)", file.display()),
                    )
                    .to_string();
                output.push_str(&diff);
                output.push('\n');
            }
        }
        if output.is_empty() {
            Ok(format!("{} files touched", self.changesets.len()))
        } else {
            Ok(output)
        }
    }
    /// DEBUG: return all registered symbol IDs currently indexed.
    pub fn debug_list_symbol_ids(&self) -> Vec<String> {
        self.registry.handles.keys().cloned().collect()
    }
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


fn parse_i64(value: Option<&String>) -> Option<i64> {
    value.and_then(|v| v.parse::<i64>().ok())
}
