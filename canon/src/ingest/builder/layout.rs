use crate::ir::Module;
use crate::layout::{LayoutAssignment, LayoutFile, LayoutGraph, LayoutModule, LayoutNode};

#[derive(Default)]
pub(crate) struct LayoutAccumulator {
    pub assignments: Vec<LayoutAssignment>,
    /// file_id → (module_id, path)
    pub file_registry: std::collections::HashMap<String, (String, String)>,
}

impl LayoutAccumulator {
    pub fn assign(&mut self, node: LayoutNode, file_id: Option<String>) {
        if let Some(file_id) = file_id {
            self.assignments.push(LayoutAssignment {
                node,
                file_id,
                rationale: "ING-001: inferred from source".to_owned(),
            });
        }
    }

    /// Called once per file during module building so layout knows which
    /// files belong to which module.
    pub fn register_file(&mut self, module_id: &str, file_id: &str, path: &str) {
        self.file_registry
            .entry(file_id.to_owned())
            .or_insert_with(|| (module_id.to_owned(), path.to_owned()));
    }

    pub fn into_graph(self, modules: &[Module]) -> LayoutGraph {
        use std::collections::HashMap;
        // Group registered files by module_id
        let mut files_by_module: HashMap<&str, Vec<LayoutFile>> = HashMap::new();
        for (file_id, (module_id, path)) in &self.file_registry {
            files_by_module
                .entry(module_id.as_str())
                .or_default()
                .push(LayoutFile {
                    id: file_id.clone(),
                    path: path.clone(),
                    use_block: Vec::new(),
                });
        }
        let layout_modules = modules
            .iter()
            .map(|module| {
                let mut files = files_by_module
                    .remove(module.id.as_str())
                    .unwrap_or_default();
                files.sort_by(|a, b| a.id.cmp(&b.id));
                LayoutModule {
                    id: module.id.clone(),
                    name: module.name.clone(),
                    files,
                    imports: Vec::new(),
                }
            })
            .collect();
        LayoutGraph {
            modules: layout_modules,
            routing: self.assignments,
        }
    }
}
