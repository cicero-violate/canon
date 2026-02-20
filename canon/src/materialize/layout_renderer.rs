use std::collections::HashMap;
use crate::layout::{FileBinding, LayoutFile, FileTopology, LayoutModule, LayoutNode};
use crate::ir::{Function, Struct, Trait};
/// Helper map to look up file assignments for nodes.
pub struct LayoutLookup<'a> {
    files: HashMap<&'a str, &'a LayoutFile>,
    routing: HashMap<&'a str, &'a FileBinding>,
    modules: HashMap<&'a str, &'a LayoutModule>,
}
impl<'a> LayoutLookup<'a> {
    pub fn new(graph: &'a FileTopology) -> Self {
        let mut files = HashMap::new();
        let mut modules = HashMap::new();
        for module in &graph.modules {
            modules.insert(module.id.as_str(), module);
            for file in &module.files {
                files.insert(file.id.as_str(), file);
            }
        }
        let routing = graph
            .routing
            .iter()
            .map(|assignment| {
                let key = match &assignment.node {
                    LayoutNode::Struct(id)
                    | LayoutNode::Enum(id)
                    | LayoutNode::Trait(id)
                    | LayoutNode::Impl(id)
                    | LayoutNode::Function(id) => id.as_str(),
                };
                (key, assignment)
            })
            .collect();
        Self { files, routing, modules }
    }
    pub fn file_for_function(&self, function: &Function) -> Option<&LayoutFile> {
        self.routing
            .get(function.id.as_str())
            .and_then(|assignment| self.files.get(assignment.file_id.as_str()))
            .copied()
    }
    pub fn files_for_module(&self, module_id: &str) -> Vec<&LayoutFile> {
        self.modules
            .get(module_id)
            .map(|module| module.files.iter().collect())
            .unwrap_or_default()
    }
    pub fn assignment_for_struct(&self, structure: &Struct) -> Option<&FileBinding> {
        self.routing.get(structure.id.as_str()).copied()
    }
    pub fn assignment_for_trait(&self, tr: &Trait) -> Option<&FileBinding> {
        self.routing.get(tr.id.as_str()).copied()
    }
}
