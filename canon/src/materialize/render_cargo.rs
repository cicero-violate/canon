use crate::ir::{ExternalDependency, Project};

pub fn render_cargo_toml(project: &Project, dependencies: &[ExternalDependency]) -> String {
    let mut out = String::new();
    out.push_str("# Derived from Canonical IR. Do not edit.\n");
    out.push_str("[package]\n");
    out.push_str(&format!("name = \"{}\"\n", project.name));
    out.push_str(&format!("version = \"{}\"\n", project.version));
    out.push_str("edition = \"2024\"\n\n");
    out.push_str("[dependencies]\n");
    if dependencies.is_empty() {
        out.push_str("# no external dependencies\n");
    } else {
        let mut deps = dependencies.to_vec();
        deps.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
        for dep in deps {
            if dep.name == dep.source {
                out.push_str(&format!("{} = \"{}\"\n", dep.name, dep.version));
            } else {
                out.push_str(&format!(
                    "{} = {{ package = \"{}\", version = \"{}\" }}\n",
                    dep.name, dep.source, dep.version
                ));
            }
        }
    }
    out
}
