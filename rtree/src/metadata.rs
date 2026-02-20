use anyhow::{Context, Result};
use cargo_metadata::MetadataCommand;
use std::path::PathBuf;

pub struct ProjectMetadata {
    pub root: PathBuf,
    pub src_root: PathBuf,
    pub package_name: String,
}

pub fn get_project_metadata(path: &str) -> Result<ProjectMetadata> {
    let metadata = MetadataCommand::new().manifest_path(find_cargo_toml(path)?).exec().context("Failed to execute cargo metadata")?;

    let package = metadata.root_package().context("No root package found. Is this a Cargo project?")?;

    let root = package.manifest_path.parent().context("manifest_path has no parent")?.to_path_buf();
    let src_root = root.join("src");

    Ok(ProjectMetadata { root: root.into(), src_root: src_root.into(), package_name: package.name.clone() })
}

fn find_cargo_toml(start_path: &str) -> Result<PathBuf> {
    let mut current = PathBuf::from(start_path);

    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            return Ok(cargo_toml);
        }

        if !current.pop() {
            anyhow::bail!("Could not find Cargo.toml in {} or parent directories", start_path);
        }
    }
}
