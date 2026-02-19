use anyhow::{Context, Result};
use std::path::Path;

use super::collect::emit_names;
use super::rename::apply_rename;

pub fn run_names(args: &[String]) -> Result<()> {
    let mut project = None;
    let mut out = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--project" => {
                project = Some(&args[i + 1]);
                i += 2;
            }
            "--out" => {
                out = Some(&args[i + 1]);
                i += 2;
            }
            _ => i += 1,
        }
    }
    let project = project.context("--project required")?;
    let out = out.map_or(".semantic-lint/names.json", |v| v);
    emit_names(Path::new(project), Path::new(out))
}

pub fn run_rename(args: &[String]) -> Result<()> {
    let mut project = None;
    let mut map_path = None;
    let mut out_path = None;
    let mut dry_run = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--project" => {
                project = Some(&args[i + 1]);
                i += 2;
            }
            "--map" => {
                map_path = Some(&args[i + 1]);
                i += 2;
            }
            "--out" => {
                out_path = Some(&args[i + 1]);
                i += 2;
            }
            "--dry-run" => {
                dry_run = true;
                i += 1;
            }
            _ => i += 1,
        }
    }
    let project = project.context("--project required")?;
    let map_path = map_path.context("--map required")?;
    apply_rename(
        Path::new(project),
        Path::new(map_path),
        dry_run,
        out_path.map(Path::new),
    )
}
