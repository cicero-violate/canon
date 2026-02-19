#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

use rename::{collect_names};
use rename::rename::core::normalize_symbol_id;
use rename::rustc_integration::frontends::rustc::RustcFrontend;
use rename::rustc_integration::multi_capture::capture_project;
use rename::rustc_integration::project::CargoProject;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/rename");

    let report = collect_names(project_path)?;
    let target = report
        .symbols
        .iter()
        .find(|s| s.id.contains("rename::core::project_editor::ProjectEditor"))
        .or_else(|| report.symbols.iter().find(|s| s.name == "ProjectEditor" && s.kind == "struct"))
        .or_else(|| report.symbols.iter().find(|s| s.kind == "struct"))
        .ok_or("no symbols found")?;

    let syn_id = normalize_symbol_id(&target.id);
    println!("syn id: {syn_id}");

    let cargo = CargoProject::from_entry(project_path)?;
    let frontend = RustcFrontend::new();
    let artifacts = match capture_project(&frontend, &cargo, &[]) {
        Ok(artifacts) => artifacts,
        Err(err) => {
            eprintln!("rustc capture unavailable: {err}");
            return Ok(());
        }
    };

    let mut rustc_id = None;
    let mut rustc_candidates = Vec::new();
    for node in artifacts.snapshot.nodes() {
        let key = normalize_symbol_id(node.key.as_ref());
        if key.contains("ProjectEditor") {
            rustc_candidates.push(key.clone());
        }
        if key == syn_id {
            rustc_id = Some(key);
            break;
        }
    }

    match rustc_id {
        Some(id) => {
            println!("rustc id: {id}");
            println!("join key match: true");
        }
        None => {
            println!("rustc id: <not found>");
            println!("join key match: false");
            if !rustc_candidates.is_empty() {
                println!("rustc candidates:");
                for candidate in rustc_candidates {
                    println!("  {candidate}");
                }
            }
        }
    }

    Ok(())
}
