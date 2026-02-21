use anyhow::Result;


use std::path::PathBuf;


use std::process::Command;


pub fn format_files(paths: &[PathBuf]) -> Result<Vec<String>> {
    let existing: Vec<_> = paths.iter().filter(|p| p.exists()).collect();
    if existing.is_empty() {
        return Ok(Vec::new());
    }
    let mut errors = Vec::new();
    for path in &existing {
        let mut cmd = Command::new("rustfmt");
        cmd.arg("--edition").arg("2021");
        cmd.arg(path);
        match cmd.status() {
            Ok(status) if status.success() => {}
            Ok(status) => {
                errors
                    .push(
                        format!(
                            "rustfmt failed for {}: exit code {}", path.display(), status
                            .code().unwrap_or(- 1)
                        ),
                    );
            }
            Err(e) => {
                errors
                    .push(format!("failed to run rustfmt on {}: {}", path.display(), e));
            }
        }
    }
    if !errors.is_empty() {
        eprintln!("Warning: {} rustfmt errors occurred:", errors.len());
        for error in &errors {
            eprintln!("  - {}", error);
        }
    }
    Ok(errors)
}
