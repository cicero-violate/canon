#![cfg(feature = "rustc_frontend")]
use super::frontend_context::FrontendMetadata;
use super::RustcFrontendError;
use crate::compiler_capture::graph::GraphDelta;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::Stdio;

#[derive(Serialize, Deserialize)]
pub struct CaptureRequest {
    pub entry: String,
    pub args: Vec<String>,
    pub env_vars: Vec<(String, String)>,
    pub metadata: FrontendMetadata,
}
/// Real rustc frontend that runs `rustc_driver` to capture MIR call graphs.
#[derive(Debug, Clone)]
pub struct RustcFrontend {
    edition: String,
    crate_type: String,
    rust_version: Option<String>,
    target_name: Option<String>,
    workspace_root: Option<PathBuf>,
    package_name: Option<String>,
    package_version: Option<String>,
    package_features: Vec<String>,
    cfg_flags: Vec<String>,
}
impl Default for RustcFrontend {
    fn default() -> Self {
        Self {
            edition: "2021".into(),
            crate_type: "lib".into(),
            rust_version: None,
            target_name: None,
            workspace_root: None,
            package_name: None,
            package_version: None,
            package_features: Vec::new(),
            cfg_flags: Vec::new(),
        }
    }
}
impl RustcFrontend {
    /// Creates a new frontend with default options.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the crate type used when invoking rustc (e.g., `lib`, `bin`).
    pub fn with_crate_type(mut self, crate_type: impl Into<String>) -> Self {
        self.crate_type = crate_type.into();
        self
    }
    /// Sets the Rust edition supplied to rustc.
    pub fn with_edition(mut self, edition: impl Into<String>) -> Self {
        self.edition = edition.into();
        self
    }
    /// Sets the Rust toolchain version recorded in snapshot metadata.
    pub fn with_rust_version(mut self, rust_version: impl Into<String>) -> Self {
        self.rust_version = Some(rust_version.into());
        self
    }
    /// Sets the Cargo target name (lib/bin/test name).
    pub fn with_target_name(mut self, target_name: impl Into<String>) -> Self {
        self.target_name = Some(target_name.into());
        self
    }
    /// Sets the workspace root recorded for this capture.
    pub fn with_workspace_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.workspace_root = Some(root.into());
        self
    }
    /// Declares the Cargo package metadata.
    pub fn with_package_info(mut self, name: impl Into<String>, version: impl Into<String>) -> Self {
        self.package_name = Some(name.into());
        self.package_version = Some(version.into());
        self
    }
    /// Declares the set of enabled Cargo features.
    pub fn with_package_features<I, S>(mut self, features: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.package_features = features.into_iter().map(|s| s.into()).collect();
        self
    }
    /// Declares the cfg flags passed to rustc.
    pub fn with_cfg_flags<I, S>(mut self, cfgs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.cfg_flags = cfgs.into_iter().map(|s| s.into()).collect();
        self
    }
    /// Captures graph deltas from the provided Rust source file.
    pub fn capture_deltas<P: AsRef<Path>>(&self, entry: P, extra_args: &[String], env_vars: &[(String, String)]) -> Result<Vec<GraphDelta>, RustcFrontendError> {
        let entry = fs::canonicalize(entry)?;
        let mut args = vec!["rustc".to_string(), entry.display().to_string(), format!("--crate-type={}", self.crate_type), format!("--edition={}", self.edition)];
        if let Some(name) = entry.file_stem().and_then(|s| s.to_str()) {
            args.push(format!("--crate-name={}", name.replace('-', "_")));
        }
        let sysroot = resolve_rustc_sysroot().map_err(RustcFrontendError::Sysroot)?;
        args.push("--sysroot".into());
        args.push(sysroot.to_string_lossy().into());
        args.extend(extra_args.iter().cloned());
        let metadata = FrontendMetadata {
            edition: self.edition.clone(),
            rust_version: self.rust_version.clone(),
            crate_type: self.crate_type.clone(),
            target_triple: std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()),
            target_name: self.target_name.clone(),
            workspace_root: self.workspace_root.as_ref().map(|p| p.display().to_string()),
            package_name: self.package_name.clone(),
            package_version: self.package_version.clone(),
            package_features: self.package_features.clone(),
            cfg_flags: {
                let mut cfgs = self.cfg_flags.clone();
                cfgs.extend(extract_cfg_flags(extra_args));
                cfgs.sort();
                cfgs.dedup();
                cfgs
            },
        };
        let request = CaptureRequest { entry: entry.display().to_string(), args, env_vars: env_vars.to_vec(), metadata };
        let request_json = serde_json::to_string(&request).map_err(|e| RustcFrontendError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
        let driver = resolve_capture_driver()?;
        let lib_path = sysroot.join("lib").display().to_string();
        let ld_path = match std::env::var("LD_LIBRARY_PATH") {
            Ok(existing) if !existing.is_empty() => format!("{lib_path}:{existing}"),
            _ => lib_path,
        };
        let mut child = Command::new(&driver).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).env("LD_LIBRARY_PATH", &ld_path).spawn()?;
        child.stdin.take().unwrap().write_all(request_json.as_bytes())?;
        let output = child.wait_with_output()?;
        if !output.status.success() {
            return Err(RustcFrontendError::Io(std::io::Error::new(std::io::ErrorKind::Other, format!("capture_driver exited with {:?}", output.status.code()))));
        }
        serde_json::from_slice(&output.stdout).map_err(|e| RustcFrontendError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))
    }
}
fn extract_cfg_flags(extra_args: &[String]) -> Vec<String> {
    let mut cfgs = Vec::new();
    let mut iter = extra_args.iter().peekable();
    while let Some(arg) = iter.next() {
        if let Some(rest) = arg.strip_prefix("--cfg=") {
            cfgs.push(rest.to_string());
            continue;
        }
        if arg == "--cfg" {
            if let Some(next) = iter.next() {
                cfgs.push(next.to_string());
            }
            continue;
        }
    }
    cfgs
}
fn resolve_rustc_sysroot() -> Result<PathBuf, std::io::Error> {
    if let Ok(sysroot) = std::env::var("SYSROOT") {
        return Ok(PathBuf::from(sysroot));
    }
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let output = Command::new(rustc).args(["--print", "sysroot"]).output()?;
    Ok(PathBuf::from(String::from_utf8_lossy(&output.stdout).trim()))
}

fn resolve_capture_driver() -> Result<PathBuf, RustcFrontendError> {
    if let Ok(mut exe) = std::env::current_exe() {
        exe.pop();
        let c = exe.join("capture_driver");
        if c.exists() {
            return Ok(c);
        }
    }
    if let Ok(mut cwd) = std::env::current_dir() {
        loop {
            let c = cwd.join("target/debug/capture_driver");
            if c.exists() {
                return Ok(c);
            }
            if !cwd.pop() {
                break;
            }
        }
    }
    Err(RustcFrontendError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "capture_driver not found; run `cargo build -p compiler_capture --bin capture_driver`")))
}
