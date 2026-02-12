#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_session;

extern crate lints;
extern crate serde;
extern crate serde_json;

use rustc_driver::Callbacks;
use rustc_session::EarlyDiagCtxt;
use std::{
    fs,
    io,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(serde::Serialize)]
struct PersistedSignals {
    crate_name: String,
    captured_at_unix: u64,
    signals: Vec<lints::LintSignal>,
}

struct LintCallbacks;

impl Callbacks for LintCallbacks {
    fn config(&mut self, config: &mut rustc_interface::Config) {
        let prev = config.register_lints.take();
        config.register_lints = Some(Box::new(move |sess, lint_store| {
            if let Some(prev) = &prev {
                prev(sess, lint_store);
            }
            lints::register_lints(lint_store);
        }));
    }
}

fn exec_real_rustc(real_rustc: &str, args: &[String], reason: &str) -> ! {
    let status = std::process::Command::new(real_rustc)
        .args(args)
        .status()
        .unwrap_or_else(|err| panic!("failed to exec real rustc ({reason}): {err:?}"));
    std::process::exit(status.code().unwrap_or(0));
}

fn main() {
    let mut callbacks = LintCallbacks;
    let mut argv: Vec<String> = std::env::args().collect();
    let real_rustc = argv.get(1).cloned().expect("missing real rustc path");
    let crate_name = find_flag_value(&argv, "--crate-name");
    let workspace_root = workspace_root_from_exe();
    if let Some(root) = workspace_root.as_ref() {
        std::env::set_var("CANON_WORKSPACE_ROOT", root);
    }

    let is_probe = argv.iter().any(|a| a.starts_with("--print="))
        || argv.iter().any(|a| a == "-")
        || argv.windows(2).any(|w| w[0] == "--crate-name" && w[1] == "___")
        || argv.iter().any(|a| a == "-vV" || a == "--version");

    if is_probe {
        exec_real_rustc(&real_rustc, &argv[2..], "probe");
    }

    let is_primary = std::env::var_os("CARGO_PRIMARY_PACKAGE").is_some();

    // Only run lints for primary packages.
    // All dependencies are delegated to real rustc.
    if !is_primary {
        exec_real_rustc(&real_rustc, &argv[2..], "dependency");
    }

    let mut args = Vec::with_capacity(argv.len().saturating_sub(1));
    args.push("rustc".to_string());
    args.extend(argv.drain(2..));

    let _diag = EarlyDiagCtxt::new(rustc_session::config::ErrorOutputType::default());
    rustc_driver::run_compiler(&args, &mut callbacks);

    let signals = {
        let guard = lints::LINT_SIGNALS.lock().unwrap();
        guard.clone()
    };
    if !signals.is_empty() {
        if let Some(repo_root) = workspace_root.as_ref() {
            if let Some(crate_name) = crate_name.as_deref() {
                if let Err(err) = persist_signals(&repo_root, crate_name, &signals) {
                    eprintln!(
                        "warning: failed to persist lint signals for {crate_name}: {err}"
                    );
                }
            }
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&signals)
                .expect("failed to serialize judgment signals")
        );
    }

    std::process::exit(0);
}

fn find_flag_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].clone())
}

fn workspace_root_from_exe() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let debug_dir = exe.parent()?;
    let target_dir = debug_dir.parent()?;
    target_dir.parent().map(|p| p.to_path_buf())
}

fn persist_signals(
    repo_root: &Path,
    crate_name: &str,
    signals: &[lints::LintSignal],
) -> io::Result<()> {
    let store_dir = repo_root.join("canon_store").join("lint_signals");
    fs::create_dir_all(&store_dir)?;
    let payload = PersistedSignals {
        crate_name: crate_name.to_string(),
        captured_at_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        signals: signals.to_vec(),
    };
    let file = fs::File::create(store_dir.join(format!("{crate_name}.json")))?;
    serde_json::to_writer_pretty(file, &payload)
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;
    Ok(())
}
