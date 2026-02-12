#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_session;

extern crate lints;
extern crate serde_json;

use rustc_driver::Callbacks;
use rustc_session::EarlyDiagCtxt;

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

    let signals = lints::LINT_SIGNALS.lock().unwrap();
    if !signals.is_empty() {
        println!("{}", serde_json::to_string_pretty(&*signals)
            .expect("failed to serialize judgment signals"));
    }

    std::process::exit(0);
}
