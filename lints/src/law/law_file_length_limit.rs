use std::cell::RefCell;
use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};

use lazy_static::lazy_static;
use rustc_lint::{LateContext, LintContext, LintStore};
use rustc_session::declare_lint;
use rustc_span::{Span, StableSourceFileId};

const MAX_LINES_PER_FILE: usize = 500;

lazy_static! {
    static ref WORKSPACE_PREFIX: String = resolve_workspace_prefix();
}

thread_local! {
    static CHECKED_FILES: RefCell<HashSet<StableSourceFileId>> = RefCell::new(HashSet::new());
}

declare_lint! {
    pub FILE_TOO_LONG,
    Warn,
    "source file exceeds the maximum allowed number of lines"
}

pub fn register_law(store: &mut LintStore) {
    store.register_lints(&[&FILE_TOO_LONG]);
}

pub fn enforce_file_length(cx: &LateContext<'_>, span: Span) {
    let source_map = cx.sess().source_map();
    let file_rc = source_map.lookup_source_file(span.lo());
    let file = file_rc.as_ref();

    if file.is_imported() {
        return;
    }

    let already_checked = CHECKED_FILES.with(|set| {
        let mut guard = set.borrow_mut();
        if guard.contains(&file.stable_id) {
            true
        } else {
            guard.insert(file.stable_id);
            false
        }
    });

    if already_checked {
        return;
    }

    let workspace_root = Path::new(WORKSPACE_PREFIX.as_str());
    let raw_name = file.name.prefer_local_unconditionally().to_string();
    let raw_path = Path::new(&raw_name);
    let absolute_path = if raw_path.is_absolute() { raw_path.to_path_buf() } else { workspace_root.join(raw_path) };
    let canonical_path = match absolute_path.canonicalize() {
        Ok(path) => path,
        Err(_) => return,
    };

    if !canonical_path.starts_with(workspace_root) {
        return;
    }

    let display_path = canonical_path.strip_prefix(workspace_root).map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|_| canonical_path.to_string_lossy().into_owned());

    let line_count = file.count_lines();
    if line_count <= MAX_LINES_PER_FILE {
        return;
    }

    let span = Span::with_root_ctxt(file.start_pos, file.end_position());
    cx.span_lint(FILE_TOO_LONG, span, |diag| {
        diag.note(format!("file `{}` has {line_count} lines which exceeds the limit of {MAX_LINES_PER_FILE}", display_path));
    });
}

pub fn reset_cache() {
    CHECKED_FILES.with(|set| set.borrow_mut().clear());
}

fn resolve_workspace_prefix() -> String {
    let candidate = env::var("CANON_WORKSPACE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().expect("lints crate must have parent workspace directory").to_path_buf());

    let canonical = candidate.canonicalize().unwrap_or(candidate).to_string_lossy().into_owned();
    eprintln!("LAW using workspace prefix: {}", canonical);
    canonical
}
