use crate::delta::Delta;
use crate::memory_engine::CanonicalState;
use crate::page::Page;
use crate::page::{DeltaAppliable, PageAccess};
use crate::tlog::TlogManager;
use std::path::Path;

pub fn apply_log(pages: &mut [Page], deltas: &[Delta]) {
    for delta in deltas {
        if let Some(page) = pages.iter_mut().find(|p| p.id() == delta.page_id) {
            let _ = page.apply_delta(delta);
        }
    }
}

pub fn replay(path: &Path) -> std::io::Result<CanonicalState> {
    let manager = TlogManager::new(path)?;
    let mut state = CanonicalState::new_empty();
    manager.replay_all(&mut state)?;
    Ok(state)
}
