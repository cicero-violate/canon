pub enum LinuxFact {
    Exists(PathBuf),
    File(PathBuf),
    Dir(PathBuf),
    ProcessRunning(String),
    BinaryInstalled(String),
}


fn binary_on_path(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            if exists_executable(&dir.join(name)) {
                return true;
            }
        }
    }
    false
}


fn exists_executable(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = path.metadata() {
            return meta.permissions().mode() & 0o111 != 0;
        }
        false
    }
    #[cfg(not(unix))] { true }
}


pub fn probe_fact(fact: &LinuxFact) -> bool {
    match fact {
        LinuxFact::Exists(path) => path.exists(),
        LinuxFact::File(path) => path.is_file(),
        LinuxFact::Dir(path) => path.is_dir(),
        LinuxFact::ProcessRunning(_name) => false,
        LinuxFact::BinaryInstalled(name) => binary_on_path(name),
    }
}
