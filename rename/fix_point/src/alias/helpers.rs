pub fn extract_module_from_path(path: &str) -> String {
    if let Some(last_sep) = path.rfind("::") {
        path[..last_sep].to_string()
    } else {
        "crate".to_string()
    }
}
