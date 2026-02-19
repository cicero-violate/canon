pub fn normalize_symbol_id(raw: &str) -> String {
    normalize_symbol_id_with_crate(raw, None)
}

pub fn normalize_symbol_id_with_crate(raw: &str, crate_name: Option<&str>) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed == "crate" {
        return "crate".to_string();
    }

    let mut s = trimmed.trim_start_matches("::").to_string();
    if s.starts_with("self::") {
        s = format!("crate::{}", &s["self::".len()..]);
    }

    let parts: Vec<&str> = s.split("::").collect();
    let mut normalized_parts = Vec::with_capacity(parts.len());
    for part in parts {
        let cleaned = strip_hash_suffix(part);
        normalized_parts.push(cleaned.to_string());
    }
    s = normalized_parts.join("::");

    if let Some(crate_name) = crate_name {
        let prefix = format!("{crate_name}::");
        if s.starts_with(&prefix) {
            s = format!("crate::{}", &s[prefix.len()..]);
        } else if s == crate_name {
            s = "crate".to_string();
        }
    }

    s
}

fn strip_hash_suffix(segment: &str) -> &str {
    if let Some(idx) = segment.find('[') {
        &segment[..idx]
    } else {
        segment
    }
}
