pub fn normalize_symbol_id(raw: &str) -> String {
    normalize_symbol_id_with_crate(raw, None)
}


pub fn normalize_symbol_id(raw: &str) -> String {
    normalize_symbol_id_with_crate(raw, None)
}


pub fn normalize_symbol_id_with_crate(raw: &str, crate_name: Option<&str>) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut s = if let Some(start) = trimmed.find(" ~ ") {
        let tail = &trimmed[start + 3..];
        tail.trim_end_matches(')').to_string()
    } else {
        trimmed.to_string()
    };
    if trimmed == "crate" {
        return "crate".to_string();
    }
    s = s.trim_start_matches("::").to_string();
    if s.starts_with("self::") {
        s = format!("crate::{}", & s["self::".len()..]);
    }
    if s.starts_with("lib::") {
        s = format!("crate::{}", & s["lib::".len()..]);
    } else if s.starts_with("main::") {
        s = format!("crate::{}", & s["main::".len()..]);
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
            s = format!("crate::{}", & s[prefix.len()..]);
        } else if s == crate_name {
            s = "crate".to_string();
        } else if !s.starts_with("crate::") {
            s = format!("crate::{s}");
        }
    }
    s
}


pub fn normalize_symbol_id_with_crate(raw: &str, crate_name: Option<&str>) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut s = if let Some(start) = trimmed.find(" ~ ") {
        let tail = &trimmed[start + 3..];
        tail.trim_end_matches(')').to_string()
    } else {
        trimmed.to_string()
    };
    if trimmed == "crate" {
        return "crate".to_string();
    }
    s = s.trim_start_matches("::").to_string();
    if s.starts_with("self::") {
        s = format!("crate::{}", & s["self::".len()..]);
    }
    if s.starts_with("lib::") {
        s = format!("crate::{}", & s["lib::".len()..]);
    } else if s.starts_with("main::") {
        s = format!("crate::{}", & s["main::".len()..]);
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
            s = format!("crate::{}", & s[prefix.len()..]);
        } else if s == crate_name {
            s = "crate".to_string();
        } else if !s.starts_with("crate::") {
            s = format!("crate::{s}");
        }
    }
    s
}


fn strip_hash_suffix(segment: &str) -> &str {
    if let Some(idx) = segment.find('[') { &segment[..idx] } else { segment }
}


fn strip_hash_suffix(segment: &str) -> &str {
    if let Some(idx) = segment.find('[') { &segment[..idx] } else { segment }
}


pub fn normalize_symbol_id(raw: &str) -> String {
    normalize_symbol_id_with_crate(raw, None)
}


pub fn normalize_symbol_id(raw: &str) -> String {
    normalize_symbol_id_with_crate(raw, None)
}


pub fn normalize_symbol_id_with_crate(raw: &str, crate_name: Option<&str>) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut s = if let Some(start) = trimmed.find(" ~ ") {
        let tail = &trimmed[start + 3..];
        tail.trim_end_matches(')').to_string()
    } else {
        trimmed.to_string()
    };
    if trimmed == "crate" {
        return "crate".to_string();
    }
    s = s.trim_start_matches("::").to_string();
    if s.starts_with("self::") {
        s = format!("crate::{}", & s["self::".len()..]);
    }
    if s.starts_with("lib::") {
        s = format!("crate::{}", & s["lib::".len()..]);
    } else if s.starts_with("main::") {
        s = format!("crate::{}", & s["main::".len()..]);
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
            s = format!("crate::{}", & s[prefix.len()..]);
        } else if s == crate_name {
            s = "crate".to_string();
        } else if !s.starts_with("crate::") {
            s = format!("crate::{s}");
        }
    }
    s
}


pub fn normalize_symbol_id_with_crate(raw: &str, crate_name: Option<&str>) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut s = if let Some(start) = trimmed.find(" ~ ") {
        let tail = &trimmed[start + 3..];
        tail.trim_end_matches(')').to_string()
    } else {
        trimmed.to_string()
    };
    if trimmed == "crate" {
        return "crate".to_string();
    }
    s = s.trim_start_matches("::").to_string();
    if s.starts_with("self::") {
        s = format!("crate::{}", & s["self::".len()..]);
    }
    if s.starts_with("lib::") {
        s = format!("crate::{}", & s["lib::".len()..]);
    } else if s.starts_with("main::") {
        s = format!("crate::{}", & s["main::".len()..]);
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
            s = format!("crate::{}", & s[prefix.len()..]);
        } else if s == crate_name {
            s = "crate".to_string();
        } else if !s.starts_with("crate::") {
            s = format!("crate::{s}");
        }
    }
    s
}


fn strip_hash_suffix(segment: &str) -> &str {
    if let Some(idx) = segment.find('[') { &segment[..idx] } else { segment }
}


fn strip_hash_suffix(segment: &str) -> &str {
    if let Some(idx) = segment.find('[') { &segment[..idx] } else { segment }
}
