fn strip_hash_suffix(segment: &str) -> &str {
    if let Some(idx) = segment.find('[') { &segment[..idx] } else { segment }
}
