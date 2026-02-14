use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::{Local, TimeZone};
use serde_json::Value;
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

fn main() {
    let mut args = env::args().skip(1);
    let mut base_dir = ".".to_string();
    let mut rebuild = false;

    while let Some(arg) = args.next() {
        if arg == "--rebuild" {
            rebuild = true;
        } else {
            base_dir = arg;
        }
    }
    let base_path = PathBuf::from(base_dir);

    if !base_path.is_dir() {
        eprintln!("Base path is not a directory: {}", base_path.display());
        std::process::exit(1);
    }

    let mut json_files = Vec::new();

    for entry in WalkDir::new(&base_path)
        .follow_links(true)
        .into_iter()
        .filter_map(|entry| entry.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.components().any(|c| c.as_os_str() == "codeblock_extractor") {
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        if !is_message_json(&path, &base_path) {
            continue;
        }
        json_files.push(path.to_path_buf());
    }

    json_files.sort();

    let mut counters = if rebuild {
        HashMap::new()
    } else {
        scan_existing_counts(&base_path)
    };
    let meta_map = match build_message_meta(&json_files) {
        Ok(map) => map,
        Err(err) => {
            eprintln!("Failed to build message metadata: {err}");
            HashMap::new()
        }
    };

    for path in json_files {
        if let Err(err) = process_message_file(&path, &base_path, &mut counters, &meta_map, rebuild) {
            eprintln!("Failed to process {}: {}", path.display(), err);
        }
    }

    if let Err(err) = fix_shell_intents_in_tree(&base_path) {
        eprintln!("Failed to rewrite shell intents: {err}");
    }
}

fn process_message_file(
    path: &Path,
    base_path: &Path,
    counters: &mut HashMap<String, Counters>,
    meta_map: &HashMap<String, MessageMeta>,
    rebuild: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let json: Value = serde_json::from_str(&contents)?;

    let conversation_id = match json.get("conversationId").and_then(Value::as_str) {
        Some(value) => value,
        None => return Ok(()),
    };
    let message_id = match json.get("messageId").and_then(Value::as_str) {
        Some(value) => value,
        None => return Ok(()),
    };

    let mut output_dir = base_path.join(conversation_id).join(message_id);
    if !output_dir.is_dir() {
        let base_name = base_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if base_name == conversation_id {
            let alt_dir = base_path.join(message_id);
            if alt_dir.is_dir() {
                output_dir = alt_dir;
            } else {
                return Err(format!(
                    "Output directory missing: {}",
                    output_dir.display()
                )
                .into());
            }
        } else {
            return Err(format!(
                "Output directory missing: {}",
                output_dir.display()
            )
            .into());
        }
    }

    let response_text = json
        .get("responseText")
        .and_then(Value::as_str)
        .map(|text| text.to_string())
        .or_else(|| extract_response_from_sse(json.get("sseEvents")));

    if rebuild {
        cleanup_message_dir(&output_dir)?;
    } else if has_existing_codeblocks(&output_dir) {
        if let Err(err) = rewrite_shell_intents_with_cwd(&output_dir) {
            eprintln!("Failed to update shell intents in {}: {err}", output_dir.display());
        }
        return Ok(());
    }

    if let Some(text) = response_text {
        let meta_key = format!("{}/{}", conversation_id, message_id);
        let meta = meta_map.get(&meta_key);
        let message_name = meta
            .map(|m| format!("message_{}{}.md", m.seq_label, m.branch_suffix))
            .unwrap_or_else(|| "message.md".to_string());
        let message_path = output_dir.join(message_name);
        let mut message_body = String::new();
        message_body.push_str(&format!("conversationId: {}\n", conversation_id));
        message_body.push_str(&format!("messageId: {}\n", message_id));
        if let Some(timestamp) = json.get("timestamp").and_then(Value::as_i64) {
            message_body.push_str(&format!(
                "timestamp: {}\n",
                format_timestamp(timestamp)
            ));
        }
        message_body.push('\n');
        message_body.push_str(&text);
        fs::write(&message_path, message_body.as_bytes())?;

        let branch_suffix = meta.map(|m| m.branch_suffix.clone()).unwrap_or_default();
        let counter_key = branch_counter_key(conversation_id, &branch_suffix);
        let counter = counters
            .entry(counter_key)
            .or_insert_with(Counters::default);

        let code_blocks = extract_code_blocks(&text);
        for block in code_blocks {
            let block_type = classify_block(&block);
            let file_name = match block_type {
                BlockType::Patch => {
                    let seq = counter.next_patch_seq();
                    Some(format!("patch_06_delta_{:02}.json", seq))
                }
                BlockType::Shell => {
                    let seq = counter.next_shell_seq();
                    Some(format!("shell_01_intent_{:02}.json", seq))
                }
                BlockType::Plan => {
                    let seq = counter.next_plan_seq();
                    Some(format!("plan_01_intent_{:02}.json", seq))
                }
                BlockType::Rust => {
                    let seq = counter.next_rust_seq();
                    Some(format!("rustfile_{:02}.rs", seq))
                }
                BlockType::Python => {
                    let seq = counter.next_python_seq();
                    Some(format!("pythonfile_{:02}.py", seq))
                }
                BlockType::Julia => {
                    let seq = counter.next_julia_seq();
                    Some(format!("juliafile_{:02}.jl", seq))
                }
                BlockType::Json => {
                    let seq = counter.next_json_seq();
                    Some(format!("jsonfile_{:02}.json", seq))
                }
                BlockType::Other => None,
            };

            if let Some(file_name) = file_name {
                let suffix = branch_suffix.clone();
                let block_path = output_dir.join(apply_suffix(&file_name, &suffix));
                match block_type {
                    BlockType::Shell => {
                        write_shell_intent_at_path(
                            &block_path,
                            &output_dir,
                            &block.content,
                        )?;
                    }
                    BlockType::Plan => {
                        write_plan_intent_at_path(&block_path, &block.content)?;
                    }
                    BlockType::Patch => {
                        write_patch_artifacts_at_path(&block_path, &block.content)?;
                    }
                    _ => {
                        fs::write(&block_path, block.content.as_bytes())?;
                    }
                }
            }
        }
        remove_legacy_patch_files(&output_dir)?;
    }

    Ok(())
}

fn is_message_json(path: &Path, base_path: &Path) -> bool {
    if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
        return false;
    }
    if let Ok(relative) = path.strip_prefix(base_path) {
        if relative.components().any(|c| {
            let part = c.as_os_str();
            part == "codeblock_extractor" || part == "message_pipeline" || part == "target"
        }) {
            return false;
        }
    }
    let file_name = match path.file_name().and_then(|s| s.to_str()) {
        Some(name) => name,
        None => return false,
    };
    if file_name.starts_with("jsonfile_") {
        return false;
    }
    let file_stem = match path.file_stem().and_then(|s| s.to_str()) {
        Some(stem) => stem,
        None => return false,
    };
    let parent_name = match path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()) {
        Some(name) => name,
        None => return false,
    };
    file_stem == parent_name
}

#[derive(Default)]
struct Counters {
    patch: usize,
    shell: usize,
    plan: usize,
    rust: usize,
    python: usize,
    julia: usize,
    json: usize,
}

impl Counters {
    fn next_patch_seq(&mut self) -> usize {
        self.patch += 1;
        self.patch
    }
    fn next_shell_seq(&mut self) -> usize {
        self.shell += 1;
        self.shell
    }
    fn next_plan_seq(&mut self) -> usize {
        self.plan += 1;
        self.plan
    }
    fn next_rust_seq(&mut self) -> usize {
        self.rust += 1;
        self.rust
    }
    fn next_python_seq(&mut self) -> usize {
        self.python += 1;
        self.python
    }
    fn next_julia_seq(&mut self) -> usize {
        self.julia += 1;
        self.julia
    }
    fn next_json_seq(&mut self) -> usize {
        self.json += 1;
        self.json
    }
}

enum BlockType {
    Patch,
    Shell,
    Plan,
    Rust,
    Python,
    Julia,
    Json,
    Other,
}

struct CodeBlock {
    language: Option<String>,
    content: String,
}

fn extract_response_from_sse(sse_events: Option<&Value>) -> Option<String> {
    let events = sse_events?.as_array()?;
    let mut current_text = String::new();
    let mut capturing = false;

    for event in events {
        match event {
            Value::String(_) => continue,
            Value::Object(map) => {
                let mut patch_applied = false;
                if let Some(patches) = map.get("v").and_then(Value::as_array) {
                    patch_applied = apply_patches(patches, &mut current_text, &mut capturing);
                } else if map.get("o").and_then(Value::as_str) == Some("patch") {
                    if let Some(patches) = map.get("v").and_then(Value::as_array) {
                        patch_applied = apply_patches(patches, &mut current_text, &mut capturing);
                    }
                }

                if let (Some(path), Some(op), Some(value)) = (
                    map.get("p").and_then(Value::as_str),
                    map.get("o").and_then(Value::as_str),
                    map.get("v"),
                ) {
                    if apply_patch(path, op, value, &mut current_text, &mut capturing) {
                        patch_applied = true;
                    }
                }

                if !patch_applied {
                    if let Some(Value::String(text)) = map.get("v") {
                    if capturing {
                        current_text.push_str(text);
                    }
                }
                }

                if let Some(message) = map
                    .get("message")
                    .or_else(|| map.get("v").and_then(|v| v.get("message")))
                {
                    if let Some(text) = extract_assistant_text(message) {
                        current_text = text;
                        capturing = true;
                    }
                }
            }
            _ => {}
        }
    }

    if current_text.is_empty() {
        None
    } else {
        Some(current_text)
    }
}

fn apply_patches(patches: &[Value], current_text: &mut String, capturing: &mut bool) -> bool {
    let mut applied = false;
    for patch in patches {
        if let Value::Object(map) = patch {
            if let (Some(path), Some(op), Some(value)) = (
                map.get("p").and_then(Value::as_str),
                map.get("o").and_then(Value::as_str),
                map.get("v"),
            ) {
                if apply_patch(path, op, value, current_text, capturing) {
                    applied = true;
                }
            }
        }
    }
    applied
}

fn apply_patch(
    path: &str,
    op: &str,
    value: &Value,
    current_text: &mut String,
    capturing: &mut bool,
) -> bool {
    if path == "/message/content/parts/0" {
        if let Some(text) = value.as_str() {
            match op {
                "append" => {
                    current_text.push_str(text);
                    *capturing = true;
                    return true;
                }
                "replace" => {
                    current_text.clear();
                    current_text.push_str(text);
                    *capturing = true;
                    return true;
                }
                _ => {}
            }
        }
    }
    false
}

fn extract_assistant_text(message: &Value) -> Option<String> {
    let role = message
        .get("author")
        .and_then(|author| author.get("role"))
        .and_then(Value::as_str);
    if role != Some("assistant") {
        return None;
    }

    let parts = message.get("content").and_then(|content| content.get("parts"));

    match parts {
        Some(Value::Array(items)) => {
            let mut joined = String::new();
            for (idx, part) in items.iter().enumerate() {
                if let Some(part) = part.as_str() {
                    if idx > 0 {
                        joined.push('\n');
                    }
                    joined.push_str(part);
                }
            }
            if joined.is_empty() {
                None
            } else {
                Some(joined)
            }
        }
        Some(Value::String(text)) => Some(text.clone()),
        _ => None,
    }
}

fn extract_code_blocks(text: &str) -> Vec<CodeBlock> {
    let mut blocks = Vec::new();
    let mut in_block = false;
    let mut current = String::new();
    let mut current_lang: Option<String> = None;

    for raw_line in text.lines() {
        let line = raw_line.trim_end_matches('\r');
        if line.starts_with("```") {
            if in_block {
                if current.is_empty() && line.trim() != "```" {
                    let lang = line.trim_start_matches("```").trim();
                    if !lang.is_empty() {
                        current_lang = Some(lang.to_string());
                    }
                    continue;
                }
                if current.ends_with('\n') {
                    current.pop();
                }
                if !current.is_empty() {
                    blocks.push(CodeBlock {
                        language: current_lang.take(),
                        content: current.clone(),
                    });
                }
                current.clear();
                in_block = false;
            } else {
                in_block = true;
                let lang = line.trim_start_matches("```").trim();
                current_lang = if lang.is_empty() {
                    None
                } else {
                    Some(lang.to_string())
                };
            }
            continue;
        }

        if in_block {
            current.push_str(line);
            current.push('\n');
        }
    }

    if in_block && !current.is_empty() {
        if current.ends_with('\n') {
            current.pop();
        }
        blocks.push(CodeBlock {
            language: current_lang,
            content: current,
        });
    }

    blocks
}

fn format_timestamp(ts_ms: i64) -> String {
    if ts_ms <= 0 {
        return "-".to_string();
    }
    match Local.timestamp_millis_opt(ts_ms).single() {
        Some(dt) => dt.format("%Y-%m-%d-%I-%M-%p").to_string(),
        None => ts_ms.to_string(),
    }
}

#[derive(Clone)]
struct MessageMeta {
    seq_label: String,
    branch_suffix: String,
}

fn build_message_meta(
    json_files: &[PathBuf],
) -> Result<HashMap<String, MessageMeta>, Box<dyn std::error::Error>> {
    let mut messages: HashMap<String, Vec<MessageMetaEntry>> = HashMap::new();

    for path in json_files {
        let contents = fs::read_to_string(path)?;
        let json: Value = serde_json::from_str(&contents)?;
        let conversation_id = match json.get("conversationId").and_then(Value::as_str) {
            Some(value) => value.to_string(),
            None => continue,
        };
        let message_id = match json.get("messageId").and_then(Value::as_str) {
            Some(value) => value.to_string(),
            None => continue,
        };
        let timestamp = json.get("timestamp").and_then(Value::as_i64).unwrap_or(0);
        let parent_id = extract_parent_id(&json);

        messages
            .entry(conversation_id)
            .or_default()
            .push(MessageMetaEntry {
                message_id,
                parent_id,
                timestamp,
            });
    }

    let mut meta_map = HashMap::new();

    for (conversation_id, mut entries) in messages {
        entries.sort_by_key(|entry| entry.timestamp);
        let mut order_map = HashMap::new();
        for (idx, entry) in entries.iter().enumerate() {
            let seq_label = format!("{:02}", idx + 1);
            order_map.insert(entry.message_id.clone(), seq_label);
        }

        let mut children_map: HashMap<String, Vec<String>> = HashMap::new();
        for entry in &entries {
            if let Some(parent) = &entry.parent_id {
                children_map
                    .entry(parent.clone())
                    .or_default()
                    .push(entry.message_id.clone());
            }
        }

        let mut branch_map: HashMap<String, String> = HashMap::new();
        for children in children_map.values_mut() {
            children.sort_by_key(|id| {
                entries
                    .iter()
                    .find(|e| &e.message_id == id)
                    .map(|e| e.timestamp)
                    .unwrap_or(0)
            });
            for (idx, child_id) in children.iter().enumerate() {
                branch_map.insert(child_id.clone(), format!("-{}", branch_label(idx)));
            }
        }

        for entry in entries {
            let seq_label = order_map
                .get(&entry.message_id)
                .cloned()
                .unwrap_or_else(|| "00".to_string());
            let branch_suffix = branch_map
                .get(&entry.message_id)
                .cloned()
                .unwrap_or_default();
            let key = format!("{}/{}", conversation_id, entry.message_id);
            meta_map.insert(
                key,
                MessageMeta {
                    seq_label,
                    branch_suffix,
                },
            );
        }
    }

    Ok(meta_map)
}

fn branch_label(idx: usize) -> String {
    let mut n = idx;
    let mut label = String::new();
    loop {
        let ch = ((n % 26) as u8 + b'A') as char;
        label.insert(0, ch);
        if n < 26 {
            break;
        }
        n = n / 26 - 1;
    }
    label
}

struct MessageMetaEntry {
    message_id: String,
    parent_id: Option<String>,
    timestamp: i64,
}

fn extract_parent_id(json: &Value) -> Option<String> {
    let events = json.get("sseEvents")?.as_array()?;
    for event in events {
        if let Value::Object(map) = event {
            if let Some(message) = map
                .get("message")
                .or_else(|| map.get("v").and_then(|v| v.get("message")))
            {
                if let Some(parent_id) = message
                    .get("metadata")
                    .and_then(|meta| meta.get("parent_id"))
                    .and_then(Value::as_str)
                {
                    return Some(parent_id.to_string());
                }
            }
            if let Some(parent_id) = map
                .get("input_message")
                .and_then(|msg| msg.get("metadata"))
                .and_then(|meta| meta.get("parent_id"))
                .and_then(Value::as_str)
            {
                return Some(parent_id.to_string());
            }
        }
    }
    None
}

fn classify_block(block: &CodeBlock) -> BlockType {
    if let Some(lang) = block.language.as_deref() {
        let lang = lang.to_ascii_lowercase();
        if lang.contains("diff") || lang.contains("patch") {
            return BlockType::Patch;
        }
        if lang == "sh" || lang == "bash" || lang.contains("shell") {
            return BlockType::Shell;
        }
        if lang == "plan" {
            return BlockType::Plan;
        }
        if lang == "rs" || lang == "rust" {
            return BlockType::Rust;
        }
        if lang == "py" || lang == "python" {
            return BlockType::Python;
        }
        if lang == "jl" || lang == "julia" {
            return BlockType::Julia;
        }
        if lang == "json" {
            return BlockType::Json;
        }
    }

    if block.content.starts_with("*** Begin Patch") {
        return BlockType::Patch;
    }
    if block.content.starts_with("#!") {
        return BlockType::Shell;
    }
    if block.content.trim_start().starts_with('{') || block.content.trim_start().starts_with('[') {
        return BlockType::Json;
    }

    BlockType::Other
}

fn scan_existing_counts(base_path: &Path) -> HashMap<String, Counters> {
    let mut counters: HashMap<String, Counters> = HashMap::new();
    for entry in WalkDir::new(base_path)
        .follow_links(true)
        .into_iter()
        .filter_map(|entry| entry.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.components().any(|c| c.as_os_str() == "codeblock_extractor") {
            continue;
        }
        let file_name = match path.file_name().and_then(|s| s.to_str()) {
            Some(name) => name,
            None => continue,
        };

        let conversation_id = match path.strip_prefix(base_path).ok().and_then(|p| p.components().next()) {
            Some(component) => component.as_os_str().to_string_lossy().to_string(),
            None => continue,
        };

        if let Some((value, branch)) = parse_ordered_with_branch(file_name, "patch_", "intent", ".json") {
            let key = format!("{}::{}", conversation_id, branch);
            let counter = counters.entry(key).or_insert_with(Counters::default);
            counter.patch = counter.patch.max(value);
        } else if let Some((value, branch)) = parse_ordered_with_branch(file_name, "shell_", "intent", ".json") {
            let key = format!("{}::{}", conversation_id, branch);
            let counter = counters.entry(key).or_insert_with(Counters::default);
            counter.shell = counter.shell.max(value);
        } else if let Some((value, branch)) = parse_ordered_with_branch(file_name, "plan_", "intent", ".json") {
            let key = format!("{}::{}", conversation_id, branch);
            let counter = counters.entry(key).or_insert_with(Counters::default);
            counter.plan = counter.plan.max(value);
        } else if let Some((value, branch)) = parse_index_with_branch(file_name, "rustfile_", ".rs") {
            let key = format!("{}::{}", conversation_id, branch);
            let counter = counters.entry(key).or_insert_with(Counters::default);
            counter.rust = counter.rust.max(value);
        } else if let Some((value, branch)) = parse_index_with_branch(file_name, "pythonfile_", ".py") {
            let key = format!("{}::{}", conversation_id, branch);
            let counter = counters.entry(key).or_insert_with(Counters::default);
            counter.python = counter.python.max(value);
        } else if let Some((value, branch)) = parse_index_with_branch(file_name, "juliafile_", ".jl") {
            let key = format!("{}::{}", conversation_id, branch);
            let counter = counters.entry(key).or_insert_with(Counters::default);
            counter.julia = counter.julia.max(value);
        } else if let Some((value, branch)) = parse_index_with_branch(file_name, "jsonfile_", ".json") {
            let key = format!("{}::{}", conversation_id, branch);
            let counter = counters.entry(key).or_insert_with(Counters::default);
            counter.json = counter.json.max(value);
        }
    }
    counters
}

fn parse_index_with_branch(file_name: &str, prefix: &str, suffix: &str) -> Option<(usize, String)> {
    if !file_name.starts_with(prefix) || !file_name.ends_with(suffix) {
        return None;
    }
    let number = file_name
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?;
    let mut parts = number.splitn(2, '-');
    let num = parts.next()?;
    let branch = parts.next().unwrap_or("").to_string();
    let value = num.parse::<usize>().ok()?;
    Some((value, branch))
}

fn parse_ordered_with_branch(
    file_name: &str,
    prefix: &str,
    stage: &str,
    suffix: &str,
) -> Option<(usize, String)> {
    if !file_name.starts_with(prefix) || !file_name.ends_with(suffix) {
        return None;
    }
    let stem = file_name.strip_suffix(suffix)?;
    let after_prefix = stem.strip_prefix(prefix)?;
    let mut parts = after_prefix.splitn(3, '_');
    let _stage = parts.next()?;
    let label = parts.next()?;
    if label != stage {
        return None;
    }
    let seq_and_branch = parts.next()?;
    let mut seq_parts = seq_and_branch.splitn(2, '-');
    let seq = seq_parts.next()?;
    let branch = seq_parts.next().unwrap_or("").to_string();
    let value = seq.parse::<usize>().ok()?;
    Some((value, branch))
}

fn has_existing_codeblocks(dir: &Path) -> bool {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = match name.to_str() {
                Some(name) => name,
                None => continue,
            };
            if name.starts_with("patch_")
                || name.starts_with("shell_")
                || name.starts_with("plan_")
                || name.starts_with("rustfile_")
                || name.starts_with("pythonfile_")
                || name.starts_with("juliafile_")
                || name.starts_with("jsonfile_")
            {
                return true;
            }
        }
    }
    false
}

fn cleanup_message_dir(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(name) => name,
                None => continue,
            };
            if name.starts_with("message_") && name.ends_with(".md") {
                fs::remove_file(path)?;
                continue;
            }
            if name == "message.md" {
                fs::remove_file(path)?;
                continue;
            }
            if name.ends_with(".patch")
                || name.starts_with("patch_")
                || name.starts_with("shell_")
                || name.starts_with("plan_")
                || name.starts_with("shellfile_")
                || name.starts_with("rustfile_")
                || name.starts_with("pythonfile_")
                || name.starts_with("juliafile_")
                || name.starts_with("jsonfile_")
                || name.ends_with(".response")
            {
                fs::remove_file(path)?;
            }
        }
    }
    Ok(())
}

fn apply_suffix(file_name: &str, suffix: &str) -> String {
    if suffix.is_empty() {
        return file_name.to_string();
    }
    if let Some((base, ext)) = file_name.rsplit_once('.') {
        format!("{}{}.{ext}", base, suffix)
    } else {
        format!("{}{}", file_name, suffix)
    }
}

fn write_shell_intent_at_path(
    intent_path: &Path,
    output_dir: &Path,
    content: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let cwd_value = read_cwd_json(output_dir);
    if intent_path.exists() {
        if let Some(cwd) = cwd_value.clone() {
            if needs_cwd_rewrite(intent_path)? {
                let intent_json = build_shell_intent_json(content, Some(cwd))?;
                fs::write(intent_path, intent_json.as_bytes())?;
            }
        }
        return Ok(());
    }
    let intent_json = build_shell_intent_json(content, cwd_value)?;
    fs::write(intent_path, intent_json.as_bytes())?;
    Ok(())
}

fn write_plan_intent_at_path(
    intent_path: &Path,
    content: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if intent_path.exists() {
        return Ok(());
    }
    let plan = parse_plan_block(content)?;
    let intent_json = build_plan_intent_json(&plan)?;
    fs::write(intent_path, intent_json.as_bytes())?;
    Ok(())
}

struct PlanStep {
    status: String,
    step: String,
}

struct PlanBlock {
    explanation: Option<String>,
    steps: Vec<PlanStep>,
}

fn parse_plan_block(content: &str) -> Result<PlanBlock, Box<dyn std::error::Error>> {
    let mut explanation: Option<String> = None;
    let mut steps = Vec::new();

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if line.to_ascii_uppercase().starts_with("EXPLANATION:") {
            let rest = line.splitn(2, ':').nth(1).unwrap_or("").trim();
            if !rest.is_empty() {
                explanation = Some(rest.to_string());
            }
            continue;
        }
        let mut parts = line.splitn(2, '.');
        let _index = parts.next().unwrap_or("").trim();
        let rest = parts.next().unwrap_or("").trim();
        if rest.is_empty() {
            continue;
        }
        let status_start = rest.find('[').unwrap_or(0);
        let status_end = rest.find(']').unwrap_or(0);
        if status_start == status_end {
            continue;
        }
        let status_raw = rest[status_start + 1..status_end].trim().to_ascii_lowercase();
        let status = match status_raw.as_str() {
            "pending" => "pending",
            "in_progress" | "in progress" => "in_progress",
            "completed" => "completed",
            _ => continue,
        };
        let step = rest[status_end + 1..].trim();
        if step.is_empty() {
            continue;
        }
        steps.push(PlanStep {
            status: status.to_string(),
            step: step.to_string(),
        });
    }

    if steps.is_empty() {
        return Err("Plan block has no steps".into());
    }

    Ok(PlanBlock { explanation, steps })
}

fn build_plan_intent_json(plan: &PlanBlock) -> Result<String, Box<dyn std::error::Error>> {
    let timestamp = Local::now().to_rfc3339();
    let plan_steps: Vec<Value> = plan
        .steps
        .iter()
        .map(|step| {
            Value::Object(
                [
                    ("status".to_string(), Value::String(step.status.clone())),
                    ("step".to_string(), Value::String(step.step.clone())),
                ]
                .into_iter()
                .collect(),
            )
        })
        .collect();

    let mut intent = BTreeMap::new();
    intent.insert("schema".to_string(), Value::String("plan.intent.v1".to_string()));
    intent.insert(
        "intent_class".to_string(),
        Value::String("update_plan".to_string()),
    );
    intent.insert(
        "intent_id".to_string(),
        Value::String("sha256:__DERIVED__".to_string()),
    );
    intent.insert("timestamp".to_string(), Value::String(timestamp));
    intent.insert(
        "actor".to_string(),
        Value::Object(
            [
                ("type".to_string(), Value::String("agent".to_string())),
                ("id".to_string(), Value::String("chatgpt".to_string())),
            ]
            .into_iter()
            .collect(),
        ),
    );
    intent.insert(
        "summary".to_string(),
        Value::String("Plan update".to_string()),
    );
    if let Some(explanation) = plan.explanation.as_ref() {
        intent.insert(
            "explanation".to_string(),
            Value::String(explanation.clone()),
        );
    } else {
        intent.insert("explanation".to_string(), Value::Null);
    }
    intent.insert("plan".to_string(), Value::Array(plan_steps));
    intent.insert("schema_version".to_string(), Value::Number(1.into()));

    let canonical = serde_json::to_string(&intent)?;
    let hash = sha256_hex(canonical.as_bytes());
    intent.insert(
        "intent_id".to_string(),
        Value::String(format!("sha256:{hash}")),
    );
    Ok(serde_json::to_string_pretty(&intent)?)
}

fn needs_cwd_rewrite(intent_path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(intent_path)?;
    let json: Value = serde_json::from_str(&contents)?;
    Ok(json.get("cwd").map(Value::is_null).unwrap_or(true))
}

fn rewrite_shell_intents_with_cwd(
    dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let cwd_value = match read_cwd_json(dir) {
        Some(value) => value,
        None => return Ok(()),
    };
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = match path.file_name().and_then(|s| s.to_str()) {
            Some(name) => name,
            None => continue,
        };
        if !name.starts_with("shell_01_intent_") || !name.ends_with(".json") {
            continue;
        }
        let contents = fs::read_to_string(&path)?;
        let json: Value = serde_json::from_str(&contents)?;
        if !json.get("cwd").map(Value::is_null).unwrap_or(true) {
            continue;
        }
        let obj = match json.as_object() {
            Some(obj) => obj,
            None => continue,
        };
        let command = match obj.get("command").and_then(Value::as_array) {
            Some(command) => command.clone(),
            None => continue,
        };
        let intent_class = obj
            .get("intent_class")
            .cloned()
            .unwrap_or(Value::String("shell_command".to_string()));
        let constraints = obj.get("constraints").cloned().unwrap_or(Value::Null);
        let declared_inputs = obj.get("declared_inputs").cloned().unwrap_or(Value::Null);
        let declared_outputs = obj.get("declared_outputs").cloned().unwrap_or(Value::Null);
        let delta_type = obj
            .get("delta_type")
            .and_then(Value::as_str)
            .unwrap_or("shell.intent.v1");
        let env = obj.get("env").cloned().unwrap_or(Value::Null);

        let mut intent = BTreeMap::new();
        intent.insert("command".to_string(), Value::Array(command));
        intent.insert("intent_class".to_string(), intent_class);
        intent.insert("constraints".to_string(), constraints);
        intent.insert("cwd".to_string(), Value::String(cwd_value.clone()));
        intent.insert("declared_inputs".to_string(), declared_inputs);
        intent.insert("declared_outputs".to_string(), declared_outputs);
        intent.insert("delta_type".to_string(), Value::String(delta_type.to_string()));
        intent.insert("env".to_string(), env);
        intent.insert(
            "hash".to_string(),
            Value::String("sha256:__DERIVED__".to_string()),
        );

        let canonical = serde_json::to_string(&intent)?;
        let hash = sha256_hex(canonical.as_bytes());
        intent.insert("hash".to_string(), Value::String(format!("sha256:{hash}")));
        let updated = serde_json::to_string_pretty(&intent)?;
        fs::write(&path, updated.as_bytes())?;
    }
    Ok(())
}

fn fix_shell_intents_in_tree(base_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut stack = vec![base_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(err) => {
                if err.kind() == std::io::ErrorKind::PermissionDenied {
                    continue;
                }
                return Err(err.into());
            }
        };

        let mut has_cwd = false;
        let mut has_shell_intent = false;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.file_name().and_then(|s| s.to_str()) == Some("invariant_01_intent_01.json") {
                has_cwd = true;
            }
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|name| name.starts_with("shell_01_intent_") && name.ends_with(".json"))
                .unwrap_or(false)
            {
                has_shell_intent = true;
            }
        }

        if has_cwd && has_shell_intent {
            rewrite_shell_intents_with_cwd(&dir)?;
        }
    }
    Ok(())
}

fn read_cwd_json(dir: &Path) -> Option<String> {
    let path = dir.join("invariant_01_intent_01.json");
    let contents = fs::read_to_string(path).ok()?;
    let json: Value = serde_json::from_str(&contents).ok()?;
    json.get("binding")
        .and_then(|binding| binding.get("active_cwd"))
        .and_then(Value::as_str)
        .map(|s| s.to_string())
}

fn build_shell_intent_json(
    content: &str,
    cwd: Option<String>,
) -> Result<String, Box<dyn std::error::Error>> {
    let command = parse_shell_command(content);
    let mut intent = BTreeMap::new();
    intent.insert(
        "command".to_string(),
        Value::Array(command.into_iter().map(Value::String).collect()),
    );
    intent.insert(
        "intent_class".to_string(),
        Value::String("shell_command".to_string()),
    );
    intent.insert("constraints".to_string(), Value::Null);
    intent.insert(
        "cwd".to_string(),
        cwd.map(Value::String).unwrap_or(Value::Null),
    );
    intent.insert("declared_inputs".to_string(), Value::Null);
    intent.insert("declared_outputs".to_string(), Value::Null);
    intent.insert(
        "delta_type".to_string(),
        Value::String("shell.intent.v1".to_string()),
    );
    intent.insert("env".to_string(), Value::Null);
    intent.insert(
        "hash".to_string(),
        Value::String("sha256:__DERIVED__".to_string()),
    );

    let canonical = serde_json::to_string(&intent)?;
    let hash = sha256_hex(canonical.as_bytes());
    intent.insert("hash".to_string(), Value::String(format!("sha256:{hash}")));
    Ok(serde_json::to_string_pretty(&intent)?)
}

fn parse_shell_command(content: &str) -> Vec<String> {
    let cleaned = sanitize_shell_block_content(content);
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        return vec![];
    }
    let lines: Vec<&str> = trimmed
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect();
    if lines.len() <= 1 {
        return lines
            .first()
            .map(|line| line.split_whitespace().map(|s| s.to_string()).collect())
            .unwrap_or_default();
    }
    vec!["sh".to_string(), "-lc".to_string(), trimmed.to_string()]
}

fn sanitize_shell_block_content(content: &str) -> String {
    let mut sanitized = Vec::new();
    for line in content.lines() {
        if let Some((before, _)) = line.split_once("```") {
            let trimmed = before.trim_end();
            if !trimmed.is_empty() {
                sanitized.push(trimmed.to_string());
            }
            continue;
        }
        let trimmed = line.trim_end();
        if !trimmed.is_empty() {
            sanitized.push(trimmed.to_string());
        }
    }
    sanitized.join("\n")
}

fn write_patch_artifacts_at_path(
    delta_path: &Path,
    content: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if delta_path.exists() {
        return Ok(());
    }
    let (op, path) = extract_patch_op_and_path(content);
    let timestamp = Local::now().to_rfc3339();
    let delta_json = build_patch_delta_json(content, op, path, &timestamp)?;
    fs::write(delta_path, delta_json.as_bytes())?;

    let delta_id = extract_delta_id(&delta_json)?;
    let intent_json = build_patch_intent_json(&delta_id, &timestamp)?;
    let verify_json = build_patch_verify_json(&delta_id, &timestamp)?;
    let approval_json = build_patch_approval_json(&delta_id, &timestamp)?;
    let execution_json = build_patch_execution_json(&delta_id, &timestamp)?;

    let intent_path = sibling_patch_path(delta_path, "patch_01_intent_")?;
    let approval_path = sibling_patch_path(delta_path, "patch_02_approval_")?;
    let verify_path = sibling_patch_path(delta_path, "patch_03_verify_")?;
    let execution_path = sibling_patch_path(delta_path, "patch_04_execution_")?;

    fs::write(intent_path, intent_json.as_bytes())?;
    fs::write(approval_path, approval_json.as_bytes())?;
    fs::write(verify_path, verify_json.as_bytes())?;
    fs::write(execution_path, execution_json.as_bytes())?;
    Ok(())
}

fn extract_patch_op_and_path(content: &str) -> (String, String) {
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("*** Add File: ") {
            return ("add".to_string(), rest.trim().to_string());
        }
        if let Some(rest) = line.strip_prefix("*** Update File: ") {
            return ("update".to_string(), rest.trim().to_string());
        }
        if let Some(rest) = line.strip_prefix("*** Delete File: ") {
            return ("delete".to_string(), rest.trim().to_string());
        }
    }
    ("update".to_string(), "PATCH".to_string())
}

fn build_patch_delta_json(
    patch_text: &str,
    op: String,
    path: String,
    timestamp: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut delta = BTreeMap::new();
    delta.insert("schema".to_string(), Value::String("patch.delta.v1".to_string()));
    delta.insert(
        "delta_id".to_string(),
        Value::String("sha256:__DERIVED__".to_string()),
    );
    delta.insert("parent".to_string(), Value::Null);
    delta.insert(
        "timestamp".to_string(),
        Value::String(timestamp.to_string()),
    );
    delta.insert(
        "operations".to_string(),
        Value::Array(vec![Value::Object(
            [
                ("op".to_string(), Value::String(op)),
                ("path".to_string(), Value::String(path)),
                ("content".to_string(), Value::String(patch_text.to_string())),
                ("move_to".to_string(), Value::Null),
            ]
            .into_iter()
            .collect(),
        )]),
    );
    delta.insert("author".to_string(), Value::String("chatgpt".to_string()));
    delta.insert("schema_version".to_string(), Value::Number(1.into()));

    let canonical = serde_json::to_string(&delta)?;
    let hash = sha256_hex(canonical.as_bytes());
    delta.insert("delta_id".to_string(), Value::String(format!("sha256:{hash}")));
    Ok(serde_json::to_string_pretty(&delta)?)
}

fn build_patch_intent_json(delta_id: &str, timestamp: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut intent = BTreeMap::new();
    intent.insert("schema".to_string(), Value::String("patch.intent.v1".to_string()));
    intent.insert(
        "intent_class".to_string(),
        Value::String("patch_change".to_string()),
    );
    intent.insert("delta_ref".to_string(), Value::String(delta_id.to_string()));
    intent.insert(
        "intent_id".to_string(),
        Value::String("sha256:__DERIVED__".to_string()),
    );
    intent.insert("timestamp".to_string(), Value::String(timestamp.to_string()));
    intent.insert(
        "actor".to_string(),
        Value::Object(
            [
                ("type".to_string(), Value::String("agent".to_string())),
                ("id".to_string(), Value::String("chatgpt".to_string())),
            ]
            .into_iter()
            .collect(),
        ),
    );
    intent.insert(
        "summary".to_string(),
        Value::String("Patch from message".to_string()),
    );
    intent.insert("motivation".to_string(), Value::String(String::new()));
    intent.insert("constraints".to_string(), Value::Array(vec![]));
    intent.insert("schema_version".to_string(), Value::Number(1.into()));

    let canonical = serde_json::to_string(&intent)?;
    let hash = sha256_hex(canonical.as_bytes());
    intent.insert("intent_id".to_string(), Value::String(format!("sha256:{hash}")));
    Ok(serde_json::to_string_pretty(&intent)?)
}

fn build_patch_verify_json(delta_id: &str, timestamp: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut verify = BTreeMap::new();
    verify.insert("schema".to_string(), Value::String("patch.verify.v1".to_string()));
    verify.insert("intent_hash".to_string(), Value::String(delta_id.to_string()));
    verify.insert("timestamp".to_string(), Value::String(timestamp.to_string()));
    verify.insert("verified".to_string(), Value::String("n/a".to_string()));
    verify.insert("checks".to_string(), Value::Array(vec![]));
    verify.insert("errors".to_string(), Value::Array(vec![]));
    Ok(serde_json::to_string_pretty(&verify)?)
}

fn build_patch_approval_json(delta_id: &str, timestamp: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut approval = BTreeMap::new();
    approval.insert("schema".to_string(), Value::String("patch.approval.v1".to_string()));
    approval.insert("intent_hash".to_string(), Value::String(delta_id.to_string()));
    approval.insert("timestamp".to_string(), Value::String(timestamp.to_string()));
    approval.insert("approved".to_string(), Value::String("n/a".to_string()));
    approval.insert("reason".to_string(), Value::String("stage not executed".to_string()));
    Ok(serde_json::to_string_pretty(&approval)?)
}

fn build_patch_execution_json(
    _delta_id: &str,
    _timestamp: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut execution = BTreeMap::new();
    execution.insert("cwd".to_string(), Value::String("n/a".to_string()));
    execution.insert("command".to_string(), Value::String("n/a".to_string()));
    execution.insert("env".to_string(), Value::String("n/a".to_string()));
    Ok(serde_json::to_string_pretty(&execution)?)
}

fn sibling_patch_path(delta_path: &Path, prefix: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let file_name = delta_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("Invalid patch delta filename")?;
    let stripped = file_name
        .strip_prefix("patch_06_delta_")
        .and_then(|s| s.strip_suffix(".json"))
        .ok_or("Invalid patch delta filename")?;
    Ok(delta_path.with_file_name(format!("{prefix}{stripped}.json")))
}

fn extract_delta_id(delta_json: &str) -> Result<String, Box<dyn std::error::Error>> {
    let value: Value = serde_json::from_str(delta_json)?;
    Ok(value
        .get("delta_id")
        .and_then(Value::as_str)
        .unwrap_or("sha256:__UNKNOWN__")
        .to_string())
}

fn remove_legacy_patch_files(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|name| name.ends_with(".patch"))
                .unwrap_or(false)
            {
                fs::remove_file(path)?;
            }
        }
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{:02x}", byte));
    }
    out
}

fn branch_counter_key(conversation_id: &str, branch_suffix: &str) -> String {
    let branch = branch_suffix.trim_start_matches('-');
    format!("{}::{}", conversation_id, branch)
}
