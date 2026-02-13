impl Drop for RunReceipt {
    fn drop(&mut self) {
        // Optional: future logging hook.
    }
}

#[derive(Debug, Clone)]
struct CommandOutcome {
    state_hash: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct LedgerEntry {
    shell_id: ShellId,
    epoch: u64,
    command: String,
    state_hash: String,
}

fn append_ledger(path: &Path, entry: &LedgerEntry) -> Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", serde_json::to_string(entry)?)?;
    Ok(())
}

fn read_ledger(path: &Path) -> Result<Vec<LedgerEntry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut entries = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: LedgerEntry = serde_json::from_str(&line)?;
        entries.push(entry);
    }
    Ok(entries)
}

fn rewrite_ledger(path: &Path, entries: &[LedgerEntry]) -> Result<()> {
    let mut file = File::create(path)?;
    for entry in entries {
        writeln!(file, "{}", serde_json::to_string(entry)?)?;
    }
    Ok(())
}

fn shell_id_from_filename(path: &Path) -> Option<ShellId> {
    let name = path.file_name()?.to_str()?;
    if !name.starts_with("shell_") || !name.ends_with(".ledger") {
        return None;
    }
    let id_part = &name["shell_".len()..name.len() - ".ledger".len()];
    id_part.parse::<u64>().ok()
}

#[derive(Debug, Serialize, Deserialize)]
struct ShellState {
    cwd: String,
    env: BTreeMap<String, String>,
    aliases: BTreeMap<String, String>,
    options: BTreeMap<String, bool>,
    ulimit: String,
}

fn parse_capture(output: &str) -> Result<ShellState> {
    let start = output
        .find(CAPTURE_BEGIN)
        .ok_or_else(|| anyhow!("capture block not found"))?;
    let end = output[start..]
        .find(CAPTURE_END)
        .ok_or_else(|| anyhow!("capture terminator missing"))?
        + start;
    let capture = &output[start + CAPTURE_BEGIN.len()..end];
    let mut section = "";
    let mut pwd = String::new();
    let mut env_lines = Vec::new();
    let mut alias_lines = Vec::new();
    let mut option_lines = Vec::new();
    let mut ulimit_lines = Vec::new();
    for line in capture.lines() {
        match line {
            SECTION_PWD => {
                section = SECTION_PWD;
                continue;
            }
            SECTION_ENV => {
                section = SECTION_ENV;
                continue;
            }
            SECTION_ALIAS => {
                section = SECTION_ALIAS;
                continue;
            }
            SECTION_OPTIONS => {
                section = SECTION_OPTIONS;
                continue;
            }
            SECTION_ULIMIT => {
                section = SECTION_ULIMIT;
                continue;
            }
            "" => continue,
            _ => {}
        }
        match section {
            SECTION_PWD => pwd = line.trim().to_string(),
            SECTION_ENV => env_lines.push(line.to_string()),
            SECTION_ALIAS => alias_lines.push(line.to_string()),
            SECTION_OPTIONS => option_lines.push(line.to_string()),
            SECTION_ULIMIT => ulimit_lines.push(line.to_string()),
            _ => {}
        }
    }
    let env = parse_declares(&env_lines);
    let aliases = parse_aliases(&alias_lines);
    let options = parse_shopts(&option_lines);
    let ulimit = ulimit_lines.join("\n").trim().to_string();
    Ok(ShellState {
        cwd: pwd,
        env,
        aliases,
        options,
        ulimit,
    })
}

fn parse_declares(lines: &[String]) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for line in lines {
        let line = line.trim();
        if !line.starts_with("declare -x ") {
            continue;
        }
        let rest = &line["declare -x ".len()..];
        let mut parts = rest.splitn(2, '=');
        if let Some(key) = parts.next() {
            let value = parts
                .next()
                .map(|v| unquote(v.trim()))
                .unwrap_or_else(|| String::new());
            map.insert(key.to_string(), value);
        }
    }
    map
}

fn parse_aliases(lines: &[String]) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for line in lines {
        let line = line.trim();
        if !line.starts_with("alias ") {
            continue;
        }
        let rest = &line["alias ".len()..];
        if let Some((name, value)) = rest.split_once('=') {
            map.insert(name.to_string(), unquote(value.trim()));
        }
    }
    map
}

fn parse_shopts(lines: &[String]) -> BTreeMap<String, bool> {
    let mut map = BTreeMap::new();
    for line in lines {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("shopt -p ") {
            map.insert(rest.to_string(), true);
        } else if let Some(rest) = line.strip_prefix("shopt -u ") {
            map.insert(rest.to_string(), false);
        }
    }
    map
}

fn unquote(value: &str) -> String {
    if value.starts_with('\'') && value.ends_with('\'') && value.len() >= 2 {
        value[1..value.len() - 1].to_string()
    } else if value.starts_with('"') && value.ends_with('"') && value.len() >= 2 {
        let inner = &value[1..value.len() - 1];
        inner.replace("\\\"", "\"").replace("\\\\", "\\")
    } else {
        value.to_string()
    }
}

fn hash_state(state: &ShellState) -> Result<String> {
    let json = serde_json::to_vec(state)?;
    Ok(blake3_hash(&json).to_hex().to_string())
}
