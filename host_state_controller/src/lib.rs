use anyhow::{anyhow, Result};
use blake3::hash as blake3_hash;
use portable_pty::{native_pty_system, Child, CommandBuilder, PtySize};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

pub type ShellId = u64;

const PROMPT_PREFIX: &str = "__HOSTCTL_PROMPT__";
const CAPTURE_BEGIN: &str = "__HOSTCTL_CAPTURE_BEGIN__";
const CAPTURE_END: &str = "__HOSTCTL_CAPTURE_END__";
const SECTION_PWD: &str = "__HOSTCTL_SECTION_PWD__";
const SECTION_ENV: &str = "__HOSTCTL_SECTION_ENV__";
const SECTION_ALIAS: &str = "__HOSTCTL_SECTION_ALIAS__";
const SECTION_OPTIONS: &str = "__HOSTCTL_SECTION_OPTIONS__";
const SECTION_ULIMIT: &str = "__HOSTCTL_SECTION_ULIMIT__";

fn capture_block() -> String {
    format!(
        "printf '\n{begin}\n'; \
         printf '{pwd}\n'; pwd; \
         printf '{env}\n'; declare -px; \
         printf '{alias}\n'; alias -p; \
         printf '{opts}\n'; shopt -p; \
         printf '{ul}\n'; ulimit -a; \
         printf '{end}\n';",
        begin = CAPTURE_BEGIN,
        pwd = SECTION_PWD,
        env = SECTION_ENV,
        alias = SECTION_ALIAS,
        opts = SECTION_OPTIONS,
        ul = SECTION_ULIMIT,
        end = CAPTURE_END
    )
}

#[derive(Clone)]
struct ShellNode {
    parent: Option<ShellId>,
    epoch: u64,
    state_hash: Option<String>,
    ledger_path: PathBuf,
}

pub struct StateController {
    shells: HashMap<ShellId, ShellHandle>,
    nodes: HashMap<ShellId, ShellNode>,
    next_shell_id: ShellId,
    state_dir: PathBuf,
}

impl StateController {
    pub fn new(state_dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = state_dir.into();
        fs::create_dir_all(&dir)?;
        let mut controller = Self {
            shells: HashMap::new(),
            nodes: HashMap::new(),
            next_shell_id: 1,
            state_dir: dir,
        };
        controller.reload_from_disk()?;
        Ok(controller)
    }

    fn reload_from_disk(&mut self) -> Result<()> {
        if !self.state_dir.exists() {
            return Ok(());
        }
        for entry in fs::read_dir(&self.state_dir)? {
            let entry = entry?;
            let path = entry.path();
            let Some(shell_id) = shell_id_from_filename(&path) else {
                continue;
            };
            let entries = read_ledger(&path)?;
            let (epoch, hash) = entries
                .last()
                .map(|entry| (entry.epoch, Some(entry.state_hash.clone())))
                .unwrap_or((0, None));
            self.nodes.insert(
                shell_id,
                ShellNode {
                    parent: None,
                    epoch,
                    state_hash: hash,
                    ledger_path: path.clone(),
                },
            );
            self.next_shell_id = self.next_shell_id.max(shell_id.saturating_add(1));
        }
        Ok(())
    }

    pub fn spawn_root(&mut self) -> Result<ShellId> {
        let id = self.next_shell_id;
        self.next_shell_id += 1;
        let ledger_path = self.state_dir.join(format!("shell_{id}.ledger"));
        File::create(&ledger_path)?;
        let handle = ShellHandle::spawn(id)?;
        let node = ShellNode {
            parent: None,
            epoch: 0,
            state_hash: None,
            ledger_path,
        };
        self.shells.insert(id, handle);
        self.nodes.insert(id, node);
        Ok(id)
    }

    pub fn branch(&mut self, parent: ShellId, epoch: u64) -> Result<ShellId> {
        let parent_node = self
            .nodes
            .get(&parent)
            .ok_or_else(|| anyhow!("parent shell {parent} not found"))?
            .clone();
        let entries = read_ledger(&parent_node.ledger_path)?;
        let subset: Vec<LedgerEntry> = entries.into_iter().filter(|e| e.epoch <= epoch).collect();
        if epoch > 0 && subset.last().map(|e| e.epoch) != Some(epoch) {
            return Err(anyhow!("parent shell lacks epoch {epoch}"));
        }
        let target_hash = subset.last().map(|e| e.state_hash.clone());
        let child_id = self.spawn_root()?;
        let ledger_path = self
            .nodes
            .get(&child_id)
            .expect("missing child node")
            .ledger_path
            .clone();
        if let Some(handle) = self.shells.remove(&child_id) {
            handle.terminate();
        }
        let mut handle = ShellHandle::spawn(child_id)?;
        let mut replayed = Vec::new();
        for entry in &subset {
            let outcome = handle.execute(&entry.command)?;
            if outcome.state_hash != entry.state_hash {
                return Err(anyhow!(
                    "replay mismatch epoch {} expected {} got {}",
                    entry.epoch,
                    entry.state_hash,
                    outcome.state_hash
                ));
            }
            replayed.push(LedgerEntry {
                shell_id: child_id,
                epoch: entry.epoch,
                command: entry.command.clone(),
                state_hash: entry.state_hash.clone(),
            });
        }
        rewrite_ledger(&ledger_path, &replayed)?;
        handle.epoch = epoch;
        self.shells.insert(child_id, handle);
        if let Some(node) = self.nodes.get_mut(&child_id) {
            node.parent = Some(parent);
            node.epoch = epoch;
            node.state_hash = target_hash.clone();
        }
        Ok(child_id)
    }

    pub fn run(&mut self, shell_id: ShellId, command: &str) -> Result<RunReceipt> {
        self.ensure_shell_handle(shell_id)?;
        let handle = self
            .shells
            .get_mut(&shell_id)
            .ok_or_else(|| anyhow!("shell {shell_id} not found"))?;
        let outcome = handle.execute(command)?;
        handle.epoch = handle.epoch.saturating_add(1);
        let epoch = handle.epoch;
        let node = self
            .nodes
            .get_mut(&shell_id)
            .ok_or_else(|| anyhow!("node {shell_id} missing"))?;
        node.epoch = epoch;
        node.state_hash = Some(outcome.state_hash.clone());
        Ok(RunReceipt {
            epoch,
            state_hash: outcome.state_hash,
            shell_id,
            command: command.to_string(),
        })
    }

    /// Persist a previously returned [`RunReceipt`] into this shell's ledger.
    ///
    /// This lets the caller decide when ledger entries become durable so that
    /// higher-level commit logic (Canon + Lean) can gate writes before they
    /// appear in the cache.
    pub fn persist_receipt(&mut self, receipt: &RunReceipt) -> Result<()> {
        let ledger_path = self.ledger_path_for(receipt.shell_id);

        let entry = LedgerEntry {
            shell_id: receipt.shell_id,
            epoch: receipt.epoch,
            command: receipt.command.clone(),
            state_hash: receipt.state_hash.clone(),
        };
        append_ledger(&ledger_path, &entry)?;

        let node = self
            .nodes
            .entry(receipt.shell_id)
            .or_insert_with(|| ShellNode {
                parent: None,
                epoch: receipt.epoch,
                state_hash: Some(receipt.state_hash.clone()),
                ledger_path: ledger_path.clone(),
            });
        node.epoch = receipt.epoch;
        node.state_hash = Some(receipt.state_hash.clone());

        Ok(())
    }

    pub fn restore(&mut self, shell_id: ShellId, epoch: u64) -> Result<()> {
        let node = self
            .nodes
            .get(&shell_id)
            .ok_or_else(|| anyhow!("shell {shell_id} not found"))?
            .clone();
        let entries = read_ledger(&node.ledger_path)?;
        let subset: Vec<LedgerEntry> = entries.into_iter().filter(|e| e.epoch <= epoch).collect();
        if epoch > 0 && subset.last().map(|e| e.epoch) != Some(epoch) {
            return Err(anyhow!("shell lacks epoch {epoch}"));
        }
        if let Some(handle) = self.shells.remove(&shell_id) {
            handle.terminate();
        }
        let mut handle = ShellHandle::spawn(shell_id)?;
        for entry in &subset {
            let outcome = handle.execute(&entry.command)?;
            if outcome.state_hash != entry.state_hash {
                return Err(anyhow!(
                    "restore mismatch epoch {} expected {} got {}",
                    entry.epoch,
                    entry.state_hash,
                    outcome.state_hash
                ));
            }
        }
        rewrite_ledger(&node.ledger_path, &subset)?;
        handle.epoch = epoch;
        self.shells.insert(shell_id, handle);
        if let Some(node_mut) = self.nodes.get_mut(&shell_id) {
            node_mut.epoch = epoch;
            node_mut.state_hash = subset.last().map(|e| e.state_hash.clone());
        }
        Ok(())
    }

    pub fn query_hash(&self, shell_id: ShellId) -> Option<(u64, String)> {
        self.nodes
            .get(&shell_id)
            .and_then(|node| node.state_hash.clone().map(|hash| (node.epoch, hash)))
    }

    pub fn latest_epoch(&self, shell_id: ShellId) -> Option<u64> {
        self.nodes.get(&shell_id).map(|node| node.epoch)
    }

    pub fn ensure_shell_registered(&self, shell_id: ShellId) -> Result<()> {
        if self.nodes.contains_key(&shell_id) {
            Ok(())
        } else {
            Err(anyhow!("shell {shell_id} not found"))
        }
    }

    fn ensure_shell_handle(&mut self, shell_id: ShellId) -> Result<()> {
        if self.shells.contains_key(&shell_id) {
            return Ok(());
        }
        let node = self
            .nodes
            .get(&shell_id)
            .ok_or_else(|| anyhow!("shell {shell_id} not found"))?
            .clone();
        let mut handle = ShellHandle::spawn(shell_id)?;
        if node.epoch > 0 {
            let entries = read_ledger(&node.ledger_path)?;
            for entry in &entries {
                let outcome = handle.execute(&entry.command)?;
                if outcome.state_hash != entry.state_hash {
                    return Err(anyhow!(
                        "replay mismatch epoch {} expected {} got {}",
                        entry.epoch,
                        entry.state_hash,
                        outcome.state_hash
                    ));
                }
            }
            handle.epoch = node.epoch;
        }
        self.shells.insert(shell_id, handle);
        Ok(())
    }

    fn ledger_path_for(&self, shell_id: ShellId) -> PathBuf {
        self.nodes
            .get(&shell_id)
            .map(|node| node.ledger_path.clone())
            .unwrap_or_else(|| self.state_dir.join(format!("shell_{shell_id}.ledger")))
    }
}

pub struct RunReceipt {
    pub epoch: u64,
    pub state_hash: String,
    pub shell_id: ShellId,
    pub command: String,
}

struct ShellHandle {
    prompt: String,
    capture_block: String,
    reader: Box<dyn Read + Send>,
    writer: Box<dyn Write + Send>,
    child: Box<dyn Child + Send>,
    pending: String,
    epoch: u64,
}

impl ShellHandle {
    fn spawn(shell_id: ShellId) -> Result<Self> {
        let pty_system = native_pty_system();
        let pair = pty_system.openpty(PtySize {
            rows: 40,
            cols: 120,
            pixel_width: 0,
            pixel_height: 0,
        })?;
        let mut cmd = CommandBuilder::new("bwrap");
        cmd.arg("--die-with-parent");
        cmd.arg("--unshare-user");
        cmd.arg("--unshare-pid");
        cmd.arg("--unshare-uts");
        cmd.arg("--unshare-ipc");
        cmd.arg("--unshare-cgroup");
        cmd.arg("--unshare-mount");
        cmd.arg("--uid");
        cmd.arg("0");
        cmd.arg("--gid");
        cmd.arg("0");
        cmd.arg("--ro-bind");
        cmd.arg("/");
        cmd.arg("/");
        cmd.arg("--dev-bind");
        cmd.arg("/dev");
        cmd.arg("/dev");
        cmd.arg("--proc");
        cmd.arg("/proc");
        cmd.arg("--tmpfs");
        cmd.arg("/sandbox");
        cmd.arg("--tmpfs");
        cmd.arg("/tmp");
        cmd.arg("--setenv");
        cmd.arg("HOME");
        cmd.arg("/sandbox");
        cmd.arg("--setenv");
        cmd.arg("PATH");
        cmd.arg("/usr/bin:/bin");
        cmd.arg("--chdir");
        cmd.arg("/sandbox");
        cmd.arg("/bin/bash");
        cmd.arg("--noprofile");
        cmd.arg("--norc");
        cmd.arg("-i");
        let child = pair.slave.spawn_command(cmd)?;
        drop(pair.slave);
        let reader = pair.master.try_clone_reader()?;
        let writer = pair.master.take_writer()?;
        let prompt = format!("{PROMPT_PREFIX}{shell_id}_# ");
        let mut handle = Self {
            prompt: prompt.clone(),
            capture_block: capture_block(),
            reader,
            writer,
            child,
            pending: String::new(),
            epoch: 0,
        };
        handle.configure_prompt()?;
        Ok(handle)
    }

    fn configure_prompt(&mut self) -> Result<()> {
        self.send_line("unset HISTFILE")?;
        self.send_line("stty -echo")?;
        self.send_line(&format!("export PS1='{}'", self.prompt))?;
        let _ = self.read_until_prompt()?;
        Ok(())
    }

    fn send_line(&mut self, line: &str) -> Result<()> {
        let mut data = line.as_bytes().to_vec();
        data.push(b'\n');
        self.writer.write_all(&data)?;
        self.writer.flush()?;
        Ok(())
    }

    fn execute(&mut self, command: &str) -> Result<CommandOutcome> {
        self.send_line(command)?;
        let capture = self.capture_block.clone();
        self.send_line(&capture)?;
        let raw = self.read_until_prompt()?;
        let state = parse_capture(&raw)?;
        let state_hash = hash_state(&state)?;
        Ok(CommandOutcome { state_hash })
    }

    fn read_until_prompt(&mut self) -> Result<String> {
        loop {
            if let Some(idx) = self.pending.find(&self.prompt) {
                let output = self.pending[..idx].to_string();
                self.pending = self.pending[idx + self.prompt.len()..].to_string();
                return Ok(output);
            }
            let mut buffer = [0u8; 4096];
            let n = self.reader.read(&mut buffer)?;
            if n == 0 {
                return Err(anyhow!("shell exited unexpectedly"));
            }
            let chunk = String::from_utf8_lossy(&buffer[..n]);
            self.pending.push_str(&chunk);
        }
    }

    fn terminate(mut self) {
        let _ = self.child.kill();
    }
}

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
