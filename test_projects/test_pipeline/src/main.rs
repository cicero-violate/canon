use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
struct Node {
    id: usize,
    name: String,
    file: PathBuf,
}

#[derive(Clone, Debug)]
struct GraphSnapshot {
    nodes: HashMap<usize, Node>,
}

#[derive(Clone, Debug)]
struct MoveOp {
    node_id: usize,
    new_file: PathBuf,
}

#[derive(Clone, Debug)]
struct Plan1 {
    file_map: HashMap<PathBuf, Vec<Node>>,
}

fn capture_project(root: &Path) -> GraphSnapshot {
    println!("[capture_project] root={}", root.display());
    let mut nodes = HashMap::new();
    let mut id = 0;
    for entry in walk(root) {
        if entry.extension().and_then(|e| e.to_str()) == Some("rs") {
            id += 1;
            println!("[capture_project] discovered node id={} file={}", id, entry.display());
            nodes.insert(id, Node { id, name: format!("module_{}", id), file: entry.clone() });
        }
    }
    GraphSnapshot { nodes }
}

fn apply_moves_to_snapshot(mut snap: GraphSnapshot, moves: &[MoveOp]) -> GraphSnapshot {
    println!("[apply_moves_to_snapshot] move_count={}", moves.len());
    for mv in moves {
        if let Some(node) = snap.nodes.get_mut(&mv.node_id) {
            println!("[apply_moves_to_snapshot] moving node id={} from={} to={}", mv.node_id, node.file.display(), mv.new_file.display());
            node.file = mv.new_file.clone();
        }
    }
    snap
}

fn projection_plan(snap: &GraphSnapshot) -> Plan1 {
    println!("[project_plan] node_count={}", snap.nodes.len());
    let mut file_map: HashMap<PathBuf, Vec<Node>> = HashMap::new();
    for node in snap.nodes.values() {
        println!("[project_plan] assigning node id={} to file={}", node.id, node.file.display());
        file_map.entry(node.file.clone()).or_default().push(node.clone());
    }
    Plan1 { file_map }
}

fn emit_plan(plan: &Plan1, fix_root: &Path) -> std::io::Result<()> {
    println!("[emit_plan] file_count={} fix_root={}", plan.file_map.len(), fix_root.display());
    for (file, nodes) in &plan.file_map {
        let relative = file.file_name().unwrap();
        let target = fix_root.join(relative);
        println!("[emit_plan] writing target={} node_count={}", target.display(), nodes.len());
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut content = String::new();
        for n in nodes {
            println!("[emit_plan] emitting node id={} name={} into={}", n.id, n.name, target.display());
            content.push_str(&format!("pub mod {};\n", n.name));
        }
        fs::write(target, content)?;
    }
    Ok(())
}

fn walk(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(root) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(walk(&p));
            } else {
                out.push(p);
            }
        }
    }
    out
}

fn main() -> std::io::Result<()> {
    let root = PathBuf::from("./src");
    let fix_root = PathBuf::from("./fix_point");

    println!("[main] starting pipeline");
    let snap0 = capture_project(&root);

    let moves = vec![MoveOp { node_id: 1, new_file: PathBuf::from("moved.rs") }];

    let snap1 = apply_moves_to_snapshot(snap0, &moves);
    let plan1 = projection_plan(&snap1);
    emit_plan(&plan1, &fix_root)?;

    println!("[main] pipeline complete");
    Ok(())
}
