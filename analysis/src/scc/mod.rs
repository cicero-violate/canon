//! Strongly connected components in the call graph (Tarjan's algorithm).
//!
//! Variables:
//!   index(v)   = discovery order of v
//!   lowlink(v) = lowest index reachable from v's subtree
//!
//! Equations:
//!   lowlink(v) = min({ index(v) }
//!                  ∪ { lowlink(w) | (v,w) ∈ E, w on stack }
//!                  ∪ { lowlink(w) | (v,w) ∈ E, w not yet visited })
//!   v is SCC root iff lowlink(v) = index(v)

use std::collections::HashMap;

pub fn tarjan_scc(adj: &HashMap<String, Vec<String>>) -> Vec<Vec<String>> {
    let nodes: Vec<String> = adj.keys().cloned().collect();
    let mut index_counter = 0usize;
    let mut stack: Vec<String> = Vec::new();
    let mut on_stack: HashMap<String, bool> = HashMap::new();
    let mut index_map: HashMap<String, usize> = HashMap::new();
    let mut lowlink: HashMap<String, usize> = HashMap::new();
    let mut sccs: Vec<Vec<String>> = Vec::new();

    fn strongconnect(
        v: &str,
        adj: &HashMap<String, Vec<String>>,
        index_counter: &mut usize,
        stack: &mut Vec<String>,
        on_stack: &mut HashMap<String, bool>,
        index_map: &mut HashMap<String, usize>,
        lowlink: &mut HashMap<String, usize>,
        sccs: &mut Vec<Vec<String>>,
    ) {
        index_map.insert(v.to_string(), *index_counter);
        lowlink.insert(v.to_string(), *index_counter);
        *index_counter += 1;
        stack.push(v.to_string());
        on_stack.insert(v.to_string(), true);

        if let Some(neighbours) = adj.get(v) {
            for w in neighbours {
                if !index_map.contains_key(w) {
                    strongconnect(w, adj, index_counter, stack, on_stack, index_map, lowlink, sccs);
                    let lv = lowlink[v].min(lowlink[w]);
                    lowlink.insert(v.to_string(), lv);
                } else if *on_stack.get(w).unwrap_or(&false) {
                    let lv = lowlink[v].min(index_map[w]);
                    lowlink.insert(v.to_string(), lv);
                }
            }
        }

        if lowlink[v] == index_map[v] {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.insert(w.clone(), false);
                scc.push(w.clone());
                if w == v { break; }
            }
            sccs.push(scc);
        }
    }

    for v in &nodes {
        if !index_map.contains_key(v) {
            strongconnect(v, adj, &mut index_counter, &mut stack, &mut on_stack,
                &mut index_map, &mut lowlink, &mut sccs);
        }
    }
    sccs
}
