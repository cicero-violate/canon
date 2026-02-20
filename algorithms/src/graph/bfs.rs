use std::collections::VecDeque;

pub fn bfs(adj: &[Vec<usize>], start: usize) -> Vec<usize> {
    let mut visited = vec![false; adj.len()];
    let mut queue = VecDeque::new();
    let mut order = Vec::new();

    visited[start] = true;
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &n in &adj[node] {
            if !visited[n] {
                visited[n] = true;
                queue.push_back(n);
            }
        }
    }
    order
}
