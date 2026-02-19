#include <cuda_runtime.h>
#include <stdint.h>

// One thread per frontier node.
// For each frontier node u, iterate its neighbors.
// If neighbor v is unvisited and passes edge filter, atomically claim it.
__global__ void bfs_kernel(
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    const uint8_t*  edge_kinds,
    int32_t*        dist,
    const uint32_t* frontier,
    uint32_t        frontier_size,
    uint32_t*       next_frontier,
    uint32_t*       next_size,
    int32_t         current_dist,
    uint8_t         edge_filter,   // 255 = no filter
    uint32_t        n_nodes
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    uint32_t u     = frontier[tid];
    uint32_t start = row_offsets[u];
    uint32_t end   = row_offsets[u + 1];

    for (uint32_t e = start; e < end; e++) {
        if (edge_filter != 255 && edge_kinds[e] != edge_filter) continue;

        uint32_t v = col_indices[e];
        if (v >= n_nodes) continue;

        // Atomically mark v as visited at current_dist + 1
        int32_t expected = -1;
        int32_t desired  = current_dist + 1;

        // Use atomicCAS to avoid races
        int32_t old = atomicCAS((int*)&dist[v], expected, desired);
        if (old == -1) {
            // We claimed v — add to next frontier
            uint32_t pos = atomicAdd(next_size, 1u);
            next_frontier[pos] = v;
        }
    }
}

extern "C"
void launch_bfs_kernel(
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    const uint8_t*  edge_kinds,
    int32_t*        dist,
    const uint32_t* frontier,
    uint32_t        frontier_size,
    uint32_t*       next_frontier,
    uint32_t*       next_size,
    int32_t         current_dist,
    uint8_t         edge_filter,
    uint32_t        n_nodes
) {
    if (frontier_size == 0) return;
    const int threads = 256;
    int blocks = (frontier_size + threads - 1) / threads;
    bfs_kernel<<<blocks, threads>>>(
        row_offsets, col_indices, edge_kinds,
        dist, frontier, frontier_size,
        next_frontier, next_size,
        current_dist, edge_filter, n_nodes
    );
}
