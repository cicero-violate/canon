#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

// Variables:
//   V          = number of vertices
//   E          = number of directed edges
//   edges      = flat array of (u, v, weight) triples, length E*3
//   dist       = distance array length V, initialised INF except source=0
//   changed    = flag: did any thread relax an edge this pass?
//   INF        = UINT64_MAX / 2
//
// Equation (parallel edge relaxation):
//   Each thread owns one edge (u,v,w):
//     if dist[u] + w < dist[v]: atomicMin(&dist[v], dist[u]+w), changed=1
//   Repeat V-1 passes.
//   Pass V: if changed => negative cycle.
//
// atomicMin on uint64_t via CAS loop (CUDA has no native u64 atomicMin).

#define INF (UINT64_MAX / 2ULL)

__device__ void atomic_min_u64(uint64_t* addr, uint64_t val) {
    unsigned long long* a = (unsigned long long*)addr;
    unsigned long long v  = (unsigned long long)val;
    unsigned long long old = *a;
    unsigned long long assumed;

    do {
        assumed = old;
        if (assumed <= v) break;
        old = atomicCAS(a, assumed, v);
    } while (old != assumed);
}

__global__ void relax_kernel(
    const uint64_t* __restrict__ edges,   // (u,v,w) interleaved, length E*3
    uint64_t* __restrict__ dist,
    int E,
    int* changed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    uint64_t u = edges[tid * 3 + 0];
    uint64_t v = edges[tid * 3 + 1];
    uint64_t w = edges[tid * 3 + 2];

    uint64_t du = dist[u];
    if (du == INF) return;

    uint64_t nd = du + w;

    if (nd < dist[v]) {
        atomic_min_u64(&dist[v], nd);
        atomicExch(changed, 1);
    }
}

// Returns 1 if a negative cycle was detected, 0 otherwise.
// dist_out must be pre-allocated to V uint64_t values.
extern "C" int gpu_bellman_ford(
    const uint64_t* edges_flat,   // E*(u,v,w) triples
    int V,
    int E,
    int source,
    uint64_t* dist_out)
{
    size_t dist_bytes = (size_t)V * sizeof(uint64_t);
    size_t edge_bytes = (size_t)E * 3 * sizeof(uint64_t);

    uint64_t* d_dist;
    uint64_t* d_edges;
    int* d_changed;

    cudaMalloc(&d_dist,    dist_bytes);
    cudaMalloc(&d_edges,   edge_bytes);
    cudaMalloc(&d_changed, sizeof(int));

    uint64_t* h_dist = (uint64_t*)malloc(dist_bytes);
    for (int i = 0; i < V; i++) {
        h_dist[i] = INF;
    }
    h_dist[source] = 0;

    cudaMemcpy(d_dist,  h_dist,     dist_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, edges_flat, edge_bytes, cudaMemcpyHostToDevice);
    free(h_dist);

    int threads = 256;
    int blocks  = (E + threads - 1) / threads;

    for (int pass = 0; pass < V; pass++) {
        cudaMemset(d_changed, 0, sizeof(int));

        relax_kernel<<<blocks, threads>>>(
            d_edges, d_dist, E, d_changed);

        cudaDeviceSynchronize();

        int h_changed = 0;
        cudaMemcpy(&h_changed, d_changed,
                   sizeof(int), cudaMemcpyDeviceToHost);

        if (pass < V - 1 && !h_changed) {
            break; // early exit
        }

        if (pass == V - 1 && h_changed) {
            cudaFree(d_dist);
            cudaFree(d_edges);
            cudaFree(d_changed);
            return 1; // negative cycle
        }
    }

    cudaMemcpy(dist_out, d_dist,
               dist_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_dist);
    cudaFree(d_edges);
    cudaFree(d_changed);

    return 0;
}
