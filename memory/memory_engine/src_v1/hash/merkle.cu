#include <cuda_runtime.h>
#include <stdint.h>

#define HASH_SIZE 32
#define BLOCK_SIZE 64

// ---------------- SHA256 constants ----------------

__device__ __constant__ uint32_t k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
    0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
    0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
    0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
    0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ void sha256_internal(
    const uint8_t* left,
    const uint8_t* right,
    uint8_t* out
) {
    uint8_t block[BLOCK_SIZE];

    // Build 64-byte block = left || right
    #pragma unroll
    for (int i = 0; i < 32; i++) block[i] = left[i];
    #pragma unroll
    for (int i = 0; i < 32; i++) block[i] = right[i];

    uint32_t w[64];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] =
            (block[i*4+0] << 24) |
            (block[i*4+1] << 16) |
            (block[i*4+2] << 8)  |
            (block[i*4+3]);
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a=0x6a09e667;
    uint32_t b=0xbb67ae85;
    uint32_t c=0x3c6ef372;
    uint32_t d=0xa54ff53a;
    uint32_t e=0x510e527f;
    uint32_t f=0x9b05688c;
    uint32_t g=0x1f83d9ab;
    uint32_t h=0x5be0cd19;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t temp1 = h + S1 + ch + k[i] + w[i];
        uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h=g;
        g=f;
        f=e;
        e=d + temp1;
        d=c;
        c=b;
        b=a;
        a=temp1 + temp2;
    }

    uint32_t state[8] = {
        a+0x6a09e667,
        b+0xbb67ae85,
        c+0x3c6ef372,
        d+0xa54ff53a,
        e+0x510e527f,
        f+0x9b05688c,
        g+0x1f83d9ab,
        h+0x5be0cd19
    };

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i*4+0] = (state[i] >> 24) & 0xff;
        out[i*4+1] = (state[i] >> 16) & 0xff;
        out[i*4+2] = (state[i] >> 8)  & 0xff;
        out[i*4+3] = state[i] & 0xff;
    }
}

// ---------------- Multi-level kernel ----------------

extern "C" __global__
void merkle_rebuild_kernel(
    uint8_t* tree,
    uint64_t tree_size
)
{
    uint64_t level_size   = tree_size;
    uint64_t level_offset = tree_size;

    while (level_size > 1)
    {
        uint64_t parent_count = level_size >> 1;
        uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid < parent_count)
        {
            uint64_t left_index   = level_offset + tid * 2;
            uint64_t right_index  = left_index + 1;
            uint64_t parent_index = (level_offset >> 1) + tid;

            uint8_t* left  = &tree[left_index  * HASH_SIZE];
            uint8_t* right = &tree[right_index * HASH_SIZE];
            uint8_t* out   = &tree[parent_index * HASH_SIZE];

            sha256_internal(left, right, out);
        }

        __syncthreads();
        level_offset >>= 1;
        level_size   >>= 1;
    }
}
