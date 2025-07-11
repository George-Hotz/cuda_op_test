#ifndef SOFTMAX_KERNEL_H
#define SOFTMAX_KERNEL_H

#include <cfloat>           
#include <cmath>          
#include <cuda_runtime.h>      
#include <cuda_fp16.h>
#include <device_launch_parameters.h> 

#define WARP_SIZE 32
#define HALF_MAX 65504.0f

// Warp Reduce Sum for FP16
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// Warp Reduce Max for FP16
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_max_f16(half val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = __hmax(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template <const int NUM_THREADS = 256>
__global__ void softmax_f16_per_mat_kernel(half* out, const half* inp, int N, int C) {
    constexpr int warpsPerBlock = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ half shared[2 * warpsPerBlock];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    half* maxvals = shared;
    half* sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const half* x = inp + bid * C;

    // Step 1: Find max value
    half maxval = -HALF_MAX;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = __hmax(maxval, x[i]);
    }
    maxval = warp_reduce_max_f16<WARP_SIZE>(maxval);

    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    if (tid == 0) {
        half val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = __hmax(val, maxvals[i]);
        }
        maxvals[0] = val;
    }
    __syncthreads();
    half offset = maxvals[0];

    // Step 2: Compute exp(x - maxval)
    for (int i = tid; i < C; i += blockDim.x) {
        out[bid * C + i] = hexp(__hsub(x[i], offset));
    }

    // Step 3: Sum all exp values
    x = out + bid * C;
    half sumval = __float2half(0.0f);
    for (int i = tid; i < C; i += blockDim.x) {
        sumval = __hadd(sumval, x[i]);
    }
    sumval = warp_reduce_sum_f16<WARP_SIZE>(sumval);

    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    if (tid == 0) {
        half val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val = __hadd(val, sumvals[i]);
        }
        sumvals[0] = val;
    }
    __syncthreads();
    half sum = sumvals[0];

    // Step 4: Normalize by sum
    for (int i = tid; i < C; i += blockDim.x) {
        out[bid * C + i] = __hdiv(x[i], sum);
    }
}

// Helper function to launch the kernel
inline void launch_softmax_kernel(half* out, const half* inp, int N, int C, cudaStream_t stream = 0) {
    constexpr int NUM_THREADS = 256;
    dim3 blocks(N);
    dim3 threads(NUM_THREADS);
    softmax_f16_per_mat_kernel<NUM_THREADS><<<blocks, threads, 0, stream>>>(out, inp, N, C);
}

#endif