#pragma once

#ifdef MULTI_GPU
#include <mpi.h>
#include <rccl/rccl.h>
#endif

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_fp16.h>

#define hipProfilerStart(x) hipSuccess
#define hipProfilerStop(x) hipSuccess
#define nvtxRangePush(x) {}
#define nvtxRangePop(x) {}
#define nvtxNameCudaStreamA(x,y) {}
#define nvtxNameCudaEventA(x,y) {}
#define cudaStreamWaitEvent(x,y) hipStreamWaitEvent(x,y,0)

static __device__ __forceinline__ hip_bfloat16 __float2bfloat16_rn(float f) {
    return hip_bfloat16::round_to_bfloat16(f);
}

static __device__ __forceinline__ float __bfloat162float(hip_bfloat16 f) {
    return static_cast<float>(f);
}

template <typename T>
static __device__ __forceinline__ T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize) {
    return __shfl_xor(var, laneMask, width);
}

template <typename T>
static __device__ __forceinline__ T __shfl_down_sync(unsigned mask, T var, int laneMask, int width=warpSize) {
    return __shfl_down(var, laneMask, width);
}

// provide cache hints where possible
#define __stcs(ptr, val) patched_stcs(ptr, val)
#define __ldcs(ptr) patched_ldcs(ptr)
#define __stcg(ptr, val) {*(ptr) = val;}
static __device__ __forceinline__ void patched_stcs(float *addr, float val) {
    __builtin_nontemporal_store(val, addr);
}
static __device__ __forceinline__ void patched_stcs(hip_bfloat16 *addr, hip_bfloat16 val) {
    *addr = val;
}
static __device__ __forceinline__ void patched_stcs(int4 *addr, int4 val) {
    int *a = (int *)addr;
    __builtin_nontemporal_store(val.x, a);
    __builtin_nontemporal_store(val.y, a+1);
    __builtin_nontemporal_store(val.z, a+2);
    __builtin_nontemporal_store(val.w, a+3);
}
static __device__ __forceinline__ float patched_ldcs(const float *addr) {
    return __builtin_nontemporal_load(addr);
}
static __device__ __forceinline__ int4 patched_ldcs(const int4 *addr) {
    const int *a = (const int *) addr;
    return make_int4(__builtin_nontemporal_load(a),
        __builtin_nontemporal_load(a+1),
        __builtin_nontemporal_load(a+2),
        __builtin_nontemporal_load(a+3));
}
static __device__ __forceinline__ hip_bfloat16 patched_ldcs(const hip_bfloat16 *addr) {
    return *addr;
}

