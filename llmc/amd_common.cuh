/*

Goal: unobtrusively provide support for AMD devices with minimal changes to the main CUDA code

Example (assuming ROCm 6.1.1 installed in /opt/rocm, or ROCM_PATH environment variable is set):

*/

#pragma once

#ifdef MULTI_GPU
#include <mpi.h>
#include <rccl/rccl.h>
#endif

#if defined(__gfx1100__) || defined(__gfx1103__)
#define AMD_TARGET_ARCH_RDNA3
#elif defined(__gfx90a__)
#define AMD_TARGET_ARCH_CDNA2
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define AMD_TARGET_ARCH_CDNA3
#endif

#include <hip/hip_bfloat16.h>

#ifndef DISABLE_CK

#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/ck.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// cublaslt does not have kernels for gfx11, so best alternative in terms of perf/effort seems to be composite_kernels
// somewhat janky to invoke with all of the templating, but works..
static inline void matmul_forward_gfx11(hip_bfloat16* out,
                   const hip_bfloat16* inp, const hip_bfloat16* weight, const hip_bfloat16* bias,
                   int B, int T, int C, int OC, cudaStream_t stream) {
    using AElementOp = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp = ck::tensor_operation::element_wise::PassThrough;
    using CDEElementOp = ck::tensor_operation::element_wise::Add;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};
    auto cde_element_op = CDEElementOp{};

    if (bias == NULL) {
        auto device_op = ck::tensor_operation::device::DeviceGemmWmma_CShuffle <
            ck::tensor_layout::gemm::RowMajor,
            ck::tensor_layout::gemm::ColumnMajor,
            ck::tensor_layout::gemm::RowMajor,
            ck::bhalf_t,
            ck::bhalf_t,
            ck::bhalf_t,
            float,
            ck::bhalf_t,
            AElementOp,
            BElementOp,
            CElementOp,
            GemmSpec,
            256,
            128,
            256,
            8,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            1,
            1,
            S<1, 32, 1, 8>,
            8,
            1>{};
        auto invoker = device_op.MakeInvoker();
        auto argument = device_op.MakeArgument(
            reinterpret_cast<ck::bhalf_t*>(const_cast<hip_bfloat16 *>(inp)),
            reinterpret_cast<ck::bhalf_t*>(const_cast<hip_bfloat16 *>(weight)),
            reinterpret_cast<ck::bhalf_t*>(out),
            B*T,
            OC,
            C,
            C,
            C,
            OC,
            a_element_op,
            b_element_op,
            c_element_op);
        invoker.Run(argument, StreamConfig{stream});
    } else {
        auto device_op = ck::tensor_operation::device::DeviceGemmMultipleD_Wmma_CShuffle <
            ck::tensor_layout::gemm::RowMajor,
            ck::tensor_layout::gemm::ColumnMajor,
            ck::Tuple<ck::tensor_layout::gemm::RowMajor>,
            ck::tensor_layout::gemm::RowMajor,
            ck::bhalf_t,
            ck::bhalf_t,
            ck::Tuple<ck::bhalf_t>,
            ck::bhalf_t,
            float,
            ck::bhalf_t,
            AElementOp,
            BElementOp,
            CDEElementOp,
            GemmSpec,
            256,
            128,
            256,
            8,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            1,
            1,
            S<1, 32, 1, 8>,
            8>{};
        auto invoker = device_op.MakeInvoker();
        auto argument = device_op.MakeArgument(
            reinterpret_cast<ck::bhalf_t*>(const_cast<hip_bfloat16 *>(inp)),
            reinterpret_cast<ck::bhalf_t*>(const_cast<hip_bfloat16 *>(weight)),
            std::array<const void*, 1>{reinterpret_cast<ck::bhalf_t*>(const_cast<hip_bfloat16 *>(bias))},
            reinterpret_cast<ck::bhalf_t*>(out),
            B*T,
            OC,
            C,
            C,
            C,
            std::array<ck::index_t, 1>{0},
            OC,
            a_element_op,
            b_element_op,
            cde_element_op);
        invoker.Run(argument, StreamConfig{stream});
    }
}

#endif

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_cooperative_groups.h>

// macros below handle mostly cublaslt stuff not handled by hipify (yet)
#define cublasLtMatmulPreferenceSetAttribute hipblasLtMatmulPreferenceSetAttribute
#define CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
#define cublasLtMatmulPreferenceCreate hipblasLtMatmulPreferenceCreate
#define cublasLtMatmulDescSetAttribute hipblasLtMatmulDescSetAttribute
#define cublasLtMatmulPreferenceDestroy hipblasLtMatmulPreferenceDestroy
#define cublasLtMatmulDescDestroy hipblasLtMatmulDescDestroy
#define cublasLtMatmulAlgoGetHeuristic hipblasLtMatmulAlgoGetHeuristic
#define cublasLtMatrixLayoutDestroy hipblasLtMatrixLayoutDestroy
#define CUBLASLT_EPILOGUE_GELU_BIAS HIPBLASLT_EPILOGUE_GELU_BIAS
#define CUBLASLT_EPILOGUE_GELU HIPBLASLT_EPILOGUE_GELU
#define CUBLASLT_EPILOGUE_BIAS HIPBLASLT_EPILOGUE_BIAS
#define CUBLASLT_EPILOGUE_DEFAULT HIPBLASLT_EPILOGUE_DEFAULT
#define cublasLtEpilogue_t hipblasLtEpilogue_t
#define cublasLtMatmulHeuristicResult_t hipblasLtMatmulHeuristicResult_t
#define cublasLtMatrixLayout_t hipblasLtMatrixLayout_t
#define cublasLtMatmulPreference_t hipblasLtMatmulPreference_t
#define cublasLtMatmulDesc_t hipblasLtMatmulDesc_t
#define cublasLtHandle_t hipblasLtHandle_t
#define cublasLtMatmul hipblasLtMatmul
#define CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT
#define CUBLASLT_MATMUL_DESC_TRANSA HIPBLASLT_MATMUL_DESC_TRANSA
#define CUBLASLT_MATMUL_DESC_TRANSB HIPBLASLT_MATMUL_DESC_TRANSB
#define CUBLASLT_MATMUL_DESC_EPILOGUE HIPBLASLT_MATMUL_DESC_EPILOGUE
#define CUBLASLT_MATMUL_DESC_BIAS_POINTER HIPBLASLT_MATMUL_DESC_BIAS_POINTER
#define CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET
#define CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD
#define CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE
#define CUBLASLT_MATMUL_DESC_SCALE_TYPE HIPBLASLT_MATMUL_DESC_SCALE_TYPE
#define CUBLASLT_EPILOGUE_BGRADB HIPBLASLT_EPILOGUE_BGRADB
#define CUBLASLT_EPILOGUE_GELU_AUX HIPBLASLT_EPILOGUE_GELU_AUX
#define CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER
#define CUBLASLT_EPILOGUE_DGELU HIPBLASLT_EPILOGUE_DGELU
#define CUBLASLT_EPILOGUE_GELU_AUX_BIAS HIPBLASLT_EPILOGUE_GELU_AUX_BIAS
#define cublasLtCreate hipblasLtCreate
#define cublasLtDestroy hipblasLtDestroy
#define cublasLtMatrixLayoutCreate hipblasLtMatrixLayoutCreate
#define cublasLtMatmulDescCreate hipblasLtMatmulDescCreate
#define cublasLtMatrixLayoutSetAttribute hipblasLtMatrixLayoutSetAttribute
#define cublasSetMathMode(handle, mode) HIPBLAS_STATUS_SUCCESS
#define hipblasSetMathMode(handle, mode) HIPBLAS_STATUS_SUCCESS
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define cublasMath_t hipblasMath_t
#define CUBLAS_TF32_TENSOR_OP_MATH HIPBLAS_TF32_TENSOR_OP_MATH
#define CUBLAS_DEFAULT_MATH HIPBLAS_DEFAULT_MATH
#define hipFuncSetAttribute(x,y,z) 0
#define hipProfilerStart(x) hipSuccess
#define hipProfilerStop(x) hipSuccess
#define nvtxRangePush(x) {}
#define nvtxRangePop(x) {}
#define nvtxNameCudaStreamA(x,y) {}
#define cublasSetWorkspace(x,y,z) HIPBLAS_STATUS_SUCCESS
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

#if defined(AMD_TARGET_ARCH_RDNA3)
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    asm volatile ("ds_swizzle_b32 v1, %0 offset:swizzle(SWAP,16) \n"\
                  "s_waitcnt lgkmcnt(0) \n"\
                  "v_add_f32_e32 %0, %0, v1 \n"
                  "s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) \n"\
                  "v_add_f32_dpp %0, %0, %0 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1 \n"\
                  "v_add_f32_dpp %0, %0, %0 row_ror:4 row_mask:0xf bank_mask:0xf bound_ctrl:1 \n"\
                  "s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) \n"\
                  "v_add_f32_dpp %0, %0, %0 row_ror:2 row_mask:0xf bank_mask:0xf bound_ctrl:1 \n"\
                  "v_add_f32_dpp %0, %0, %0 row_ror:1 row_mask:0xf bank_mask:0xf bound_ctrl:1 \n"
                  : "+v"(x) : : "v1");
    return x;
}

static __device__ __forceinline__ float warp_reduce_max(float x) {
    asm volatile ("s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) \n"\
                  "v_max_f32_dpp %0, %0, %0 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1 \n"\
                  "v_max_f32_dpp %0, %0, %0 row_ror:4 row_mask:0xf bank_mask:0xf bound_ctrl:1 \n"\
                  "s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) \n"\
                  "v_max_f32_dpp %0, %0, %0 row_ror:2 row_mask:0xf bank_mask:0xf bound_ctrl:1 \n"\
                  "v_max_f32_dpp %0, %0, %0 row_ror:1 row_mask:0xf bank_mask:0xf bound_ctrl:1 \n"\
                  "ds_swizzle_b32 v1, %0 offset:swizzle(SWAP,16) \n"\
                  "s_waitcnt lgkmcnt(0) \n"\
                  "v_max_f32_e32 %0, %0, v1 \n"
                  : "+v"(x) : : "v1");
    return x;
}
#else
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#ifdef WAVEFRONTSIZE64
    for (int mask = 32; mask > 0; mask >>= 1) { x += __shfl_xor(x, mask, 64); }
#else
    for (int mask = 16; mask > 0; mask >>= 1) { x += __shfl_xor(x, mask, 32); }
#endif
    return x;
}

static __device__ __forceinline__ float warp_reduce_max(float x) {
#ifdef WAVEFRONTSIZE64
    for (int mask = 32; mask > 0; mask >>= 1) { x = fmaxf(x, __shfl_xor(x, mask, 64)); }
#else
    for (int mask = 16; mask > 0; mask >>= 1) { x = fmaxf(x, __shfl_xor(x, mask, 32)); }
#endif
    return x;
}
#endif

namespace cooperative_groups {
template <typename T>
struct reduce_operator {
    static __device__ __forceinline__ T reduce(const T a, const T b) { return a+b; };
};

template <typename T>
struct plus : public reduce_operator<T> {
    static __device__ __forceinline__ T reduce(const T a, const T b) {
        return a + b;
    }
};

template <typename T>
struct greater : public reduce_operator<T> {
    static __device__ __forceinline__ T reduce(const T a, const T b) {
        return fmaxf(a, b);
    }
};

template <typename T>
static __device__ __forceinline__ float reduce(const thread_block_tile<32>& warp, float x, const plus<T>& op) {
    return warp_reduce_sum(x);
}

template <typename T>
static __device__ __forceinline__ float reduce(const thread_block_tile<32>& warp, float x, const greater<T>& op) {
    return warp_reduce_max(x);
}

template struct plus<float>;
template struct greater<float>;
}
