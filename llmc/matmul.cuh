/*
Matrix Multiplication, with help from cuBLASLt
*/
#ifdef USE_CK
#include <hip/hip_bfloat16.h>
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/ck.hpp"
#endif

#include <assert.h>
#include <type_traits>      // std::bool_constant
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "cublas_common.h"
// GELU can be either fused (cublasLt) or non-fused (gelu.h)
#include "gelu.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

template<typename OutFloat, bool UseAuxBuffer>
__global__ void matmul_backward_bias_kernel9(OutFloat* dbias, const floatX* dout, int B, int T, int OC,
                                             std::bool_constant<UseAuxBuffer>) {
    constexpr const int bdx = 4;
    constexpr const int bdy = WARP_SIZE / bdx;
    assert(blockDim.x == bdx);
    assert(blockDim.y == bdy);

    int warp_d = (int)threadIdx.x;
    int warp_c = (int)threadIdx.y;
    int block_d = (int)threadIdx.z;

    const int OC_per_warp = bdy * x128::size;  // 64 at BF16

    int local_oc = warp_c * x128::size;
    int global_oc = blockIdx.x * OC_per_warp + local_oc;

    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim.z;

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }

    if(global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = blockIdx.y * bt_per_block + local_bt; idx < B * T; idx += gridDim.y * bt_per_block) {
            x128 packed_dout = load128(dout + global_oc + idx*OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
    }

    __shared__ float sub_results[x128::size][WARP_SIZE][bdy];

    // reduce within-warp results
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        v += __shfl_down_sync(0xffffffff, v, 1, 4);
        v += __shfl_down_sync(0xffffffff, v, 2, 4);
        if(warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    __syncthreads();

    // block-wide reductions
    for (int k = block_d; k < x128::size; k += blockDim.z) {
        float a = 0.f;
        for (int r = warp_d; r < blockDim.z; r += bdx) {
            float v = sub_results[k][r][warp_c];
            v += __shfl_down_sync(0xffffffff, v, 1, 4);
            v += __shfl_down_sync(0xffffffff, v, 2, 4);
            a += v;
        }
        if(warp_d == 0 && global_oc < OC) {
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + blockIdx.y * OC] = a;
            }
        }
    }
}

__global__ void reduce_add_sum_kernel(floatX* dst, const float* src, size_t n, size_t m) {
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * f128::size;
    assert(n % x128::size == 0);
    if (idx < n) {
        f128 acc;
        for(int k = 0; k < f128::size; ++k) {
            acc[k] = 0.f;
        }

        for(int l = 0; l < m; ++l) {
            f128 s = load128(src + idx + n * l);
            for(int k = 0; k < f128::size; ++k) {
                acc[k] += s[k];
            }
        }
        for(int k = 0; k < f128::size; ++k) {
            dst[idx + k] = (floatX) ((float)dst[idx + k] + acc[k]);
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

#ifdef USE_CK

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

void matmul_ck(__hip_bfloat16*__restrict__ d, const __hip_bfloat16*__restrict__ a, const __hip_bfloat16*__restrict__ b, const __hip_bfloat16*__restrict__ bias,
                     int m, int n, int k, cudaStream_t stream=0, bool transA=true, bool transB=false,
                     int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
                     bool accumulate=false, __hip_bfloat16*__restrict__ pre_gelu=NULL, bool backward=false)
{
    NVTX_RANGE_FN();
    if (pre_gelu) { printf("%s: GELU in matmul unsupported\n", __PRETTY_FUNCTION__); exit(-1); }
    if (transA != true || transB != false) { printf("%s: unsupported transA/B\n", __PRETTY_FUNCTION__); exit(-1); }
    if (batch_count != 0 || strideA != 0 || strideB != 0 || strideOut != 0) { printf("%s: batch_count != 0 not supported\n", __PRETTY_FUNCTION__); exit(-1); }
    if (accumulate) { printf("%s: accumulate without batch not supported\n", __PRETTY_FUNCTION__); exit(-1); }

    using ALayout          = ck::tensor_layout::gemm::RowMajor;
    using BLayout          = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout          = ck::tensor_layout::gemm::RowMajor;
    using ELayout          = ck::tensor_layout::gemm::RowMajor;

    using DataType         = ck::bhalf_t;
    using AccDataType      = float;

    using AElementOp       = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp       = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp       = ck::tensor_operation::element_wise::PassThrough;
    using CDEElementOp     = ck::tensor_operation::element_wise::Add;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};
    auto cde_element_op = CDEElementOp{};

    auto StrideA = k; auto StrideB = k; auto StrideC = m;

    if (bias) {
#ifdef BUILD_XDL
        auto device_op = ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle<
            ALayout, BLayout, ck::Tuple<CLayout>, ELayout,
            DataType, DataType, AccDataType, DataType, ck::Tuple<DataType>, DataType,
            AElementOp, BElementOp, CDEElementOp,
            GemmDefault,
            1, 256, 256, 128, 32, 8, 8, 32, 32, 4, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, 1, 1, S<1, 32, 1, 8>, 8>{};
#else
#ifdef NEW_CK
        auto device_op = ck::tensor_operation::device::DeviceGemmMultipleD_Wmma_CShuffle<
            ALayout, BLayout, ck::Tuple<CLayout>, ELayout,
            DataType, DataType, AccDataType, DataType, ck::Tuple<DataType>, DataType,
            AElementOp, BElementOp, CDEElementOp,
            GemmDefault,
            2, 128, 64, 128, 64, 8, 16, 16, 2, 4, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true, S<4, 32, 1>,
            S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true, 1, 1, S<1, 32, 1, 4>, 8, ck::make_default_loop_scheduler(), ck::PipelineVersion::v2>{};
#else
        auto device_op = ck::tensor_operation::device::DeviceGemmMultipleD_Wmma_CShuffle <
            ALayout, BLayout, ck::Tuple<CLayout>, ELayout,
            DataType, DataType, ck::Tuple<DataType>, DataType, AccDataType, DataType,
            AElementOp, BElementOp, CDEElementOp,
            GemmSpec,
            256, 128, 256, 8, 8, 16, 16, 4, 4, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true, 1, 1, S<1, 32, 1, 8>, 8>{};
#endif
#endif
        auto invoker = device_op.MakeInvoker();
        auto argument = device_op.MakeArgument(
                reinterpret_cast<DataType*>(const_cast<__hip_bfloat16 *>(b)),
                reinterpret_cast<DataType*>(const_cast<__hip_bfloat16 *>(a)),
                std::array<const void*, 1>{reinterpret_cast<DataType*>(const_cast<__hip_bfloat16 *>(bias))},
                reinterpret_cast<DataType*>(d),
                n, m, k, StrideA, StrideB, std::array<ck::index_t, 1>{0}, StrideC,
                a_element_op, b_element_op, cde_element_op);
        invoker.Run(argument, StreamConfig{stream});

    } else {
#ifdef BUILD_XDL
        auto device_op = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
            ALayout, BLayout, ELayout,
            DataType, DataType, DataType, AccDataType, DataType,
            AElementOp, BElementOp, CElementOp,
            GemmDefault,
            256, 128, 128, 64, 8, 8, 16, 16, 4, 4, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0, 1, 2, S<1, 32, 1, 8>, 8,
            ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3>{};
#else
#ifdef NEW_CK
        auto device_op = ck::tensor_operation::device::DeviceGemmWmma_CShuffle <
            ALayout, BLayout, ELayout,
            DataType, DataType, DataType, AccDataType, DataType,
            AElementOp, BElementOp, CElementOp,
            GemmDefault,
            2, 128, 64, 128, 64, 8, 16, 16, 2, 4, S<4, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true, S<4, 32, 1>, S<1, 0, 2>,
            S<1, 0, 2>, 2, 8, 8, true, 1, 1, S<1, 32, 1, 4>, 8, ck::make_default_loop_scheduler(), ck::PipelineVersion::v2>{};
#else
        auto device_op = ck::tensor_operation::device::DeviceGemmWmma_CShuffle <
            ALayout, BLayout, ELayout,
            DataType, DataType, DataType, AccDataType, DataType,
            AElementOp, BElementOp, CElementOp,
            GemmSpec,
            256, 128, 256, 8, 8, 16, 16, 4, 4, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true, 1, 1, S<1, 32, 1, 8>, 8, 1>{};
#endif
#endif
        auto invoker = device_op.MakeInvoker();
        auto argument = device_op.MakeArgument(
                reinterpret_cast<DataType*>(const_cast<__hip_bfloat16 *>(b)),
                reinterpret_cast<DataType*>(const_cast<__hip_bfloat16 *>(a)),
                reinterpret_cast<DataType*>(d),
                n, m, k, StrideA, StrideB, StrideC,
#ifdef BUILD_XDL
                1,
#endif
                a_element_op, b_element_op, c_element_op);
        invoker.Run(argument, StreamConfig{stream});

    }
}

void matmul_ck(floatX*__restrict__ d, const floatX*__restrict__ a, const floatX*__restrict__ b, const floatX*__restrict__ bias,
                     int m, int n, int k, cudaStream_t stream=0, bool transA=true, bool transB=false,
                     int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
                     bool accumulate=false, floatX*__restrict__ pre_gelu=NULL, bool backward=false)
{
    matmul_ck(reinterpret_cast<__hip_bfloat16*__restrict__>(d),
        reinterpret_cast<const __hip_bfloat16*__restrict__>(a),
        reinterpret_cast<const __hip_bfloat16*__restrict__>(b),
        reinterpret_cast<const __hip_bfloat16*__restrict__>(bias),
        m, n, k, stream, transA, transB, batch_count, strideA, strideB, strideOut, accumulate,
        reinterpret_cast<__hip_bfloat16*__restrict__>(pre_gelu), backward
        );
}

#endif
#ifdef USE_HIPBLAS

__device__ __forceinline__ __hip_bfloat162 __float_as_bfloat162(float x) {
    unsigned int temp = __float_as_uint(x);
    return *reinterpret_cast<__hip_bfloat162 *>(&temp);
}
__device__ __forceinline__ float __bfloat162_as_float(__hip_bfloat162 x) {
    return *reinterpret_cast<float *>(&x);
}

__global__ void add_bias(floatX*__restrict__ out, const floatX*__restrict__ bias, const int rows, const int cols) {
    const int tid    = threadIdx.x;
    const int stride = blockDim.x * 8;

    floatX *__restrict__       p0 = out + (blockIdx.x * cols) + (tid * 8);
    const floatX *__restrict__ p1 = bias + (tid * 8);

    for (int i = tid*8; i < cols; i += stride) {
        float d0[4], d1[4];
        for(int x=0;x<4;x++) d0[x] = reinterpret_cast<float *__restrict__>(p0)[x];
        for(int x=0;x<4;x++) d1[x] = reinterpret_cast<const float *__restrict__>(p1)[x];
        for(int x=0;x<4;x++) {
            __hip_bfloat162 t0 = __float_as_bfloat162(d0[x]);
            __hip_bfloat162 t1 = __float_as_bfloat162(d1[x]);
            t0.x = __float2bfloat16(__bfloat162float(t0.x) + __bfloat162float(t1.x));
            t0.y = __float2bfloat16(__bfloat162float(t0.y) + __bfloat162float(t1.y));
            d0[x] = __bfloat162_as_float(t0);
        }

        float *__restrict__ pout = reinterpret_cast<float *__restrict__>(p0);
        for(int x=0;x<4;x++) pout[x] = d0[x];

        p0 += stride; p1 += stride;
    }
}

void matmul_cublas(floatX*__restrict__ d, const floatX*__restrict__ a, const floatX*__restrict__ b, const floatX*__restrict__ bias,
                     int m, int n, int k, cudaStream_t stream=0, bool transA=true, bool transB=false,
                     int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
                     bool accumulate=false, floatX*__restrict__ pre_gelu=NULL, bool backward=false)
{
    NVTX_RANGE_FN();
    if (pre_gelu != NULL) { printf("%s: GELU unsupported\n", __PRETTY_FUNCTION__); exit(-1); }

    cublasCheck(cublasSetStream(cublas_handle, stream));

    cublasOperation_t transa = transA? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = transB? CUBLAS_OP_T : CUBLAS_OP_N;

    float one = 1.0f, zero = 0.0f;

    const int lda = transA? k : m;
    const int ldb = transB? n : k;
    const int ldc = m;

    if (batch_count != 0 || strideA != 0 || strideB != 0 || strideOut != 0) {
        if (accumulate) {
            cublasCheck(cublasGemmStridedBatchedEx(cublas_handle, transa, transb, m, n, k, &one,
                        a, CUBLAS_LOWP, lda, strideA, b, CUBLAS_LOWP, ldb, strideB, &one,
                        d, CUBLAS_LOWP, ldc, strideOut, batch_count, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            cublasCheck(cublasGemmStridedBatchedEx(cublas_handle, transa, transb, m, n, k, &one,
                        a, CUBLAS_LOWP, lda, strideA, b, CUBLAS_LOWP, ldb, strideB, &zero,
                        d, CUBLAS_LOWP, ldc, strideOut, batch_count, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    } else {
        if (accumulate) {
            cublasCheck(cublasGemmEx(cublas_handle, transa, transb, m, n, k, &one,
                        a, CUBLAS_LOWP, lda, b, CUBLAS_LOWP, ldb, &one,
                        d, CUBLAS_LOWP, ldc, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            cublasCheck(cublasGemmEx(cublas_handle, transa, transb, m, n, k, &one,
                        a, CUBLAS_LOWP, lda, b, CUBLAS_LOWP, ldb, &zero,
                        d, CUBLAS_LOWP, ldc, cublas_compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }

    }
    if (bias != NULL) {
        add_bias<<<n, 256, 0, stream>>>(d, bias, n, m);
        cudaCheck(cudaGetLastError());
    }

}

#endif

// Wrapper around cublasLtMatmul that is meant to support everything we need in llm.c
// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
void matmul_cublaslt(floatX* d, const floatX* a, const floatX* b, const floatX* bias,
                     int m, int n, int k, cudaStream_t stream=0, bool transA=true, bool transB=false,
                     int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
                     bool accumulate=false, floatX* pre_gelu=NULL, bool backward=false)
{
    NVTX_RANGE_FN();
    bool has_bias = (bias != NULL);
    bool has_gelu = (pre_gelu != NULL);

    // check alignment (some modes work unaligned but it always best to be aligned for performance)
    if(((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
        printf("All cuBLASLt pointers must be aligned!\n");
        exit(EXIT_FAILURE);
    }

    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));

    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;

    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA)  ? &opTranspose : &opNoTranspose,   sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose   : &opNoTranspose, sizeof(opNoTranspose)));

    // define matrix layouts
    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t DLayout;
    cublasLtMatrixLayout_t CLayout;
    if (transA) {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, k, m, k));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, m, k, m));
    }
    if (transB) {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, n, k, n));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, k, n, k));
    }
    // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP, m, n, m));
    cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, CUBLAS_LOWP, m, n, m));

    // Strided Batched GEMM (used for non-flash attention, equivalent to cublasGemmStridedBatchedEx)
    if (batch_count) {
        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
    }

    // create a preference handle with specified max workspace
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // setup epilogue and associated pointers for bias & gelu
    cublasLtEpilogue_t epilogue;
    if (has_gelu) {
        int64_t gelu_ld = m; // todo - is this affected by anything else?
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld, sizeof(gelu_ld)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu, sizeof(pre_gelu)));
        if (backward) {
            assert(!has_bias); // we shouldn't have any backward matmuls that use both GELU and bias
            epilogue = CUBLASLT_EPILOGUE_DGELU;
        } else {
            epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX;
        }
    } else if(has_bias){
        epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
    } else {
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    }
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        // cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
        cublasDataType_t bias_data_type = (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP; // force BF16 bias for FP8 mode
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }
#ifndef BUILD_AMD
    // set scale type to FP32 (needs to be FP16 if and only if using CUBLAS_COMPUTE_16F, so it's FP32 even for FP8!)
    cublasDataType_t scale_type = CUDA_R_32F;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));
#endif
    // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
    cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
                                   preference, 1, &heuristic, &returnedResults);
    if (returnedResults == 0) {
        printf("No cuBLASLt algorithm: m: %d, n: %d, k: %d, bias: %d\n", n, m, k, has_bias);
        exit(EXIT_FAILURE);
    }

    // set whether to accumulate (i.e. D += C) or not - note this isn't considered in algorithm selection (?!)
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;

    // call the matmul
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
                               &alpha, a, ALayout, b, BLayout, &beta, d, CLayout, d, DLayout,
                               &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, stream));

    // cleanups
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));
    cudaCheck(cudaGetLastError());
}

// small wrapper around matmul_cublaslt for the forward pass (keeping historical order of arguments)
void matmul_forward_cublaslt(floatX* out,
                     floatX* inp, floatX* weight, floatX* bias,
                     int B, int T, int C, int OC, cudaStream_t stream,
                     floatX* pre_gelu=NULL, int gelu_fusion=1) {
    // By default only fuse GELU for H100+ as cuBLAS seems to be inefficient for fused GELU on Ada/Ampere (?)
    if (gelu_fusion < 1 && pre_gelu) {
#if defined(USE_CK)
        matmul_ck(pre_gelu, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, NULL, false);
#elif defined(USE_HIPBLAS)
        matmul_cublas(pre_gelu, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, NULL, false);
#else
        matmul_cublaslt(pre_gelu, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, NULL, false);
#endif
        gelu_forward(out, pre_gelu, B*T*OC, stream);
    } else {
#if defined(USE_CK)
        matmul_ck(out, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, pre_gelu, false);
#elif defined(USE_HIPBLAS)
        matmul_cublas(out, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, pre_gelu, false);
#else
        matmul_cublaslt(out, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, pre_gelu, false);
#endif
    }
}

void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                     floatX* dout, floatX* inp, floatX* weight,
                     float* dbias_buffer,
                     int B, int T, int C, int OC, cudaStream_t stream,
                     floatX* pre_gelu=NULL, int gelu_fusion=1) {
    NVTX_RANGE_FN();

    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        // Each warp is responsible for 8 * "x128::size" = 64 OCs at BF16 (OC must be a multiple of 64!)
        // Block size is 1024 | 768 threads (32|24 warps) and we reduce those values into 1 at the end

        const int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

        dim3 block_dim = {4, WARP_SIZE/4, (unsigned)block_size/WARP_SIZE};
        const int OC_per_warp = block_dim.y * x128::size; // 64 at BF16
        const int grid_size_x = CEIL_DIV(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
        const int grid_size_y = max(1, deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / (block_size * grid_size_x)); // full GPU!

        // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
        // and write results directly to the output.
        if(grid_size_y == 1) {
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias, dout, B, T, OC, False);
            cudaCheck(cudaGetLastError());
        } else {
            // kernel 9 overwrites temp buffer, so no need to memset
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias_buffer, dout, B, T, OC, True);
            cudaCheck(cudaGetLastError());
            reduce_add_sum_kernel<<<CEIL_DIV(OC, 256 * f128::size), 256, 0, stream>>>(dbias, dbias_buffer, OC, grid_size_y);
            cudaCheck(cudaGetLastError());
        }
        dbias = NULL; // prevent dbias calculation from also being fused in matmul_cublaslt below (if we enabled fusion)
    }

    // backward to input, uses = in the backward pass (set the gradient)
#if defined(USE_HIPBLAS)
    matmul_cublas(dinp, weight, dout, NULL, C, B*T, OC, stream, false, false, 0, 0, 0, 0, false,
                    gelu_fusion >= 2 ? pre_gelu : NULL, true);
#else
    matmul_cublaslt(dinp, weight, dout, NULL, C, B*T, OC, stream, false, false, 0, 0, 0, 0, false,
                    gelu_fusion >= 2 ? pre_gelu : NULL, true);
#endif

    // backward GELU (if it wasn't fused into the matmul above)
    if (gelu_fusion < 2 && pre_gelu) {
        gelu_backward_inplace(dinp, pre_gelu, B*T*C, stream);
    }

    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
#if defined(USE_HIPBLAS)
    matmul_cublas(dweight, inp, dout, NULL /*dbias*/, C, OC, B*T, stream, false, true, 0, 0, 0, 0,
                    true /* accumulate */, NULL, true);
#else
    matmul_cublaslt(dweight, inp, dout, NULL /*dbias*/, C, OC, B*T, stream, false, true, 0, 0, 0, 0,
                    true /* accumulate */, NULL, true);
#endif
}
