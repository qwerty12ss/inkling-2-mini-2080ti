// Probe cuBLASLt algorithms on the actual SM75 shapes. No weights are recast.
#include <cublasLt.h>
#include <array>
#include <cstddef>
#include <memory>

struct MatrixPlan {
    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t op{};
    cublasLtMatrixLayout_t weight{}, input{}, output{};
    cublasLtMatmulPreference_t preference{};
    std::array<cublasLtMatmulHeuristicResult_t, 48> algorithms{};
    int count{};
    size_t workspace{};
    ~MatrixPlan() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (output) cublasLtMatrixLayoutDestroy(output);
        if (input) cublasLtMatrixLayoutDestroy(input);
        if (weight) cublasLtMatrixLayoutDestroy(weight);
        if (op) cublasLtMatmulDescDestroy(op);
        if (handle) cublasLtDestroy(handle);
    }
};

#define CHECK_LT(expr) do { const auto status = (expr); if (status != CUBLAS_STATUS_SUCCESS) return int(status); } while (false)

extern "C" int make_plan(int m, int n, int k, int batch, bool float_output,
                         size_t workspace, MatrixPlan** result, int* count) {
    auto plan = std::make_unique<MatrixPlan>();
    plan->workspace = workspace;
    CHECK_LT(cublasLtCreate(&plan->handle));
    CHECK_LT(cublasLtMatmulDescCreate(&plan->op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const auto trans = CUBLAS_OP_T;
    CHECK_LT(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    CHECK_LT(cublasLtMatrixLayoutCreate(&plan->weight, CUDA_R_16F, k, n, k));
    CHECK_LT(cublasLtMatrixLayoutCreate(&plan->input, CUDA_R_16F, k, m, k));
    CHECK_LT(cublasLtMatrixLayoutCreate(&plan->output, float_output ? CUDA_R_32F : CUDA_R_16F, n, m, n));
    if (batch > 1) {
        const long long strides[] = {1LL*k*n,1LL*k*m,1LL*n*m};
        const cublasLtMatrixLayout_t layouts[] = {plan->weight,plan->input,plan->output};
        for (int i=0; i<3; ++i) {
            CHECK_LT(cublasLtMatrixLayoutSetAttribute(layouts[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
            CHECK_LT(cublasLtMatrixLayoutSetAttribute(layouts[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strides[i], sizeof(strides[i])));
        }
    }
    CHECK_LT(cublasLtMatmulPreferenceCreate(&plan->preference));
    CHECK_LT(cublasLtMatmulPreferenceSetAttribute(plan->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspace, sizeof(workspace)));
    CHECK_LT(cublasLtMatmulAlgoGetHeuristic(plan->handle, plan->op, plan->weight, plan->input,
                                            plan->output, plan->output, plan->preference,
                                            plan->algorithms.size(), plan->algorithms.data(), &plan->count));
    *count = plan->count;
    *result = plan.release();
    return 0;
}
extern "C" int run_plan(const MatrixPlan* plan, int candidate, void* x, void* w, void* y,
                        void* workspace, cudaStream_t stream) {
    if (candidate < 0 || candidate >= plan->count) return -1;
    const auto& algo = plan->algorithms[candidate];
    if (algo.state != CUBLAS_STATUS_SUCCESS) return int(algo.state);
    const float alpha = 1.0f, beta = 0.0f;
    return int(cublasLtMatmul(plan->handle,plan->op,&alpha,w,plan->weight,x,plan->input,
                              &beta,y,plan->output,y,plan->output,&algo.algo,workspace,plan->workspace,stream));
}
extern "C" void free_plan(MatrixPlan* plan) { delete plan; }
