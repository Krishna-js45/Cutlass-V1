#include <cutlass/gemm/device/gemm.h>

using Gemm = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, 
                                         float, cutlass::layout::RowMajor, 
                                         float, cutlass::layout::RowMajor>;

void run_cutlass_gemm() {
    // Define matrix sizes
    int M = 128, N = 128, K = 128;
    float alpha = 1.0f, beta = 0.0f;
    
    // Allocate device memory for matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Define GEMM arguments
    Gemm::Arguments args({M, N, K}, d_A, d_B, d_C, d_C, {alpha, beta});

    // Execute CUTLASS GEMM
    Gemm gemm_op;
    cutlass::Status status = gemm_op(args);

    // Check for success
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed!" << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
