#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

const int MATRIX_DIM = 4096;

__global__ void matmul_kernel(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C)
{
  const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  float tmp = 0.0;
  for (int i = 0; i < K; ++i)
    tmp += A[x * K + i] * B[i * N + y];

    C[x * N + y] = tmp;
}

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main() {
  const int N_TOTAL = MATRIX_DIM * MATRIX_DIM; 
  const size_t bytes = N_TOTAL * sizeof(float);

  const int WARMUP_RUNS = 10;
  const int TIMING_RUNS = 100;

  std::cout << "--- Matrix Matmul Stable Timing Test ---" << std::endl;
  std::cout << "Matrix Dimension: " << MATRIX_DIM << " x " << MATRIX_DIM << std::endl;
  std::cout << "Memory per Matrix: " << (double)bytes / (1024 * 1024) << " MB" << std::endl;

  // Host vectors
  std::vector<float> h_A(N_TOTAL, 1.0f); // Initializing with 1.0 for easy verification
  std::vector<float> h_B(N_TOTAL, 2.0f);
  std::vector<float> h_C(N_TOTAL, 0.0f);

  // Device pointers
  float *d_A, *d_B, *d_C;
  checkCudaError(cudaMalloc(&d_A, bytes), "d_A allocation");
  checkCudaError(cudaMalloc(&d_B, bytes), "d_B allocation");
  checkCudaError(cudaMalloc(&d_C, bytes), "d_C allocation");

  checkCudaError(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice), "A copy H->D");
  checkCudaError(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice), "B copy H->D");
  checkCudaError(cudaMemset(d_C, 0, bytes), "C zeroing");

  dim3 threadsPerBlock(32, 32, 1);
  dim3 numBlocks(MATRIX_DIM / 32, MATRIX_DIM / 32);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::cout << "Warming up..." << std::endl;
  for (int i = 0; i < WARMUP_RUNS; ++i) {
    // Calling C = 1.0 * (A * B) + 0.0 * C
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(MATRIX_DIM, MATRIX_DIM, MATRIX_DIM, 1.0f, d_A, d_B, 0.0f, d_C);
  }
  cudaDeviceSynchronize();

  float total_milliseconds = 0;
  std::cout << "Starting Timing Loop (" << TIMING_RUNS << " runs)..." << std::endl;

  for (int i = 0; i < TIMING_RUNS; ++i) {
    cudaEventRecord(start);
    
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(MATRIX_DIM, MATRIX_DIM, MATRIX_DIM, 1.0f, d_A, d_B, 0.0f, d_C);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    total_milliseconds += ms;
  }

  float avg_ms = total_milliseconds / TIMING_RUNS;
  std::cout << "\nAverage kernel execution time: " << avg_ms << " ms" << std::endl;

  // Copy result back for verification
  checkCudaError(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost), "C copy D->H");

  // Simple Verification (For A=1.0, B=2.0, result should be 1.0 * 2.0 * MATRIX_DIM)
  float expected = (float)MATRIX_DIM * 2.0f; 
  bool passed = true;
  for (int i = 0; i < 10; ++i) { // Check first 10 elements
      if (std::abs(h_C[i] - expected) > 1e-3) {
          passed = false;
          break;
      }
  }

  if (passed) {
      std::cout << "Verification: PASSED (checked sample of results)" << std::endl;
  } else {
      std::cout << "Verification: FAILED" << std::endl;
      std::cout << "Sample Result: " << h_C[0] << " | Expected: " << expected << std::endl;
  }

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}