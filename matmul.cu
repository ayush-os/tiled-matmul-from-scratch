#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

const int MATRIX_DIM = 4096;

template<int TILE_SIZE, int MICRO_TILE>
__global__ void matmul_kernel(int M, int N, int K, float alpha, const float *A,
                              const float *B, float beta, float *C)
{
  __shared__ float A_tile[TILE_SIZE * MICRO_TILE][TILE_SIZE];
  __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

  int x = blockIdx.x * TILE_SIZE + threadIdx.x;
  int y_base = (blockIdx.y * TILE_SIZE * MICRO_TILE) + threadIdx.y;

  float res[MICRO_TILE] = {0.f};
  for (int k = 0; k < K; k += TILE_SIZE) {
    for (int i = 0; i < MICRO_TILE; i++) {
      int load_row = (blockIdx.y * TILE_SIZE * MICRO_TILE) + (i * TILE_SIZE + threadIdx.y);
      int load_col = k + threadIdx.x;
      if (load_row < M && load_col < K) {
        A_tile[i * TILE_SIZE + threadIdx.y][threadIdx.x] = A[load_row * K + load_col];
      }
    }
    if (k + threadIdx.y < K && x < N) {
      B_tile[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + x];
    }

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; i++) {
      float b_val = B_tile[i][threadIdx.x];
      for (int j = 0; j < MICRO_TILE; j++) {
        res[j] += A_tile[threadIdx.y + j * TILE_SIZE][i] * b_val;
      }
    }

    __syncthreads();
  }

  for (int i = 0; i < MICRO_TILE; i++) {
    int y = y_base + (i * TILE_SIZE);
    if (y < M && x < N) {
      C[N * y + x] = res[i];
    }
  }
}

void checkCudaError(cudaError_t err, const char *msg)
{
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

struct Config {
  int tile_size;
  int micro_tile;
  float avg_time_ms;
  float gflops;
  size_t shared_mem_bytes;
  bool valid;
  std::string error_msg;
};

// Calculate shared memory usage
size_t calculate_shared_memory(int tile_size, int micro_tile) {
  size_t a_tile = tile_size * tile_size * micro_tile * sizeof(float);
  size_t b_tile = tile_size * tile_size * sizeof(float);
  return a_tile + b_tile;
}

template<int TILE_SIZE, int MICRO_TILE>
float benchmark_config(float *d_A, float *d_B, float *d_C, int warmup, int timing_runs)
{
  if (MATRIX_DIM % TILE_SIZE != 0 || MATRIX_DIM % (TILE_SIZE * MICRO_TILE) != 0) {
    return -1.0f;
  }

  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, 1);
  dim3 numBlocks(MATRIX_DIM / TILE_SIZE, MATRIX_DIM / (TILE_SIZE * MICRO_TILE));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup
  for (int i = 0; i < warmup; ++i) {
    matmul_kernel<TILE_SIZE, MICRO_TILE><<<numBlocks, threadsPerBlock>>>(
      MATRIX_DIM, MATRIX_DIM, MATRIX_DIM, 1.0f, d_A, d_B, 0.0f, d_C);
  }
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return -1.0f;
  }

  // Timing
  float total_ms = 0;
  for (int i = 0; i < timing_runs; ++i) {
    cudaEventRecord(start);
    matmul_kernel<TILE_SIZE, MICRO_TILE><<<numBlocks, threadsPerBlock>>>(
      MATRIX_DIM, MATRIX_DIM, MATRIX_DIM, 1.0f, d_A, d_B, 0.0f, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    total_ms += ms;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return total_ms / timing_runs;
}

float run_config(int tile_size, int micro_tile, float *d_A, float *d_B, float *d_C, 
                 int warmup, int timing_runs)
{
  // ONLY include configurations that fit in shared memory!
  // Max shared memory is typically 48KB on most GPUs
  // Formula: tile_size² × (micro_tile + 1) × 4 bytes
  
  #define RUN_CONFIG(TS, MT) \
    if (tile_size == TS && micro_tile == MT) \
      return benchmark_config<TS, MT>(d_A, d_B, d_C, warmup, timing_runs);

  // Small tile sizes (all should work)
  RUN_CONFIG(8, 2)    // 1KB
  RUN_CONFIG(8, 4)    // 1.25KB
  RUN_CONFIG(8, 8)    // 2.25KB
  RUN_CONFIG(8, 16)   // 4.25KB
  
  // Medium tile sizes
  RUN_CONFIG(16, 2)   // 3KB
  RUN_CONFIG(16, 4)   // 5KB
  RUN_CONFIG(16, 8)   // 9KB
  RUN_CONFIG(16, 16)  // 17KB
  
  // Larger tile sizes (be careful here)
  RUN_CONFIG(32, 2)   // 12KB
  RUN_CONFIG(32, 4)   // 20KB
  RUN_CONFIG(32, 8)   // 36KB
  
  // DO NOT include 64x4 (80KB) or 64x8 (144KB) - they exceed 48KB limit!
  // If you want to test 64, only use MICRO_TILE=2:
  RUN_CONFIG(64, 2)   // 48KB (might work on some GPUs)

  #undef RUN_CONFIG
  
  return -1.0f;
}

int main()
{
  const int N_TOTAL = MATRIX_DIM * MATRIX_DIM;
  const size_t bytes = N_TOTAL * sizeof(float);

  const int WARMUP_RUNS = 5;
  const int TIMING_RUNS = 20;

  // Get GPU properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  size_t max_shared_mem = prop.sharedMemPerBlock;

  std::cout << "=== CUDA Matrix Multiplication Autotuner ===" << std::endl;
  std::cout << "GPU: " << prop.name << std::endl;
  std::cout << "Max Shared Memory per Block: " << max_shared_mem << " bytes (" 
            << max_shared_mem / 1024 << " KB)" << std::endl;
  std::cout << "Matrix Dimension: " << MATRIX_DIM << " x " << MATRIX_DIM << std::endl;
  std::cout << "Memory per Matrix: " << (double)bytes / (1024 * 1024) << " MB" << std::endl;
  std::cout << std::endl;

  // Host vectors
  std::vector<float> h_A(N_TOTAL, 1.0f);
  std::vector<float> h_B(N_TOTAL, 2.0f);
  std::vector<float> h_C(N_TOTAL, 0.0f);

  // Device pointers
  float *d_A, *d_B, *d_C;
  checkCudaError(cudaMalloc(&d_A, bytes), "d_A allocation");
  checkCudaError(cudaMalloc(&d_B, bytes), "d_B allocation");
  checkCudaError(cudaMalloc(&d_C, bytes), "d_C allocation");

  checkCudaError(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice), "A copy H->D");
  checkCudaError(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice), "B copy H->D");

  // Only test configurations that will compile!
  std::vector<std::pair<int, int>> configs = {
    {8, 2}, {8, 4}, {8, 8}, {8, 16},
    {16, 2}, {16, 4}, {16, 8}, {16, 16},
    {32, 2}, {32, 4}, {32, 8}
  };

  std::vector<Config> results;
  
  std::cout << "Checking configurations..." << std::endl;
  std::cout << std::string(90, '-') << std::endl;
  std::cout << std::setw(12) << "TILE_SIZE" 
            << std::setw(14) << "MICRO_TILE"
            << std::setw(16) << "Shared Mem"
            << std::setw(16) << "Time (ms)"
            << std::setw(16) << "GFLOPS"
            << std::setw(16) << "Status" << std::endl;
  std::cout << std::string(90, '-') << std::endl;

  for (const auto& cfg : configs) {
    int tile_size = cfg.first;
    int micro_tile = cfg.second;

    Config result;
    result.tile_size = tile_size;
    result.micro_tile = micro_tile;
    result.shared_mem_bytes = calculate_shared_memory(tile_size, micro_tile);
    result.valid = false;

    // Check shared memory limit
    if (result.shared_mem_bytes > max_shared_mem) {
      result.error_msg = "Exceeds SHMEM";
      result.avg_time_ms = -1.0f;
      result.gflops = 0.0f;
      
      std::cout << std::setw(12) << tile_size
                << std::setw(14) << micro_tile
                << std::setw(16) << (result.shared_mem_bytes / 1024) << " KB"
                << std::setw(16) << "N/A"
                << std::setw(16) << "N/A"
                << std::setw(16) << "SKIP (SHMEM)" << std::endl;
    }
    // Check divisibility
    else if (MATRIX_DIM % tile_size != 0 || MATRIX_DIM % (tile_size * micro_tile) != 0) {
      result.error_msg = "Invalid dims";
      result.avg_time_ms = -1.0f;
      result.gflops = 0.0f;
      
      std::cout << std::setw(12) << tile_size
                << std::setw(14) << micro_tile
                << std::setw(16) << (result.shared_mem_bytes / 1024) << " KB"
                << std::setw(16) << "N/A"
                << std::setw(16) << "N/A"
                << std::setw(16) << "SKIP (DIMS)" << std::endl;
    }
    else {
      cudaMemset(d_C, 0, bytes);
      
      float avg_time = run_config(tile_size, micro_tile, d_A, d_B, d_C, 
                                   WARMUP_RUNS, TIMING_RUNS);

      if (avg_time > 0) {
        result.valid = true;
        result.avg_time_ms = avg_time;
        double gflops = (2.0 * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM) / 
                        (avg_time * 1e-3) / 1e9;
        result.gflops = gflops;

        std::cout << std::setw(12) << tile_size
                  << std::setw(14) << micro_tile
                  << std::setw(16) << (result.shared_mem_bytes / 1024) << " KB"
                  << std::setw(16) << std::fixed << std::setprecision(3) << avg_time
                  << std::setw(16) << std::fixed << std::setprecision(2) << gflops
                  << std::setw(16) << "OK" << std::endl;
      } else {
        result.error_msg = "Kernel failed";
        std::cout << std::setw(12) << tile_size
                  << std::setw(14) << micro_tile
                  << std::setw(16) << (result.shared_mem_bytes / 1024) << " KB"
                  << std::setw(16) << "N/A"
                  << std::setw(16) << "N/A"
                  << std::setw(16) << "FAILED" << std::endl;
      }
    }

    results.push_back(result);
  }

  std::cout << std::string(90, '-') << std::endl;

  // Find best configuration
  auto best = std::max_element(results.begin(), results.end(),
    [](const Config& a, const Config& b) {
      if (!a.valid) return true;
      if (!b.valid) return false;
      return a.gflops < b.gflops;
    });

  if (best != results.end() && best->valid) {
    std::cout << "\n*** BEST CONFIGURATION ***" << std::endl;
    std::cout << "TILE_SIZE: " << best->tile_size << std::endl;
    std::cout << "MICRO_TILE: " << best->micro_tile << std::endl;
    std::cout << "Shared Memory: " << best->shared_mem_bytes << " bytes (" 
              << best->shared_mem_bytes / 1024 << " KB)" << std::endl;
    std::cout << "Average Time: " << std::fixed << std::setprecision(3) 
              << best->avg_time_ms << " ms" << std::endl;
    std::cout << "Performance: " << std::fixed << std::setprecision(2) 
              << best->gflops << " GFLOPS" << std::endl;

    // Verify correctness
    checkCudaError(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost), "C copy D->H");
    float expected = (float)MATRIX_DIM * 2.0f;
    bool passed = true;
    for (int i = 0; i < 100; ++i) {
      if (std::abs(h_C[i] - expected) > 1e-3) {
        passed = false;
        break;
      }
    }
    std::cout << "Verification: " << (passed ? "PASSED" : "FAILED") << std::endl;
  } else {
    std::cout << "\nNo valid configuration found!" << std::endl;
  }

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}