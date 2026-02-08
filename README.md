# tiled-matmul-from-scratch

## baseline: 43.839 ms

## shared memory tiling:  32.834 ms (1.34x speedup)

### each thread does 8 via 1d blocktiling: 20.075 ms (1.64x speedup)

### autotuning results

paperspace@ps5j5ugvo7os:~/tiled-matmul-from-scratch$ ./bin/matmul 
=== CUDA Matrix Multiplication Autotuner ===
GPU: NVIDIA A100-SXM4-80GB
Max Shared Memory per Block: 49152 bytes (48 KB)
Matrix Dimension: 4096 x 4096
Memory per Matrix: 64 MB

Checking configurations...
------------------------------------------------------------------------------------------
   TILE_SIZE    MICRO_TILE      Shared Mem       Time (ms)          GFLOPS          Status
------------------------------------------------------------------------------------------
           8             2               0 KB          49.495         2776.81              OK
           8             4               1 KB          32.282         4257.49              OK
           8             8               2 KB          28.291         4858.13              OK
           8            16               4 KB          26.774         5133.31              OK
          16             2               3 KB          31.753         4328.37              OK
          16             4               5 KB          22.436         6125.71              OK
          16             8               9 KB          20.072         6847.35              OK
          16            16              17 KB          19.186         7163.42              OK
          32             2              12 KB          24.973         5503.41              OK
          32             4              20 KB          20.045         6856.45              OK
          32             8              36 KB          17.803         7719.81              OK
------------------------------------------------------------------------------------------

*** BEST CONFIGURATION ***
TILE_SIZE: 32
MICRO_TILE: 8
Shared Memory: 36864 bytes (36 KB)
Average Time: 17.803 ms
Performance: 7719.81 GFLOPS
Verification: PASSED


FINAL SPEED 17.803 ms