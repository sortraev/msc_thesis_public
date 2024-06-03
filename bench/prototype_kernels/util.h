#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cassert>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CEIL_DIV(n, m) (((n) + (m) - 1) / (m))

#define idx6(X, n0, n1, n2, n3, n4, n5, i0, i1, i2, i3, i4, i5)                \
  ((X)[(((((i0) * (n1) + (i1)) * (n2) + (i2)) * (n3) + (i3))                   \
       * (n4) + (i4)) * (n5) + (i5)])

#define idx5(X, n0, n1, n2, n3, n4, i0, i1, i2, i3, i4)                        \
  idx6(X, 0, n0, n1, n2, n3, n4, 0, i0, i1, i2, i3, i4)

#define idx4(X, n0, n1, n2, n3, i0, i1, i2, i3)                                \
  idx5(X, 0, n0, n1, n2, n3, 0, i0, i1, i2, i3)

#define idx3(X, n0, n1, n2, i0, i1, i2)                                        \
  idx4(X, 0, n0, n1, n2, 0, i0, i1, i2)

#define idx2(X, n0, n1, i0, i1)                                                \
  idx3(X, 0, n0, n1, 0, i0, i1)

#define idx4_(X, s0, s1, s2, s3, i0, i1, i2, i3)                               \
  ((X)[(i0) * (s0) + (i1) * (s1) + (i2) * (s2) + (i3) * (s3)])

#define NEAR_INT_MAX(x) ((x) >= (INT_MAX - 16384))

#define print(...)

#define RUN_KERNEL(KERNEL_NAME, is_MM)                                         \
  do {                                                                         \
    if (NEAR_INT_MAX(A_len) || NEAR_INT_MAX(B_len) || NEAR_INT_MAX(C_len)) {   \
      printf("\033[1;31mWarning: one or more input sizes near INT32_MAX!\033[0m\n");\
    }                                                                          \
    CUDASSERT(cudaMalloc((void**) &d_A, A_len * sizeof(ElTp)));                \
    CUDASSERT(cudaMalloc((void**) &d_B, B_len * sizeof(ElTp)));                \
    CUDASSERT(cudaMalloc((void**) &d_C, C_len * sizeof(ElTp)));                \
    if (!d_A || !d_B || !d_C) {                                                \
      printf("failed to allocate dev mem!\n");                                 \
      exit(1);                                                                 \
    }                                                                          \
    random_init_dev<ElTp>(d_A, A_len, -50, 50);                                \
    random_init_dev<ElTp>(d_B, B_len, -50, 50);                                \
    dim3 grid_dim(grid_dim_x, grid_dim_y, 1);                                  \
    dim3 tblock_dim(tblock_dim_x, tblock_dim_y, 1);                            \
    int shmem_size = (s_A_size + s_B_size) * sizeof(ElTp);                     \
    if (shmem_size > 49152) {                                                  \
      cudaFuncSetAttribute(                                                    \
          KERNEL_NAME                                                          \
          <ElTp, Ta, Tb, Tc, Ti, Tj, Tk, Q, Ra, Rb, Rc, Ri, Rj, Rk,            \
          USE_EPILOGUE, AUTO>,                                                 \
          cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);            \
    }                                                                          \
    /* warmup run */                                                           \
    CUDASSERT(cudaMemset((void*) d_C, 69, C_len * sizeof(ElTp)));              \
    for (int i = 0; i < NUM_RUNS; i++)                                         \
      KERNEL_NAME                                                              \
        <ElTp, Ta, Tb, Tc, Ti, Tj, Tk, Q, Ra, Rb, Rc, Ri, Rj, Rk,              \
          USE_EPILOGUE, AUTO>                                                  \
        <<<grid_dim, tblock_dim, shmem_size>>>                                 \
        (d_A, d_B, d_C, Na, Nb, Nc, Ni, Nj, Nk, Nq);                           \
    cudaDeviceSynchronize();                                                   \
    CUDASSERT(cudaPeekAtLastError());                                          \
    /* benchmark runs */                                                       \
    double elapsed;                                                            \
    struct timeval t_start, t_end, t_diff;                                     \
    gettimeofday(&t_start, NULL);                                              \
    for (int i = 0; i < NUM_RUNS; i++)                                         \
      KERNEL_NAME                                                              \
        <ElTp, Ta, Tb, Tc, Ti, Tj, Tk, Q, Ra, Rb, Rc, Ri, Rj, Rk,              \
          USE_EPILOGUE, AUTO>                                                  \
        <<<grid_dim, tblock_dim, shmem_size>>>                                 \
        (d_A, d_B, d_C, Na, Nb, Nc, Ni, Nj, Nk, Nq);                           \
    CUDASSERT(cudaDeviceSynchronize());                                        \
    gettimeofday(&t_end, NULL);                                                \
    /* error check and compute elapsed */                                      \
    CUDASSERT(cudaPeekAtLastError());                                          \
    timeval_subtract(&t_diff, &t_end, &t_start);                               \
    elapsed = (t_diff.tv_sec * 1e6 + t_diff.tv_usec) / ((double)NUM_RUNS);     \
    double gflops = (work * 1e-3f) / elapsed;                                  \
    if constexpr(is_MM)                                                        \
      printf("%d %d %d %d %d %lf\n", Ta, Tb, Q, Ra, Rb, gflops);               \
    else                                                                       \
      printf("(%d) %d %d %d %d %d %d %d %d %d %d %d %d %d %lf\n",              \
              USE_EPILOGUE, Ta, Tb, Tc, Ti, Tj, Tk, Q, Ra, Rb, Rc, Ri, Rj, Rk, gflops);     \
    /* validation */                                                           \
    if constexpr (do_validate) {                                               \
      printf("Validation on -- running sequential reference ...");             \
      fflush(stdout);                                                          \
      cpu_TC(A, B, ref_C, Na, Nb, Nc, Ni, Nj, Nk, Nq);                         \
      printf(" done\n");                                                       \
      CUDASSERT(cudaMemcpy(C, d_C, C_len * sizeof(ElTp),                       \
            cudaMemcpyDeviceToHost));                                          \
      printf("%s validation: ", (#KERNEL_NAME));                               \
      validate((int*)ref_C, (int*)C, C_len);                                   \
      /* clear result arrays before next run */                                \
      memset    ((void*) C,   0, C_len * sizeof(ElTp));                        \
      CUDASSERT(cudaMemset((void*) d_C, 0, C_len * sizeof(ElTp)));             \
    }                                                                          \
  } while (0)

enum padMethod {
	ON = 0,
	OFF = 1,
	AUTO = 2
};


bool validate(int *ref, int *actual, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (ref[i] != actual[i]) {
      std::cerr << "INVALID! Printing next 10 ...\nindex: reference, actual\n";
      for (size_t j = i; j < i + 10; j++)
        std::cerr << j << ": " << ref[j] << ", " << actual[j] << "\n";
      return false;
    }
  std::cout << "VALID RESULT!" << std::endl;
  return true;
}

int timeval_subtract(struct timeval *result, struct timeval *t2,
                     struct timeval *t1) {
  unsigned int resolution = 1000000;
  long int diff = (t2->tv_usec + resolution * t2->tv_sec) -
                  (t1->tv_usec + resolution * t1->tv_sec);
  result->tv_sec = diff / resolution;
  result->tv_usec = diff % resolution;
  return (diff < 0);
}

template <class ElTp>
__global__ void random_init_kernel(ElTp *xs, int n, int lo, int hi) {
  constexpr int PRAND_M = 314741309;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    int prand = gid * PRAND_M;
    xs[gid] = (ElTp) ((prand % (hi + 1 - lo)) + lo);
  }
}

template <class ElTp>
void random_init_dev(ElTp *xs, int n, int lo, int hi) {
  const int B = 256;
  random_init_kernel<ElTp><<<CEIL_DIV(n, B), B>>>(xs, n, lo, hi);
  cudaDeviceSynchronize();
}

void CUDASSERT(cudaError_t code) {
  if (code != cudaSuccess) {
    fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(code));
    exit(code);
  }
}

#endif // UTIL_H
