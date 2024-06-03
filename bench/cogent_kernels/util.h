#ifndef UTIL_H
#define UTIL_H

// #define MIN(a, b) ((a) < (b) ? (a) : (b))
// #define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CEIL_DIV(n, m) (((n) + (m) - 1) / (m))

// #define print(...) printf(__VA_ARGS__)
#define print(...)

#define RUN_KERNEL(KERNEL_NAME)                                                \
  do {                                                                         \
    print(">> Running \"%s\" ...\n", (#KERNEL_NAME));                          \
    print(">> CONFIG\n");                                                      \
    print("      a   b   c   i   j   k   d\n");                                \
    print("  N %3d %3d %3d %3d %3d %3d %3d\n", Na, Nb, Nc, Ni, Nj, Nk, Nq);    \
    print("\n  Work = %ld flops\n\n", (uint64_t) work);                        \
    /* warmup runs */                                                          \
    for (int i = 0; i < 1; i++)                                               \
      KERNEL_NAME                                                              \
        (Na, Nb, Nc, Ni, Nj, Nk, Nq, d_A, d_B, d_C);                           \
    cudaDeviceSynchronize();                                                   \
    /* setup benchmark */                                                      \
    double elapsed;                                                            \
    struct timeval t_start, t_end, t_diff;                                     \
    /* run benchmark */                                                        \
    gettimeofday(&t_start, NULL);                                              \
    for (int i = 0; i < NUM_RUNS; i++)                                         \
      KERNEL_NAME                                                              \
        (Na, Nb, Nc, Ni, Nj, Nk, Nq, d_A, d_B, d_C);                           \
    cudaDeviceSynchronize();                                                   \
    gettimeofday(&t_end, NULL);                                                \
    /* error check and compute elapsed */                                      \
    CUDASSERT(cudaPeekAtLastError());                                          \
    timeval_subtract(&t_diff, &t_end, &t_start);                               \
    elapsed = (t_diff.tv_sec * 1e6 + t_diff.tv_usec) / ((double)NUM_RUNS);     \
    double gflops = (work * 1e-3f) / elapsed;                                  \
    printf("work = %f\n", work);\
    printf("%.2f\n", gflops);                                                  \
    print("%s:  %.2f GFlop/s (%.2f microsecs)\n",                              \
        (#KERNEL_NAME), gflops, elapsed);                                      \
  } while (0)

void CUDASSERT(cudaError_t code) {
  if (code != cudaSuccess) {
    fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(code));
    exit(code);
  }
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

#endif // UTIL_H
