#include <iostream>

#include "util.h"
#include "cogent_icaq_qbjk_abcijk_32x32x32x32x32x32x32.cu"
#include "cogent_kiaq_bcjq_abcijk_32x32x32x32x32x32x32.cu"

#include "cogent_icaq_qbjk_abcijk_16x16x16x16x16x16x2048.cu"
#include "cogent_kiaq_bcjq_abcijk_16x16x16x16x16x16x2048.cu"

#define NUM_RUNS 200

template <class ElTp>
void run(int Na, int Nb, int Nc, int Ni, int Nj, int Nk, int Nq, char *kernel_name) {
  static_assert(std::is_arithmetic<ElTp>::value);

  double work = 2.0 * Na * Nb * Nc * Ni * Nj * Nk * Nq;
  size_t size_A = Na * Ni * Nj * Nq;
  size_t size_B = Nb * Nc * Nk * Nq;
  size_t size_C = Na * Nb * Nc * Ni * Nj * Nk;

  // prepare device memory
  ElTp *d_A, *d_B, *d_C;
  CUDASSERT(cudaMalloc((void **)&d_A, size_A * sizeof(ElTp)));
  CUDASSERT(cudaMalloc((void **)&d_B, size_B * sizeof(ElTp)));
  CUDASSERT(cudaMalloc((void **)&d_C, size_C * sizeof(ElTp)));

  random_init_dev<ElTp>(d_A, size_A, -50, 50);
  random_init_dev<ElTp>(d_B, size_B, -50, 50);

  printf("%s ", kernel_name);

  if (strcmp(kernel_name, "icaq_qbjk_abcijk_32x32x32x32x32x32x32") == 0) {
    RUN_KERNEL(icaq_qbjk_abcijk_32x32x32x32x32x32x32);
  }
  else if (strcmp(kernel_name, "kiaq_bcjq_abcijk_32x32x32x32x32x32x32") == 0) {
    RUN_KERNEL(kiaq_bcjq_abcijk_32x32x32x32x32x32x32);
  }
  else if (strcmp(kernel_name, "icaq_qbjk_abcijk_16x16x16x16x16x16x2048") == 0) {
    RUN_KERNEL(icaq_qbjk_abcijk_16x16x16x16x16x16x2048);
  }
  else if (strcmp(kernel_name, "kiaq_bcjq_abcijk_16x16x16x16x16x16x2048") == 0) {
    RUN_KERNEL(kiaq_bcjq_abcijk_16x16x16x16x16x16x2048);
  }

  CUDASSERT(cudaFree(d_A));
  CUDASSERT(cudaFree(d_B));
  CUDASSERT(cudaFree(d_C));
}

int main(int argc, char **argv) {
  srand(2);

  if (argc != 9) {
    fprintf(stderr,
          "Usage: %s <Na> <Nb> <Nc> <Ni> <Nj> <Nk> <Nq> <kernel name>\n",
        argv[0]);
    exit(0);
  }

  int Na = atoi(argv[1]);
  int Nb = atoi(argv[2]);
  int Nc = atoi(argv[3]);
  int Ni = atoi(argv[4]);
  int Nj = atoi(argv[5]);
  int Nk = atoi(argv[6]);
  int Nq = atoi(argv[7]);
  char *kernel_name = argv[8];

  run<float>(Na, Nb, Nc, Ni, Nj, Nk, Nq, kernel_name);
}
