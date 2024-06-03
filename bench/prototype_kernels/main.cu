#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <time.h>

#define DO_DEBUG 0

#include "util.h"
#include "kernels.cu"

#ifndef USE_EPILOGUE
#define USE_EPILOGUE true
#endif

#ifdef TA
constexpr int Ta = TA;
constexpr int Tb = TB;
constexpr int Tc = TC;
constexpr int Ti = TI;
constexpr int Tj = TJ;
constexpr int Tk = TK;
constexpr int Q  = QQ;
constexpr int Ra = RA;
constexpr int Rb = RB;
constexpr int Rc = RC;
constexpr int Ri = RI;
constexpr int Rj = RJ;
constexpr int Rk = RK;
#else

constexpr int Ta = 1;
constexpr int Tb = 1;
constexpr int Tc = 1;
constexpr int Ti = 8;
constexpr int Tj = 1;
constexpr int Tk = 32;
constexpr int Q  = 4;
constexpr int Ra = 4;
constexpr int Rb = 1;
constexpr int Rc = 1;
constexpr int Ri = 2;
constexpr int Rj = 2;
constexpr int Rk = 1;

#endif

constexpr int NUM_RUNS = 200;
constexpr bool do_validate = 0;

void cpu_TC(
  int *g_A,
  int *g_B,
  int *g_C,
  const int Na,
  const int Nb,
  const int Nc,
  const int Ni,
  const int Nj,
  const int Nk,
  const int Nq) {

  for (int a = 0; a < Na; a++)
  for (int b = 0; b < Nb; b++)
  for (int c = 0; c < Nc; c++)
  for (int i = 0; i < Ni; i++)
  for (int j = 0; j < Nj; j++)
  for (int k = 0; k < Nk; k++) {
    int acc = 0;
    for (int q = 0; q < Nq; q++)
      acc += idx4(g_A, Ni, Nc, Na, Nq, i, c, a, q) *
             idx4(g_B, Nq, Nb, Nj, Nk, q, b, j, k);
    idx6(g_C, Na, Nb, Nc, Ni, Nj, Nk,
              a, b, c, i, j, k) = acc;
  }
}

template<class ElTp>
void run(size_t Na, size_t Nb, size_t Nc, size_t Ni, size_t Nj, size_t Nk, size_t Nq) {
  static_assert(std::is_arithmetic<ElTp>::value);

  printf("%d %d %d %d %d %d %d ", Na, Nb, Nc, Ni, Nj, Nk, Nq);

  double work = 2.0 * Na * Nb * Nc * Ni * Nj * Nk * Nq;
  size_t C_len = Na * Nb * Nc * Ni * Nj * Nk;

  ElTp *d_A, *d_B, *d_C, *ref_C, *A, *B, *C;

  constexpr int TRa = Ta * Ra, TRb = Tb * Rb, TRc = Tc * Rc,
                TRi = Ti * Ri, TRj = Tj * Rj, TRk = Tk * Rk;
  int grid_dim_y = CEIL_DIV(Na, TRa) * CEIL_DIV(Nb, TRb) * CEIL_DIV(Nc, TRc);
  int grid_dim_x = CEIL_DIV(Ni, TRi) * CEIL_DIV(Nj, TRj) * CEIL_DIV(Nk, TRk);
  int tblock_dim_y = Ta * Tb * Tc, tblock_dim_x = Ti * Tj * Tk;

  if (COMPILE_KERNEL == 0) { //  "icaq_qbjk_abcijk"
    size_t A_len = Ni * Nc * Na * Nq;
    size_t B_len = Nq * Nb * Nj * Nk;
    int s_A_size = TRi * TRc * TRa * Q;
    int s_B_size = Q * TRb * TRj * TRk;
    RUN_KERNEL(icaq_qbjk_abcijk, false);
  }
  else if (COMPILE_KERNEL == 1) { // "kiaq_bcjq_abcijk"

    size_t A_len = Nk * Ni * Na * Nq;
    size_t B_len = Nb * Nc * Nj * Nq;
    int s_A_size = TRk * (TRi * TRa * Q + 1);
    int s_B_size = TRb * TRc * TRj * Q;
    RUN_KERNEL(kiaq_bcjq_abcijk, false);
  }
  else if (COMPILE_KERNEL == 2) { // "aq_dq_ab"
    int Nc = 1, Ni = 1, Nj = 1, Nk = 1;
    int grid_dim_y = CEIL_DIV(Na, TRa);
    int grid_dim_x = CEIL_DIV(Nb, TRb);
    int tblock_dim_y = Ta, tblock_dim_x = Tb;

    size_t A_len = Na * Nq;
    size_t B_len = Nq * Nb;

    int s_A_size = TRa * Q;
    int s_B_size = Q * TRb;
    RUN_KERNEL(aq_qb_ab, true);
  }
  else
    printf("Error: unknown kernel: \"%d\"\n", COMPILE_KERNEL);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}


int main(int argc, char **argv) {
  srand(time(NULL));

  if (argc != 8) {
    fprintf(stderr,
        "Usage: %s <Na> <Nb> <Nc> <Ni> <Nj> <Nk> <Nq>\n",
        argv[0]);
    exit(0);
  }
    
  size_t Na = atol(argv[1]);
  size_t Nb = atol(argv[2]);
  size_t Nc = atol(argv[3]);
  size_t Ni = atol(argv[4]);
  size_t Nj = atol(argv[5]);
  size_t Nk = atol(argv[6]);
  size_t Nq = atol(argv[7]);

  if constexpr(do_validate)
    run<int>(Na, Nb, Nc, Ni, Nj, Nk, Nq);
  else
    run<float>(Na, Nb, Nc, Ni, Nj, Nk, Nq);
}

