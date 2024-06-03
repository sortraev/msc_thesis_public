// created by tc_code_include() in tc_code_include.py
#include <algorithm>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
using namespace std;

// created by tc_gen_definition_new()
#define SIZE_SLICE_1_Q 16
#define SIZE_SLICE_1_K 16
#define SIZE_SLICE_1_J 4
#define SIZE_SLICE_1_B 1
#define SIZE_SLICE_1_A 8
#define SIZE_SLICE_1_C 8
#define SIZE_SLICE_1_I 1

#define SIZE_INT_UNIT_1 SIZE_SLICE_1_Q

#define SIZE_TB_1_X SIZE_SLICE_1_K *SIZE_SLICE_1_B
#define SIZE_TB_1_Y SIZE_SLICE_1_A *SIZE_SLICE_1_I
#define SIZE_REG_1_X SIZE_SLICE_1_J
#define SIZE_REG_1_Y SIZE_SLICE_1_C

#define NUM_INDEX 6

// created by tc_gen_code_Kernel()
__global__ void icaq_qbjk_abcijk_32x32x32x32x32x32x32_1(float *dev_t3, float *dev_t2, float *dev_v2,
                            int size_k, int size_j, int size_i, int size_c,
                            int size_b, int size_a, int size_q, int numBlk_k,
                            int numBlk_j, int numBlk_i, int numBlk_c,
                            int numBlk_b, int numBlk_a, int stride_int_t2,
                            int stride_int_v2, int stride_reg_x,
                            int stride_reg_y, int size_internal) {
  // For Shared Memory,
  __shared__ float sm_a[16][64];
  __shared__ float sm_b[16][64];

  // when opt_pre_computed == -1, all indices will be calculated manually
  // # of indices mapped on TB_X: 2
  // # of indices mapped on TB_Y: 2
  int idx_k = threadIdx.x % SIZE_SLICE_1_K;
  int idx_b = threadIdx.x / SIZE_SLICE_1_K;
  int idx_a = threadIdx.y % SIZE_SLICE_1_A;
  int idx_i = threadIdx.y / SIZE_SLICE_1_A;

  int tmp_blkIdx;
  int blk_idx_a =
      blockIdx.x / (numBlk_b * numBlk_c * numBlk_i * numBlk_j * numBlk_k);
  tmp_blkIdx =
      blockIdx.x % (numBlk_b * numBlk_c * numBlk_i * numBlk_j * numBlk_k);

  int blk_idx_b = tmp_blkIdx / (numBlk_c * numBlk_i * numBlk_j * numBlk_k);
  tmp_blkIdx = tmp_blkIdx % (numBlk_c * numBlk_i * numBlk_j * numBlk_k);

  int blk_idx_c = tmp_blkIdx / (numBlk_i * numBlk_j * numBlk_k);
  tmp_blkIdx = tmp_blkIdx % (numBlk_i * numBlk_j * numBlk_k);

  int blk_idx_i = tmp_blkIdx / (numBlk_j * numBlk_k);
  tmp_blkIdx = tmp_blkIdx % (numBlk_j * numBlk_k);

  int blk_idx_j = tmp_blkIdx / numBlk_k;
  tmp_blkIdx = tmp_blkIdx % (numBlk_k);

  int blk_idx_k = tmp_blkIdx;

  int t3_base_thread = blk_idx_k * SIZE_SLICE_1_K + idx_k +
                       (blk_idx_j * SIZE_SLICE_1_J +
                        (blk_idx_i * SIZE_SLICE_1_I + idx_i +
                         (blk_idx_c * SIZE_SLICE_1_C +
                          (blk_idx_b * SIZE_SLICE_1_B + idx_b +
                           (blk_idx_a * SIZE_SLICE_1_A + idx_a) * size_b) *
                              size_c) *
                             size_i) *
                            size_j) *
                           size_k;

  float temp_av;
  float temp_bv[8];
  float reg_tile[8][4];

  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 4; j++)
      reg_tile[i][j] = 0.0;

// tensor contraction: [[16, 'STR_SD2_T2_H7', 'y', 't2', ['q', 'a', 'c', 'i']],
// [16, 'STR_SD2_V2_H7', 'x', 'v2', ['k', 'j', 'b', 'q']], '+=']
#pragma unroll 1
  for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1) {
    //---------------------------------------------------------------------------------------------------
    // This is for the new version
    // This Part is for Loading Input-Left
    // tc_gen_code_Kernel_Load_Inputs_Abstracts()
    // No Need to Put Boundary-Checks before For-Statement: :
    for (int ll = 0; ll < 8; ll++) {
      // ['q', 'a', 'c', 'i']
      // Exception: Temp. version!: threadIdx.x + l
      // Exception: Temp. version!: idx_a < rng_a
      sm_a[threadIdx.x][threadIdx.y + ll * 8] =
          dev_t2[(blk_idx_a * SIZE_SLICE_1_A + idx_a +
                  (blk_idx_c * SIZE_SLICE_1_C + ll +
                   (blk_idx_i * SIZE_SLICE_1_I + 0) * size_c) *
                      size_a) *
                     size_q +
                 (threadIdx.x + l)];
    }

    // This Part is for Loading Input-Right
    // tc_gen_code_Kernel_Load_Inputs_Abstracts()
    // No Need to Put Boundary-Checks before For-Statement: :
    for (int ll = 0; ll < 4; ll++) {
      // ['k', 'j', 'b', 'q']
      // Exception: Temp. version!: threadIdx.y + l + 0
      // Exception: Temp. version!: idx_k < rng_k
      sm_b[threadIdx.y + 0][threadIdx.x + ll * 16] =
          dev_v2[blk_idx_k * SIZE_SLICE_1_K + idx_k +
                 (blk_idx_j * SIZE_SLICE_1_J + ll +
                  (blk_idx_b * SIZE_SLICE_1_B + 0) * size_j) *
                     size_k +
                 (threadIdx.y + l + 0) * stride_int_v2];
      // Exception: Temp. version!: threadIdx.y + l + 8
      // Exception: Temp. version!: idx_k < rng_k
      // Exception: Full-Full
      sm_b[threadIdx.y + 8][threadIdx.x + ll * 16] =
          dev_v2[blk_idx_k * SIZE_SLICE_1_K + idx_k +
                 (blk_idx_j * SIZE_SLICE_1_J + ll +
                  (blk_idx_b * SIZE_SLICE_1_B + 0) * size_j) *
                     size_k +
                 (threadIdx.y + l + 8) * stride_int_v2];
    }
    __syncthreads();
    //---------------------------------------------------------------------------------------------------

    // Part: Generalized Threads
    for (int ll = 0; ll < SIZE_INT_UNIT_1; ll++) {
      temp_bv[0] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 0];
      temp_bv[1] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 8];
      temp_bv[2] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 16];
      temp_bv[3] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 24];
      temp_bv[4] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 32];
      temp_bv[5] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 40];
      temp_bv[6] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 48];
      temp_bv[7] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 56];

      for (int xx = 0; xx < 4; xx++) // (1)
      {
        temp_av = sm_b[ll][idx_k + (idx_b)*SIZE_SLICE_1_K + (xx * 16)];

        reg_tile[0][xx] += temp_av * temp_bv[0];
        reg_tile[1][xx] += temp_av * temp_bv[1];
        reg_tile[2][xx] += temp_av * temp_bv[2];
        reg_tile[3][xx] += temp_av * temp_bv[3];
        reg_tile[4][xx] += temp_av * temp_bv[4];
        reg_tile[5][xx] += temp_av * temp_bv[5];
        reg_tile[6][xx] += temp_av * temp_bv[6];
        reg_tile[7][xx] += temp_av * temp_bv[7];
      }
    }
    __syncthreads();
  }

// Store Results (Registers) to Global Memory
// Part: Generalized Threads
// Part: Generalized Register-Tiling
#pragma unroll 8
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 4; j++) {
      dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] =
          reg_tile[i][j];
    }
  }
}

// created by tc_gen_code_Kernel()
__global__ void icaq_qbjk_abcijk_32x32x32x32x32x32x32_2(float *dev_t3, float *dev_t2, float *dev_v2,
                            int size_k, int size_j, int size_i, int size_c,
                            int size_b, int size_a, int size_q, int numBlk_k,
                            int numBlk_j, int numBlk_i, int numBlk_c,
                            int numBlk_b, int numBlk_a, int stride_int_t2,
                            int stride_int_v2, int stride_reg_x,
                            int stride_reg_y, int size_internal) {
  // For Shared Memory,
  __shared__ float sm_a[16][64];
  __shared__ float sm_b[16][64];

  int internal_upperbound = 0;
  int internal_offset;

  // when opt_pre_computed == -1, all indices will be calculated manually
  // # of indices mapped on TB_X: 2
  // # of indices mapped on TB_Y: 2
  int idx_k = threadIdx.x % SIZE_SLICE_1_K;
  int idx_b = threadIdx.x / SIZE_SLICE_1_K;
  int idx_a = threadIdx.y % SIZE_SLICE_1_A;
  int idx_i = threadIdx.y / SIZE_SLICE_1_A;

  int tmp_blkIdx;
  int blk_idx_a =
      blockIdx.x / (numBlk_b * numBlk_c * numBlk_i * numBlk_j * numBlk_k);
  tmp_blkIdx =
      blockIdx.x % (numBlk_b * numBlk_c * numBlk_i * numBlk_j * numBlk_k);

  int blk_idx_b = tmp_blkIdx / (numBlk_c * numBlk_i * numBlk_j * numBlk_k);
  tmp_blkIdx = tmp_blkIdx % (numBlk_c * numBlk_i * numBlk_j * numBlk_k);

  int blk_idx_c = tmp_blkIdx / (numBlk_i * numBlk_j * numBlk_k);
  tmp_blkIdx = tmp_blkIdx % (numBlk_i * numBlk_j * numBlk_k);

  int blk_idx_i = tmp_blkIdx / (numBlk_j * numBlk_k);
  tmp_blkIdx = tmp_blkIdx % (numBlk_j * numBlk_k);

  int blk_idx_j = tmp_blkIdx / numBlk_k;
  tmp_blkIdx = tmp_blkIdx % (numBlk_k);

  int blk_idx_k = tmp_blkIdx;

  int t3_base_thread = blk_idx_k * SIZE_SLICE_1_K + idx_k +
                       (blk_idx_j * SIZE_SLICE_1_J +
                        (blk_idx_i * SIZE_SLICE_1_I + idx_i +
                         (blk_idx_c * SIZE_SLICE_1_C +
                          (blk_idx_b * SIZE_SLICE_1_B + idx_b +
                           (blk_idx_a * SIZE_SLICE_1_A + idx_a) * size_b) *
                              size_c) *
                             size_i) *
                            size_j) *
                           size_k;

  float temp_av;
  float temp_bv[8];
  float reg_tile[8][4];

  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 4; j++)
      reg_tile[i][j] = 0.0;

// tensor contraction: [[16, 'STR_SD2_T2_H7', 'y', 't2', ['q', 'a', 'c', 'i']],
// [16, 'STR_SD2_V2_H7', 'x', 'v2', ['k', 'j', 'b', 'q']], '+=']
#pragma unroll 1
  for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1) {
    // Part: Generalized Contraction Index (p7b)
    internal_offset = (l + SIZE_INT_UNIT_1) - size_internal;
    if (internal_offset > 0)
      internal_upperbound = internal_offset;

    //---------------------------------------------------------------------------------------------------
    // This is for the new version
    // This Part is for Loading Input-Left
    // tc_gen_code_Kernel_Load_Inputs_Abstracts()
    if (threadIdx.x < SIZE_INT_UNIT_1 - internal_upperbound)
      for (int ll = 0; ll < 8; ll++) {
        // ['q', 'a', 'c', 'i']
        // Exception: Temp. version!: threadIdx.x + l
        // Exception: Temp. version!: idx_a < rng_a
        sm_a[threadIdx.x][threadIdx.y + ll * 8] =
            dev_t2[(blk_idx_a * SIZE_SLICE_1_A + idx_a +
                    (blk_idx_c * SIZE_SLICE_1_C + ll +
                     (blk_idx_i * SIZE_SLICE_1_I + 0) * size_c) *
                        size_a) *
                       size_q +
                   (threadIdx.x + l)];
      }

    // This Part is for Loading Input-Right
    // tc_gen_code_Kernel_Load_Inputs_Abstracts()
    if (threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
      for (int ll = 0; ll < 4; ll++) {
        // ['k', 'j', 'b', 'q']
        // Exception: Temp. version!: threadIdx.y + l + 0
        // Exception: Temp. version!: idx_k < rng_k
        sm_b[threadIdx.y + 0][threadIdx.x + ll * 16] =
            dev_v2[blk_idx_k * SIZE_SLICE_1_K + idx_k +
                   (blk_idx_j * SIZE_SLICE_1_J + ll +
                    (blk_idx_b * SIZE_SLICE_1_B + 0) * size_j) *
                       size_k +
                   (threadIdx.y + l + 0) * stride_int_v2];
        // Exception: Temp. version!: threadIdx.y + l + 8
        // Exception: Temp. version!: idx_k < rng_k
        if (threadIdx.y + l + 8 < size_internal)
          sm_b[threadIdx.y + 8][threadIdx.x + ll * 16] =
              dev_v2[blk_idx_k * SIZE_SLICE_1_K + idx_k +
                     (blk_idx_j * SIZE_SLICE_1_J + ll +
                      (blk_idx_b * SIZE_SLICE_1_B + 0) * size_j) *
                         size_k +
                     (threadIdx.y + l + 8) * stride_int_v2];
      }
    __syncthreads();
    //---------------------------------------------------------------------------------------------------

    // Part: Generalized Threads
    for (int ll = 0; ll < SIZE_INT_UNIT_1 - internal_upperbound; ll++) {
      temp_bv[0] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 0];
      temp_bv[1] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 8];
      temp_bv[2] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 16];
      temp_bv[3] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 24];
      temp_bv[4] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 32];
      temp_bv[5] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 40];
      temp_bv[6] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 48];
      temp_bv[7] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 56];

      for (int xx = 0; xx < 4; xx++) // (1)
      {
        temp_av = sm_b[ll][idx_k + (idx_b)*SIZE_SLICE_1_K + (xx * 16)];

        reg_tile[0][xx] += temp_av * temp_bv[0];
        reg_tile[1][xx] += temp_av * temp_bv[1];
        reg_tile[2][xx] += temp_av * temp_bv[2];
        reg_tile[3][xx] += temp_av * temp_bv[3];
        reg_tile[4][xx] += temp_av * temp_bv[4];
        reg_tile[5][xx] += temp_av * temp_bv[5];
        reg_tile[6][xx] += temp_av * temp_bv[6];
        reg_tile[7][xx] += temp_av * temp_bv[7];
      }
    }
    __syncthreads();
  }

// Store Results (Registers) to Global Memory
// Part: Generalized Threads
// Part: Generalized Register-Tiling
#pragma unroll 8
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 4; j++) {
      dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] =
          reg_tile[i][j];
    }
  }
}

// created by tc_gen_code_Kernel()
__global__ void icaq_qbjk_abcijk_32x32x32x32x32x32x32_3(float *dev_t3, float *dev_t2, float *dev_v2,
                            int size_k, int size_j, int size_i, int size_c,
                            int size_b, int size_a, int size_q, int numBlk_k,
                            int numBlk_j, int numBlk_i, int numBlk_c,
                            int numBlk_b, int numBlk_a, int stride_int_t2,
                            int stride_int_v2, int stride_reg_x,
                            int stride_reg_y, int size_internal) {
  // For Shared Memory,
  __shared__ float sm_a[16][64];
  __shared__ float sm_b[16][64];

  // when opt_pre_computed == -1, all indices will be calculated manually
  // # of indices mapped on TB_X: 2
  // # of indices mapped on TB_Y: 2
  int idx_k = threadIdx.x % SIZE_SLICE_1_K;
  int idx_b = threadIdx.x / SIZE_SLICE_1_K;
  int idx_a = threadIdx.y % SIZE_SLICE_1_A;
  int idx_i = threadIdx.y / SIZE_SLICE_1_A;

  int tmp_blkIdx;
  int blk_idx_a =
      blockIdx.x / (numBlk_b * numBlk_c * numBlk_i * numBlk_j * numBlk_k);
  tmp_blkIdx =
      blockIdx.x % (numBlk_b * numBlk_c * numBlk_i * numBlk_j * numBlk_k);

  int blk_idx_b = tmp_blkIdx / (numBlk_c * numBlk_i * numBlk_j * numBlk_k);
  tmp_blkIdx = tmp_blkIdx % (numBlk_c * numBlk_i * numBlk_j * numBlk_k);

  int blk_idx_c = tmp_blkIdx / (numBlk_i * numBlk_j * numBlk_k);
  tmp_blkIdx = tmp_blkIdx % (numBlk_i * numBlk_j * numBlk_k);

  int blk_idx_i = tmp_blkIdx / (numBlk_j * numBlk_k);
  tmp_blkIdx = tmp_blkIdx % (numBlk_j * numBlk_k);

  int blk_idx_j = tmp_blkIdx / numBlk_k;
  tmp_blkIdx = tmp_blkIdx % (numBlk_k);

  int blk_idx_k = tmp_blkIdx;

  int t3_base_thread = blk_idx_k * SIZE_SLICE_1_K + idx_k +
                       (blk_idx_j * SIZE_SLICE_1_J +
                        (blk_idx_i * SIZE_SLICE_1_I + idx_i +
                         (blk_idx_c * SIZE_SLICE_1_C +
                          (blk_idx_b * SIZE_SLICE_1_B + idx_b +
                           (blk_idx_a * SIZE_SLICE_1_A + idx_a) * size_b) *
                              size_c) *
                             size_i) *
                            size_j) *
                           size_k;

  // need to support partial tiles
  int rng_k, rng_j, rng_i, rng_c, rng_b, rng_a;
  if ((size_k - (blk_idx_k * SIZE_SLICE_1_K)) >= SIZE_SLICE_1_K) {
    rng_k = SIZE_SLICE_1_K;
  } else {
    rng_k = size_k % SIZE_SLICE_1_K;
  }
  if ((size_j - (blk_idx_j * SIZE_SLICE_1_J)) >= SIZE_SLICE_1_J) {
    rng_j = SIZE_SLICE_1_J;
  } else {
    rng_j = size_j % SIZE_SLICE_1_J;
  }
  if ((size_i - (blk_idx_i * SIZE_SLICE_1_I)) >= SIZE_SLICE_1_I) {
    rng_i = SIZE_SLICE_1_I;
  } else {
    rng_i = size_i % SIZE_SLICE_1_I;
  }
  if ((size_c - (blk_idx_c * SIZE_SLICE_1_C)) >= SIZE_SLICE_1_C) {
    rng_c = SIZE_SLICE_1_C;
  } else {
    rng_c = size_c % SIZE_SLICE_1_C;
  }
  if ((size_b - (blk_idx_b * SIZE_SLICE_1_B)) >= SIZE_SLICE_1_B) {
    rng_b = SIZE_SLICE_1_B;
  } else {
    rng_b = size_b % SIZE_SLICE_1_B;
  }
  if ((size_a - (blk_idx_a * SIZE_SLICE_1_A)) >= SIZE_SLICE_1_A) {
    rng_a = SIZE_SLICE_1_A;
  } else {
    rng_a = size_a % SIZE_SLICE_1_A;
  }

  float temp_av;
  float temp_bv[8];
  float reg_tile[8][4];

  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 4; j++)
      reg_tile[i][j] = 0.0;

// tensor contraction: [[16, 'STR_SD2_T2_H7', 'y', 't2', ['q', 'a', 'c', 'i']],
// [16, 'STR_SD2_V2_H7', 'x', 'v2', ['k', 'j', 'b', 'q']], '+=']
#pragma unroll 1
  for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1) {
    //---------------------------------------------------------------------------------------------------
    // This is for the new version
    // This Part is for Loading Input-Left
    // tc_gen_code_Kernel_Load_Inputs_Abstracts()
    if (idx_a < rng_a && 0 < rng_i)
      for (int ll = 0; ll < rng_c; ll++) {
        // ['q', 'a', 'c', 'i']
        // Exception: Temp. version!: threadIdx.x + l
        // Exception: Temp. version!: idx_a < rng_a
        sm_a[threadIdx.x][threadIdx.y + ll * 8] =
            dev_t2[(blk_idx_a * SIZE_SLICE_1_A + idx_a +
                    (blk_idx_c * SIZE_SLICE_1_C + ll +
                     (blk_idx_i * SIZE_SLICE_1_I + 0) * size_c) *
                        size_a) *
                       size_q +
                   (threadIdx.x + l)];
      }

    // This Part is for Loading Input-Right
    // tc_gen_code_Kernel_Load_Inputs_Abstracts()
    if (idx_k < rng_k && 0 < rng_b)
      for (int ll = 0; ll < rng_j; ll++) {
        // ['k', 'j', 'b', 'q']
        // Exception: Temp. version!: threadIdx.y + l + 0
        // Exception: Temp. version!: idx_k < rng_k
        sm_b[threadIdx.y + 0][threadIdx.x + ll * 16] =
            dev_v2[blk_idx_k * SIZE_SLICE_1_K + idx_k +
                   (blk_idx_j * SIZE_SLICE_1_J + ll +
                    (blk_idx_b * SIZE_SLICE_1_B + 0) * size_j) *
                       size_k +
                   (threadIdx.y + l + 0) * stride_int_v2];
        // Exception: Temp. version!: threadIdx.y + l + 8
        // Exception: Temp. version!: idx_k < rng_k
        if (idx_k < rng_k)
          sm_b[threadIdx.y + 8][threadIdx.x + ll * 16] =
              dev_v2[blk_idx_k * SIZE_SLICE_1_K + idx_k +
                     (blk_idx_j * SIZE_SLICE_1_J + ll +
                      (blk_idx_b * SIZE_SLICE_1_B + 0) * size_j) *
                         size_k +
                     (threadIdx.y + l + 8) * stride_int_v2];
      }
    __syncthreads();
    //---------------------------------------------------------------------------------------------------

    // Part: Generalized Threads
    for (int ll = 0; ll < SIZE_INT_UNIT_1; ll++) {
      temp_bv[0] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 0];
      temp_bv[1] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 8];
      temp_bv[2] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 16];
      temp_bv[3] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 24];
      temp_bv[4] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 32];
      temp_bv[5] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 40];
      temp_bv[6] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 48];
      temp_bv[7] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 56];

      for (int xx = 0; xx < 4; xx++) // (1)
      {
        temp_av = sm_b[ll][idx_k + (idx_b)*SIZE_SLICE_1_K + (xx * 16)];

        reg_tile[0][xx] += temp_av * temp_bv[0];
        reg_tile[1][xx] += temp_av * temp_bv[1];
        reg_tile[2][xx] += temp_av * temp_bv[2];
        reg_tile[3][xx] += temp_av * temp_bv[3];
        reg_tile[4][xx] += temp_av * temp_bv[4];
        reg_tile[5][xx] += temp_av * temp_bv[5];
        reg_tile[6][xx] += temp_av * temp_bv[6];
        reg_tile[7][xx] += temp_av * temp_bv[7];
      }
    }
    __syncthreads();
  }

  // Store Results (Registers) to Global Memory
  // Part: Generalized Threads
  // Part: Generalized Register-Tiling
  if (idx_k < rng_k && idx_b < rng_b && idx_a < rng_a && idx_i < rng_i)
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 4; j++) {
        if (i < rng_c && j < rng_j) {
          dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] =
              reg_tile[i][j];
        }
      }
    }
}

// created by tc_gen_code_Kernel()
__global__ void icaq_qbjk_abcijk_32x32x32x32x32x32x32_4(float *dev_t3, float *dev_t2, float *dev_v2,
                            int size_k, int size_j, int size_i, int size_c,
                            int size_b, int size_a, int size_q, int numBlk_k,
                            int numBlk_j, int numBlk_i, int numBlk_c,
                            int numBlk_b, int numBlk_a, int stride_int_t2,
                            int stride_int_v2, int stride_reg_x,
                            int stride_reg_y, int size_internal) {
  // For Shared Memory,
  __shared__ float sm_a[16][64];
  __shared__ float sm_b[16][64];

  int internal_upperbound = 0;
  int internal_offset;

  // when opt_pre_computed == -1, all indices will be calculated manually
  // # of indices mapped on TB_X: 2
  // # of indices mapped on TB_Y: 2
  int idx_k = threadIdx.x % SIZE_SLICE_1_K;
  int idx_b = threadIdx.x / SIZE_SLICE_1_K;
  int idx_a = threadIdx.y % SIZE_SLICE_1_A;
  int idx_i = threadIdx.y / SIZE_SLICE_1_A;

  int tmp_blkIdx;
  int blk_idx_a =
      blockIdx.x / (numBlk_b * numBlk_c * numBlk_i * numBlk_j * numBlk_k);
  tmp_blkIdx =
      blockIdx.x % (numBlk_b * numBlk_c * numBlk_i * numBlk_j * numBlk_k);

  int blk_idx_b = tmp_blkIdx / (numBlk_c * numBlk_i * numBlk_j * numBlk_k);
  tmp_blkIdx = tmp_blkIdx % (numBlk_c * numBlk_i * numBlk_j * numBlk_k);

  int blk_idx_c = tmp_blkIdx / (numBlk_i * numBlk_j * numBlk_k);
  tmp_blkIdx = tmp_blkIdx % (numBlk_i * numBlk_j * numBlk_k);

  int blk_idx_i = tmp_blkIdx / (numBlk_j * numBlk_k);
  tmp_blkIdx = tmp_blkIdx % (numBlk_j * numBlk_k);

  int blk_idx_j = tmp_blkIdx / numBlk_k;
  tmp_blkIdx = tmp_blkIdx % (numBlk_k);

  int blk_idx_k = tmp_blkIdx;

  int t3_base_thread = blk_idx_k * SIZE_SLICE_1_K + idx_k +
                       (blk_idx_j * SIZE_SLICE_1_J +
                        (blk_idx_i * SIZE_SLICE_1_I + idx_i +
                         (blk_idx_c * SIZE_SLICE_1_C +
                          (blk_idx_b * SIZE_SLICE_1_B + idx_b +
                           (blk_idx_a * SIZE_SLICE_1_A + idx_a) * size_b) *
                              size_c) *
                             size_i) *
                            size_j) *
                           size_k;

  // need to support partial tiles
  int rng_k, rng_j, rng_i, rng_c, rng_b, rng_a;
  if ((size_k - (blk_idx_k * SIZE_SLICE_1_K)) >= SIZE_SLICE_1_K) {
    rng_k = SIZE_SLICE_1_K;
  } else {
    rng_k = size_k % SIZE_SLICE_1_K;
  }
  if ((size_j - (blk_idx_j * SIZE_SLICE_1_J)) >= SIZE_SLICE_1_J) {
    rng_j = SIZE_SLICE_1_J;
  } else {
    rng_j = size_j % SIZE_SLICE_1_J;
  }
  if ((size_i - (blk_idx_i * SIZE_SLICE_1_I)) >= SIZE_SLICE_1_I) {
    rng_i = SIZE_SLICE_1_I;
  } else {
    rng_i = size_i % SIZE_SLICE_1_I;
  }
  if ((size_c - (blk_idx_c * SIZE_SLICE_1_C)) >= SIZE_SLICE_1_C) {
    rng_c = SIZE_SLICE_1_C;
  } else {
    rng_c = size_c % SIZE_SLICE_1_C;
  }
  if ((size_b - (blk_idx_b * SIZE_SLICE_1_B)) >= SIZE_SLICE_1_B) {
    rng_b = SIZE_SLICE_1_B;
  } else {
    rng_b = size_b % SIZE_SLICE_1_B;
  }
  if ((size_a - (blk_idx_a * SIZE_SLICE_1_A)) >= SIZE_SLICE_1_A) {
    rng_a = SIZE_SLICE_1_A;
  } else {
    rng_a = size_a % SIZE_SLICE_1_A;
  }

  float temp_av;
  float temp_bv[8];
  float reg_tile[8][4];

  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 4; j++)
      reg_tile[i][j] = 0.0;

// tensor contraction: [[16, 'STR_SD2_T2_H7', 'y', 't2', ['q', 'a', 'c', 'i']],
// [16, 'STR_SD2_V2_H7', 'x', 'v2', ['k', 'j', 'b', 'q']], '+=']
#pragma unroll 1
  for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1) {
    // Part: Generalized Contraction Index (p7b)
    internal_offset = (l + SIZE_INT_UNIT_1) - size_internal;
    if (internal_offset > 0)
      internal_upperbound = internal_offset;

    //---------------------------------------------------------------------------------------------------
    // This is for the new version
    // This Part is for Loading Input-Left
    // tc_gen_code_Kernel_Load_Inputs_Abstracts()
    if (idx_a < rng_a && 0 < rng_i &&
        threadIdx.x < SIZE_INT_UNIT_1 - internal_upperbound)
      for (int ll = 0; ll < rng_c; ll++) {
        // ['q', 'a', 'c', 'i']
        // Exception: Temp. version!: threadIdx.x + l
        // Exception: Temp. version!: idx_a < rng_a
        sm_a[threadIdx.x][threadIdx.y + ll * 8] =
            dev_t2[(blk_idx_a * SIZE_SLICE_1_A + idx_a +
                    (blk_idx_c * SIZE_SLICE_1_C + ll +
                     (blk_idx_i * SIZE_SLICE_1_I + 0) * size_c) *
                        size_a) *
                       size_q +
                   (threadIdx.x + l)];
      }

    // This Part is for Loading Input-Right
    // tc_gen_code_Kernel_Load_Inputs_Abstracts()
    if (idx_k < rng_k && 0 < rng_b &&
        threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
      for (int ll = 0; ll < rng_j; ll++) {
        // ['k', 'j', 'b', 'q']
        // Exception: Temp. version!: threadIdx.y + l + 0
        // Exception: Temp. version!: idx_k < rng_k
        sm_b[threadIdx.y + 0][threadIdx.x + ll * 16] =
            dev_v2[blk_idx_k * SIZE_SLICE_1_K + idx_k +
                   (blk_idx_j * SIZE_SLICE_1_J + ll +
                    (blk_idx_b * SIZE_SLICE_1_B + 0) * size_j) *
                       size_k +
                   (threadIdx.y + l + 0) * stride_int_v2];
        // Exception: Temp. version!: threadIdx.y + l + 8
        // Exception: Temp. version!: idx_k < rng_k
        if (threadIdx.y + l + 8 < size_internal && idx_k < rng_k)
          sm_b[threadIdx.y + 8][threadIdx.x + ll * 16] =
              dev_v2[blk_idx_k * SIZE_SLICE_1_K + idx_k +
                     (blk_idx_j * SIZE_SLICE_1_J + ll +
                      (blk_idx_b * SIZE_SLICE_1_B + 0) * size_j) *
                         size_k +
                     (threadIdx.y + l + 8) * stride_int_v2];
      }
    __syncthreads();
    //---------------------------------------------------------------------------------------------------

    // Part: Generalized Threads
    for (int ll = 0; ll < SIZE_INT_UNIT_1 - internal_upperbound; ll++) {
      temp_bv[0] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 0];
      temp_bv[1] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 8];
      temp_bv[2] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 16];
      temp_bv[3] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 24];
      temp_bv[4] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 32];
      temp_bv[5] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 40];
      temp_bv[6] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 48];
      temp_bv[7] = sm_a[ll][idx_a + (idx_i)*SIZE_SLICE_1_A + 56];

      for (int xx = 0; xx < 4; xx++) // (1)
      {
        temp_av = sm_b[ll][idx_k + (idx_b)*SIZE_SLICE_1_K + (xx * 16)];

        reg_tile[0][xx] += temp_av * temp_bv[0];
        reg_tile[1][xx] += temp_av * temp_bv[1];
        reg_tile[2][xx] += temp_av * temp_bv[2];
        reg_tile[3][xx] += temp_av * temp_bv[3];
        reg_tile[4][xx] += temp_av * temp_bv[4];
        reg_tile[5][xx] += temp_av * temp_bv[5];
        reg_tile[6][xx] += temp_av * temp_bv[6];
        reg_tile[7][xx] += temp_av * temp_bv[7];
      }
    }
    __syncthreads();
  }

  // Store Results (Registers) to Global Memory
  // Part: Generalized Threads
  // Part: Generalized Register-Tiling
  if (idx_k < rng_k && idx_b < rng_b && idx_a < rng_a && idx_i < rng_i)
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 4; j++) {
        if (i < rng_c && j < rng_j) {
          dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] =
              reg_tile[i][j];
        }
      }
    }
}

// written by tc_interface.tc_gen_code_interface_Header()
void icaq_qbjk_abcijk_32x32x32x32x32x32x32(int size_a, int size_b, int size_c, int size_i,
                               int size_j, int size_k, int size_q,
                               float *dev_t2, float *dev_v2, float *dev_t3) {
  int num_thread_blocks_kernel_1;

  num_thread_blocks_kernel_1 =
      CEIL_DIV(size_k, SIZE_SLICE_1_K) * CEIL_DIV(size_j, SIZE_SLICE_1_J) *
      CEIL_DIV(size_i, SIZE_SLICE_1_I) * CEIL_DIV(size_c, SIZE_SLICE_1_C) *
      CEIL_DIV(size_b, SIZE_SLICE_1_B) * CEIL_DIV(size_a, SIZE_SLICE_1_A);
  dim3 gridsize_1(num_thread_blocks_kernel_1);
  dim3 blocksize_1(SIZE_TB_1_X, SIZE_TB_1_Y);

  int stride_output_k = 1;
  int stride_output_j = stride_output_k * size_k;
  int stride_output_i = stride_output_j * size_j;
  int stride_output_c = stride_output_i * size_i;
  int stride_output_b = stride_output_c * size_c;
  int stride_output_a = stride_output_b * size_b;

  int stride_reg_x_1 = stride_output_j;
  int stride_reg_y_1 = stride_output_c;

  int size_internal = size_q;

  int stride_int_t2 = 1;
  int stride_int_v2 = size_k * size_j * size_b;

  // Decision Tree for Kernel Types
  // No Chance to Utilize the Register Transpose
  if (size_k % SIZE_SLICE_1_K == 0 && size_j % SIZE_SLICE_1_J == 0 &&
      size_i % SIZE_SLICE_1_I == 0 && size_c % SIZE_SLICE_1_C == 0 &&
      size_b % SIZE_SLICE_1_B == 0 && size_a % SIZE_SLICE_1_A == 0) {
    // [2] Extenral Index: Full
    if (size_q % SIZE_SLICE_1_Q == 0) {
      // [3] Internal Index: Full
      // >>> External: Full && Internal: Full
      icaq_qbjk_abcijk_32x32x32x32x32x32x32_1<<<gridsize_1, blocksize_1>>>(
          dev_t3, dev_t2, dev_v2, size_k, size_j, size_i, size_c, size_b,
          size_a, size_q, CEIL_DIV(size_k, SIZE_SLICE_1_K),
          CEIL_DIV(size_j, SIZE_SLICE_1_J), CEIL_DIV(size_i, SIZE_SLICE_1_I),
          CEIL_DIV(size_c, SIZE_SLICE_1_C), CEIL_DIV(size_b, SIZE_SLICE_1_B),
          CEIL_DIV(size_a, SIZE_SLICE_1_A), stride_int_t2, stride_int_v2,
          stride_reg_x_1, stride_reg_y_1, size_internal);
    } else {
      // [4] Internal Index: Partial
      // >>> External: Full && Internal: Partial
      icaq_qbjk_abcijk_32x32x32x32x32x32x32_2<<<gridsize_1, blocksize_1>>>(
          dev_t3, dev_t2, dev_v2, size_k, size_j, size_i, size_c, size_b,
          size_a, size_q, CEIL_DIV(size_k, SIZE_SLICE_1_K),
          CEIL_DIV(size_j, SIZE_SLICE_1_J), CEIL_DIV(size_i, SIZE_SLICE_1_I),
          CEIL_DIV(size_c, SIZE_SLICE_1_C), CEIL_DIV(size_b, SIZE_SLICE_1_B),
          CEIL_DIV(size_a, SIZE_SLICE_1_A), stride_int_t2, stride_int_v2,
          stride_reg_x_1, stride_reg_y_1, size_internal);
    }
  } else {
    // [2] Extenral Index: Partial
    if (size_q % SIZE_SLICE_1_Q == 0) {
      // [3] Internal Index: Full
      // >>> External: Partial && Internal: Full
      icaq_qbjk_abcijk_32x32x32x32x32x32x32_3<<<gridsize_1, blocksize_1>>>(
          dev_t3, dev_t2, dev_v2, size_k, size_j, size_i, size_c, size_b,
          size_a, size_q, CEIL_DIV(size_k, SIZE_SLICE_1_K),
          CEIL_DIV(size_j, SIZE_SLICE_1_J), CEIL_DIV(size_i, SIZE_SLICE_1_I),
          CEIL_DIV(size_c, SIZE_SLICE_1_C), CEIL_DIV(size_b, SIZE_SLICE_1_B),
          CEIL_DIV(size_a, SIZE_SLICE_1_A), stride_int_t2, stride_int_v2,
          stride_reg_x_1, stride_reg_y_1, size_internal);
    } else {
      // [4] Internal Index: Partial
      // >>> External: Partial && Internal: Partial
      icaq_qbjk_abcijk_32x32x32x32x32x32x32_4<<<gridsize_1, blocksize_1>>>(
          dev_t3, dev_t2, dev_v2, size_k, size_j, size_i, size_c, size_b,
          size_a, size_q, CEIL_DIV(size_k, SIZE_SLICE_1_K),
          CEIL_DIV(size_j, SIZE_SLICE_1_J), CEIL_DIV(size_i, SIZE_SLICE_1_I),
          CEIL_DIV(size_c, SIZE_SLICE_1_C), CEIL_DIV(size_b, SIZE_SLICE_1_B),
          CEIL_DIV(size_a, SIZE_SLICE_1_A), stride_int_t2, stride_int_v2,
          stride_reg_x_1, stride_reg_y_1, size_internal);
    }
  }
}



#undef SIZE_SLICE_1_Q
#undef SIZE_SLICE_1_K
#undef SIZE_SLICE_1_J
#undef SIZE_SLICE_1_B
#undef SIZE_SLICE_1_A
#undef SIZE_SLICE_1_C
#undef SIZE_SLICE_1_I

#undef SIZE_INT_UNIT_1

#undef SIZE_TB_1_X
#undef SIZE_TB_1_Y
#undef SIZE_REG_1_X
#undef SIZE_REG_1_Y

#undef NUM_INDEX
