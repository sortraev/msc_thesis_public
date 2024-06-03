template <
  class ElTp,
  int BLOCKDIM_Y,
  int BLOCKDIM_X,
  int T0, int T1, int T2, int T3, // tile dims.
  int s_stride0, // shmem strides.
  int s_stride1,
  int s_stride2,
  int s_stride3
>
__device__  void
copyGlb2Shr(
  ElTp *g_X,
  ElTp *s_X,
  const int ii0,
  const int ii1,
  const int ii2,
  const int ii3,
  const int N0,
  const int N1,
  const int N2,
  const int N3
  ) {

  const int g_stride0 = N1 * N2 * N3;
  const int g_stride1 =      N2 * N3;
  const int g_stride2 =           N3;
  const int g_stride3 =            1;

  constexpr int TBLOCK_SIZE = BLOCKDIM_Y * BLOCKDIM_X;
  constexpr int TILE_SIZE   = T0 * T1 * T2 * T3;
  const int tid = threadIdx.y * BLOCKDIM_X + threadIdx.x;

  constexpr int D0 = T1 * T2 * T3;
  constexpr int D1 = T2 * T3;
  constexpr int D2 = T3;

  #pragma unroll
  for (int _ii = 0; _ii < CEIL_DIV(TILE_SIZE, TBLOCK_SIZE); _ii++) {
    int ii = _ii * TBLOCK_SIZE + tid;
      // unflatten flat 4D thread index
      int i0  = ii  / D0;
      int tmp = ii  % D0;
      int i1  = tmp / D1;
      tmp     = ii  % D1;
      int i2  = tmp / D2;
      int i3  = tmp % D2;

      int s_idx = i0*s_stride0 + i1*s_stride1 + i2*s_stride2 + i3*s_stride3;
      int g_idx = (ii0 + i0)*g_stride0 + (ii1 + i1)*g_stride1 + (ii2 + i2)*g_stride2 + (ii3 + i3)*g_stride3;

      ElTp elem = (ElTp) 0;
      if (ii0 + i0 < N0 &&
          ii1 + i1 < N1 &&
          ii2 + i2 < N2 &&
          ii3 + i3 < N3)
        elem = g_X[g_idx];
      if (TILE_SIZE % TBLOCK_SIZE == 0 || ii < TILE_SIZE)
        s_X[s_idx] = elem;
  }
  return;
}

template <
  class ElTp,
  int BLOCKDIM_Y, int BLOCKDIM_X,
  int T0, int T1,
  int s_stride0, int s_stride1
>
__device__  void
copyGlb2Shr_2D(
  ElTp *g_X,
  ElTp *s_X,
  const int ii0, const int ii1,
  const int N0, const int N1
  ) {
  copyGlb2Shr
    <ElTp,
     BLOCKDIM_Y,
     BLOCKDIM_X,
     T0, T1, 1, 1,
     s_stride0, s_stride1, 0, 0
    >
    (
     g_X, s_X,
     ii0, ii1, 0, 0,
     N0, N1, 1, 1
    );
}

#define pad_term(size, pad_method) \
  (pad_method) == ON  ? 1 : (pad_method) == OFF ? 0 : 1 - ((size) % 2);

template <
  typename ElTp,
  int Ta,
  int Tb,
  int Tc,
  int Ti,
  int Tj,
  int Tk,
  int Q,
  int Ra,
  int Rb,
  int Rc,
  int Ri,
  int Rj,
  int Rk,
  bool use_epilogue = true,
  padMethod pad_method = AUTO
>
__launch_bounds__(Ta * Tb * Tc * Ti * Tj * Tk)
__global__ void icaq_qbjk_abcijk(
  ElTp *g_A,
  ElTp *g_B,
  ElTp *g_C,
  const int Na,
  const int Nb,
  const int Nc,
  const int Ni,
  const int Nj,
  const int Nk,
  const int Nq) {
  static_assert(std::is_arithmetic<ElTp>::value);

  constexpr int TRa = Ta * Ra;
  constexpr int TRb = Tb * Rb;
  constexpr int TRc = Tc * Rc;
  constexpr int TRi = Ti * Ri;
  constexpr int TRj = Tj * Rj;
  constexpr int TRk = Tk * Rk;

  constexpr int BLOCKDIM_Y = Ta * Tb * Tc;
  constexpr int BLOCKDIM_X = Ti * Tj * Tk;

  /* dynamic shared mem */
  extern __shared__ char shmem[];
  constexpr int s_A_size = TRi * TRc * TRa * Q;
  ElTp *s_A = (ElTp*) shmem;
  ElTp *s_B = s_A + s_A_size;

  constexpr int s_A_stride0 = TRc * TRa * Q;
  constexpr int s_A_stride1 =       TRa * Q;
  constexpr int s_A_stride2 =             Q;
  constexpr int s_A_stride3 =             1;

  constexpr int s_B_stride0 = TRb * TRj * TRk;
  constexpr int s_B_stride1 =       TRj * TRk;
  constexpr int s_B_stride2 =             TRk;
  constexpr int s_B_stride3 =               1;

  constexpr int TILE_SIZE_A = TRi * TRc * TRa * Q;
  constexpr int TILE_SIZE_B = Q * TRb * TRj * TRk;

  const int g_A_stride0 = Nc * Na * Nq;
  const int g_A_stride1 =      Na * Nq;
  const int g_A_stride2 =           Nq;
  const int g_A_stride3 =            1;
  const int g_B_stride0 = Nb * Nj * Nk;
  const int g_B_stride1 =      Nj * Nk;
  const int g_B_stride2 =           Nk;
  const int g_B_stride3 =            1;


  /* compute tblock offsets aa, bb, ... */
  int num_tblocks_b = CEIL_DIV(Nb, TRb);
  int num_tblocks_c = CEIL_DIV(Nc, TRc);
  int num_tblocks_j = CEIL_DIV(Nj, TRj);
  int num_tblocks_k = CEIL_DIV(Nk, TRk);

  int tmp = blockIdx.y;
  int aa  = tmp / (num_tblocks_b * num_tblocks_c) * TRa;
  tmp     = tmp % (num_tblocks_b * num_tblocks_c);
  int bb  = tmp / num_tblocks_c * TRb;
  int cc  = (tmp % num_tblocks_c) * TRc;

  tmp    = blockIdx.x;
  int ii = tmp / (num_tblocks_j * num_tblocks_k) * TRi;
  tmp    = tmp % (num_tblocks_j * num_tblocks_k);
  int jj = tmp / num_tblocks_k * TRj;
  int kk = (tmp % num_tblocks_k) * TRk;

  /* unflatten thread indices into tid_a, tid_b, ... */
  tmp       = threadIdx.y;
  int tid_a = tmp / (Tb * Tc) * Ra;
  tmp       = tmp % (Tb * Tc);
  int tid_b = tmp / Tc * Rb;
  int tid_c = (tmp % Tc) * Rc;

  tmp       = threadIdx.x;
  int tid_i = tmp / (Tj * Tk) * Ri;
  tmp       = tmp % (Tj * Tk);
  int tid_j = tmp / Tk * Rj;
  int tid_k = (tmp % Tk) * Rk;

  ElTp r_C[Ra][Rb][Rc][Ri][Rj][Rk];// = { 0 };

  /* zero-init register tile */
  for (int a = 0; a < Ra; a++)
  for (int b = 0; b < Rb; b++)
  for (int c = 0; c < Rc; c++)
  for (int i = 0; i < Ri; i++)
  for (int j = 0; j < Rj; j++)
  for (int k = 0; k < Rk; k++)
    r_C[a][b][c][i][j][k] = (ElTp) 0;

  int loop_bound;
  if constexpr(use_epilogue)
    loop_bound = Nq / Q;
  else
    loop_bound = CEIL_DIV(Nq, Q);

  for (int qq0 = 0; qq0 < loop_bound; qq0++) {
    int qq = qq0 * Q;

    const int tid = threadIdx.y * BLOCKDIM_X + threadIdx.x;

    copyGlb2Shr
      <ElTp, BLOCKDIM_Y, BLOCKDIM_X,
       TRi, TRc, TRa, Q,
       s_A_stride0, s_A_stride1, s_A_stride2, s_A_stride3>
      (g_A, s_A, ii, cc, aa, qq, Ni, Nc, Na, Nq);
    copyGlb2Shr
      <ElTp, BLOCKDIM_Y, BLOCKDIM_X,
       Q, TRb, TRj, TRk,
       s_B_stride0, s_B_stride1, s_B_stride2, s_B_stride3>
      (g_B, s_B, qq, bb, jj, kk, Nq, Nb, Nj, Nk);
    
    __syncthreads();

    /* accumulate contraction */
    for (int q = 0; q < Q; q++)
      for (int a = 0; a < Ra; a++)
      for (int b = 0; b < Rb; b++)
      for (int c = 0; c < Rc; c++)
      for (int i = 0; i < Ri; i++)
      for (int j = 0; j < Rj; j++)
      for (int k = 0; k < Rk; k++)
        r_C[a][b][c][i][j][k] +=
          idx4_(s_A, s_A_stride0, s_A_stride1, s_A_stride2, s_A_stride3,
                     tid_i + i, tid_c + c, tid_a + a, q) *
          idx4_(s_B, s_B_stride0, s_B_stride1, s_B_stride2, s_B_stride3,
                     q, tid_b + b, tid_j + j, tid_k + k);

    __syncthreads();
  }

  if constexpr (use_epilogue)
  if (Nq % Q != 0) {
    int qq = (Nq / Q) * Q;

    copyGlb2Shr
      <ElTp, BLOCKDIM_Y, BLOCKDIM_X,
       TRi, TRc, TRa, Q,
       s_A_stride0, s_A_stride1, s_A_stride2, s_A_stride3>
      (g_A, s_A, ii, cc, aa, qq, Ni, Nc, Na, Nq);
    copyGlb2Shr
      <ElTp, BLOCKDIM_Y, BLOCKDIM_X,
       Q, TRb, TRj, TRk,
       s_B_stride0, s_B_stride1, s_B_stride2, s_B_stride3>
      (g_B, s_B, qq, bb, jj, kk, Nq, Nb, Nj, Nk);

    __syncthreads();

    /* accumulate contraction */
    for (int q = 0; q < Q; q++)
      if (qq + q < Nq)
        for (int a = 0; a < Ra; a++)
        for (int b = 0; b < Rb; b++)
        for (int c = 0; c < Rc; c++)
        for (int i = 0; i < Ri; i++)
        for (int j = 0; j < Rj; j++)
        for (int k = 0; k < Rk; k++)
          r_C[a][b][c][i][j][k] +=
            idx4_(s_A, s_A_stride0, s_A_stride1, s_A_stride2, s_A_stride3,
                       tid_i + i, tid_c + c, tid_a + a, q) *
            idx4_(s_B, s_B_stride0, s_B_stride1, s_B_stride2, s_B_stride3,
                       q, tid_b + b, tid_j + j, tid_k + k);

    // __syncthreads(); // redundant in the epilogue.
  }

  /* thread global write offsets                      */
  /* (note: tid_x is already scaled by Rx for each x) */
  int aaa = aa + tid_a;
  int bbb = bb + tid_b;
  int ccc = cc + tid_c;
  int iii = ii + tid_i;
  int jjj = jj + tid_j;
  int kkk = kk + tid_k;

  /* write register tile back to gmem */
  for (int a = 0; a < Ra; a++)
  for (int b = 0; b < Rb; b++)
  for (int c = 0; c < Rc; c++)
  for (int i = 0; i < Ri; i++)
  for (int j = 0; j < Rj; j++)
  for (int k = 0; k < Rk; k++)
    if (aaa + a < Na &&
        bbb + b < Nb &&
        ccc + c < Nc &&
        iii + i < Ni &&
        jjj + j < Nj &&
        kkk + k < Nk)
      idx6(g_C, Na, Nb, Nc, Ni, Nj, Nk,
           aaa + a, bbb + b, ccc + c, iii + i, jjj + j, kkk + k) =
        r_C[a][b][c][i][j][k];
}



template <
  typename ElTp,
  int Ta,
  int Tb,
  int Tc,
  int Ti,
  int Tj,
  int Tk,
  int Q,
  int Ra,
  int Rb,
  int Rc,
  int Ri,
  int Rj,
  int Rk,
  bool use_epilogue = true,
  padMethod pad_method = AUTO
>
__launch_bounds__(Ta * Tb * Tc * Ti * Tj * Tk)
__global__ void kiaq_bcjq_abcijk(
  ElTp *g_A,
  ElTp *g_B,
  ElTp *g_C,
  const int Na,
  const int Nb,
  const int Nc,
  const int Ni,
  const int Nj,
  const int Nk,
  const int Nq) {
  static_assert(std::is_arithmetic<ElTp>::value);

  constexpr int TRa = Ta * Ra;
  constexpr int TRb = Tb * Rb;
  constexpr int TRc = Tc * Rc;
  constexpr int TRi = Ti * Ri;
  constexpr int TRj = Tj * Rj;
  constexpr int TRk = Tk * Rk;

  constexpr int BLOCKDIM_Y = Ta * Tb * Tc;
  constexpr int BLOCKDIM_X = Ti * Tj * Tk;

  /* dynamic shared mem */
  extern __shared__ char shmem[];

  constexpr int s_A_inner_size =
    TRi * TRa * Q + pad_term(TRi * TRa * Q, pad_method);
  constexpr int s_A_size = TRk * s_A_inner_size;
  ElTp *s_A = (ElTp*) shmem;
  ElTp *s_B = s_A + s_A_size;

  constexpr int s_A_stride0 = TRi * TRa * Q;
  constexpr int s_A_stride1 =       TRa * Q;
  constexpr int s_A_stride2 =             Q;
  constexpr int s_A_stride3 =             1;

  constexpr int s_B_stride0 = TRc * TRj * Q;
  constexpr int s_B_stride1 =       TRj * Q;
  constexpr int s_B_stride2 =             Q;
  constexpr int s_B_stride3 =             1;

  constexpr int TILE_SIZE_A = TRk * TRi * TRa * Q;
  constexpr int TILE_SIZE_B = TRb * TRc * TRj * Q;

  const int g_A_stride0 = Ni * Na * Nq;
  const int g_A_stride1 =      Na * Nq;
  const int g_A_stride2 =           Nq;
  const int g_A_stride3 =            1;
  const int g_B_stride0 = Nc * Nj * Nq;
  const int g_B_stride1 =      Nj * Nq;
  const int g_B_stride2 =           Nq;
  const int g_B_stride3 =            1;


  /* compute tblock offsets aa, bb, ... */
  int num_tblocks_b = CEIL_DIV(Nb, TRb);
  int num_tblocks_c = CEIL_DIV(Nc, TRc);
  int num_tblocks_j = CEIL_DIV(Nj, TRj);
  int num_tblocks_k = CEIL_DIV(Nk, TRk);

  int tmp = blockIdx.y;
  int aa  = tmp / (num_tblocks_b * num_tblocks_c) * TRa;
  tmp     = tmp % (num_tblocks_b * num_tblocks_c);
  int bb  = tmp / num_tblocks_c * TRb;
  int cc  = (tmp % num_tblocks_c) * TRc;

  tmp    = blockIdx.x;
  int ii = tmp / (num_tblocks_j * num_tblocks_k) * TRi;
  tmp    = tmp % (num_tblocks_j * num_tblocks_k);
  int jj = tmp / num_tblocks_k * TRj;
  int kk = (tmp % num_tblocks_k) * TRk;

  /* unflatten thread indices into tid_a, tid_b, ... */
  tmp       = threadIdx.y;
  int tid_a = tmp / (Tb * Tc) * Ra;
  tmp       = tmp % (Tb * Tc);
  int tid_b = tmp / Tc * Rb;
  int tid_c = (tmp % Tc) * Rc;

  tmp       = threadIdx.x;
  int tid_i = tmp / (Tj * Tk) * Ri;
  tmp       = tmp % (Tj * Tk);
  int tid_j = tmp / Tk * Rj;
  int tid_k = (tmp % Tk) * Rk;

  ElTp r_C[Ra][Rb][Rc][Ri][Rj][Rk];

  /* zero-init register tile */
  for (int a = 0; a < Ra; a++)
  for (int b = 0; b < Rb; b++)
  for (int c = 0; c < Rc; c++)
  for (int i = 0; i < Ri; i++)
  for (int j = 0; j < Rj; j++)
  for (int k = 0; k < Rk; k++)
    r_C[a][b][c][i][j][k] = (ElTp) 0;

  int loop_bound;
  if constexpr(use_epilogue)
    loop_bound = Nq / Q;
  else
    loop_bound = CEIL_DIV(Nq, Q);

  for (int qq0 = 0; qq0 < loop_bound; qq0++) {
    int qq = qq0 * Q;

    const int tid = threadIdx.y * BLOCKDIM_X + threadIdx.x;

    copyGlb2Shr
      <ElTp, BLOCKDIM_Y, BLOCKDIM_X,
       TRk, TRi, TRa, Q,
       s_A_stride0, s_A_stride1, s_A_stride2, s_A_stride3>
      (g_A, s_A, kk, ii, aa, qq, Nk, Ni, Na, Nq);
    copyGlb2Shr
      <ElTp, BLOCKDIM_Y, BLOCKDIM_X,
       TRb, TRc, TRj, Q,
       s_B_stride0, s_B_stride1, s_B_stride2, s_B_stride3>
      (g_B, s_B, bb, cc, jj, qq, Nb, Nc, Nj, Nq);
    
    __syncthreads();

    /* accumulate contraction */
    for (int q = 0; q < Q; q++)
      for (int a = 0; a < Ra; a++)
      for (int b = 0; b < Rb; b++)
      for (int c = 0; c < Rc; c++)
      for (int i = 0; i < Ri; i++)
      for (int j = 0; j < Rj; j++)
      for (int k = 0; k < Rk; k++)
        r_C[a][b][c][i][j][k] +=
          idx4_(s_A, s_A_stride0, s_A_stride1, s_A_stride2, s_A_stride3,
                     tid_k + k, tid_i + i, tid_a + a, q) *
          idx4_(s_B, s_B_stride0, s_B_stride1, s_B_stride2, s_B_stride3,
                     tid_b + b, tid_c + c, tid_j + j, q);

    __syncthreads();
  }

  if constexpr (use_epilogue)
  if (Nq % Q != 0) {
    int qq = (Nq / Q) * Q;

    copyGlb2Shr
      <ElTp, BLOCKDIM_Y, BLOCKDIM_X,
       TRk, TRi, TRa, Q,
       s_A_stride0, s_A_stride1, s_A_stride2, s_A_stride3>
      (g_A, s_A, kk, ii, aa, qq, Nk, Ni, Na, Nq);
    copyGlb2Shr
      <ElTp, BLOCKDIM_Y, BLOCKDIM_X,
       TRb, TRc, TRj, Q,
       s_B_stride0, s_B_stride1, s_B_stride2, s_B_stride3>
      (g_B, s_B, bb, cc, jj, qq, Nb, Nc, Nj, Nq);
    
    __syncthreads();

    /* accumulate contraction */
    for (int q = 0; q < Q; q++)
      for (int a = 0; a < Ra; a++)
      for (int b = 0; b < Rb; b++)
      for (int c = 0; c < Rc; c++)
      for (int i = 0; i < Ri; i++)
      for (int j = 0; j < Rj; j++)
      for (int k = 0; k < Rk; k++)
        r_C[a][b][c][i][j][k] +=
          idx4_(s_A, s_A_stride0, s_A_stride1, s_A_stride2, s_A_stride3,
                     tid_k + k, tid_i + i, tid_a + a, q) *
          idx4_(s_B, s_B_stride0, s_B_stride1, s_B_stride2, s_B_stride3,
                     tid_b + b, tid_c + c, tid_j + j, q);

    // __syncthreads(); // redundant in the epilogue.
  }

  /* thread global write offsets                      */
  /* (note: tid_x is already scaled by Rx for each x) */
  int aaa = aa + tid_a;
  int bbb = bb + tid_b;
  int ccc = cc + tid_c;
  int iii = ii + tid_i;
  int jjj = jj + tid_j;
  int kkk = kk + tid_k;

  /* write register tile back to gmem */
  for (int a = 0; a < Ra; a++)
  for (int b = 0; b < Rb; b++)
  for (int c = 0; c < Rc; c++)
  for (int i = 0; i < Ri; i++)
  for (int j = 0; j < Rj; j++)
  for (int k = 0; k < Rk; k++)
    if (aaa + a < Na &&
        bbb + b < Nb &&
        ccc + c < Nc &&
        iii + i < Ni &&
        jjj + j < Nj &&
        kkk + k < Nk)
      idx6(g_C, Na, Nb, Nc, Ni, Nj, Nk,
           aaa + a, bbb + b, ccc + c, iii + i, jjj + j, kkk + k) =
        r_C[a][b][c][i][j][k];
}

template <
  typename ElTp,
  int Ta,
  int Tb,
  int _Tc,
  int _Ti,
  int _Tj,
  int _Tk,
  int Q,
  int Ra,
  int Rb,
  int _Rc,
  int _Ri,
  int _Rj,
  int _Rk,
  bool use_epilogue = true,
  padMethod pad_method = AUTO
>
__launch_bounds__(Ta * Tb)
__global__ void aq_qb_ab(
  ElTp *g_A,
  ElTp *g_B,
  ElTp *g_C,
  const int Na,
  const int Nb,
  const int _Nc,
  const int _Ni,
  const int _Nj,
  const int _Nk,
  const int Nq) {
  static_assert(std::is_arithmetic<ElTp>::value);

  constexpr int TRa = Ta * Ra;
  constexpr int TRb = Tb * Rb;

  constexpr int BLOCKDIM_Y = Ta;
  constexpr int BLOCKDIM_X = Tb;

  /* dynamic shared mem */
  extern __shared__ char shmem[];
  constexpr int s_A_size = TRa * Q;
  ElTp *s_A = (ElTp*) shmem;
  ElTp *s_B = s_A + s_A_size;

  /* compute tblock offsets aa, bb, ... */

  int aa = blockIdx.y * TRa;
  int bb = blockIdx.x * TRb;

  /* unflatten thread indices into tid_a, tid_b, ... */
  int tid_a = threadIdx.y;
  int tid_b = threadIdx.x;

  ElTp r_C[Ra][Rb];
  /* zero-init register tile */
  for (int a = 0; a < Ra; a++)
  for (int b = 0; b < Rb; b++)
    r_C[a][b] = (ElTp) 0;


  int loop_bound;
  if constexpr(use_epilogue)
    loop_bound = Nq / Q;
  else
    loop_bound = CEIL_DIV(Nq, Q);

  for (int qq0 = 0; qq0 < loop_bound; qq0++) {
    int qq = qq0 * Q;

    copyGlb2Shr_2D
      <ElTp, BLOCKDIM_Y, BLOCKDIM_X,
       TRa, Q,
       Q, 1>
      (g_A, s_A, aa, qq, Na, Nq);
    copyGlb2Shr_2D
      <ElTp, BLOCKDIM_Y, BLOCKDIM_X,
       Q, TRb,
       TRb, 1>
      (g_B, s_B, qq, bb, Nq, Nb);
    __syncthreads();


    /* accumulate contraction */
    for (int q = 0; q < Q; q++)
      for (int a = 0; a < Ra; a++)
      for (int b = 0; b < Rb; b++)
        r_C[a][b] +=
          s_A[(tid_a + a) * Q + q] *
          s_B[q * TRb + tid_b + b];

    __syncthreads();
  }

  if constexpr(use_epilogue)
  if (Nq % Q != 0) {
    int qq = (Nq / Q) * Q;

    copyGlb2Shr_2D
      <ElTp, BLOCKDIM_Y, BLOCKDIM_X,
       TRa, Q,
       Q, 1>
      (g_A, s_A, aa, qq, Na, Nq);
    copyGlb2Shr_2D
      <ElTp, BLOCKDIM_Y, BLOCKDIM_X,
       Q, TRb,
       TRb, 1>
      (g_B, s_B, qq, bb, Nq, Nb);
    __syncthreads();


    /* accumulate contraction */
    for (int q = 0; q < Q; q++)
      for (int a = 0; a < Ra; a++)
      for (int b = 0; b < Rb; b++)
        r_C[a][b] +=
          s_A[(tid_a + a) * Q + q] *
          s_B[q * TRb + tid_b + b];

    __syncthreads(); // redundant in the epilogue
  }

  /* compute thread global write offsets.
     note: tid_x is already scaled by Rx for each x */
  int aaa = aa + tid_a;
  int bbb = bb + tid_b;

  /* write register tile back to gmem */
  for (int a = 0; a < Ra; a++)
  for (int b = 0; b < Rb; b++)
    if (aaa + a < Na && bbb + b < Nb)
      g_C[(aaa + a) * Nb + bbb + b] = r_C[a][b];
}
