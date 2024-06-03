def dotprod_f32 = \xs ys -> map2 (*) xs ys |> f32.sum

entry aq_bq_ab [a][b][q] (xss: [a][q]f32) (yss: [b][q]f32) : [a][b]f32 =
  map (\xs ->
    map (\ys ->
      dotprod_f32 xs ys
    ) yss
  ) xss
entry aq_bq_ab_NO_EPILOGUE [a][b][q] (xss: [a][q]f32) (yss: [b][q]f32) : [a][b]f32 =
  #[no_epilogue]
  map (\xs ->
    map (\ys ->
      dotprod_f32 xs ys
    ) yss
  ) xss

entry qa_bq_ab [a][b][q] (xss: [q][a]f32) (yss: [b][q]f32) : [a][b]f32 =
  map (\xs ->
    map (\ys ->
      dotprod_f32 xs ys
    ) yss
  ) (transpose xss)
entry qa_bq_ab_NO_EPILOGUE [a][b][q] (xss: [q][a]f32) (yss: [b][q]f32) : [a][b]f32 =
  #[no_epilogue]
  map (\xs ->
    map (\ys ->
      dotprod_f32 xs ys
    ) yss
  ) (transpose xss)

entry aq_qb_ab [a][b][q] (xss: [a][q]f32) (yss: [q][b]f32) : [a][b]f32 =
    map (\xs ->
      map (\ys ->
        dotprod_f32 xs ys
      ) (transpose yss)
    ) xss
entry aq_qb_ab_NO_EPILOGUE [a][b][q] (xss: [a][q]f32) (yss: [q][b]f32) : [a][b]f32 =
  #[no_epilogue]
  map (\xs ->
    map (\ys ->
      dotprod_f32 xs ys
    ) (transpose yss)
  ) xss

entry qa_qb_ab [a][b][q] (xss: [q][a]f32) (yss: [q][b]f32) : [a][b]f32 =
    map (\xs ->
      map (\ys ->
        dotprod_f32 xs ys
      ) (transpose yss)
    ) (transpose xss)
entry qa_qb_ab_NO_EPILOGUE [a][b][q] (xss: [q][a]f32) (yss: [q][b]f32) : [a][b]f32 =
  #[no_epilogue]
  map (\xs ->
    map (\ys ->
      dotprod_f32 xs ys
    ) (transpose yss)
  ) (transpose xss)

-- below test cases cover:
--   no partial tiles
--   partial tile in reduction dim
--   partial tiles in outer dims
--   partial tiles in all dims
--   unit outer dim
--   unit reduction dim
--   different size outer dims

-- ==
-- entry: aq_bq_ab aq_bq_ab_NO_EPILOGUE
-- compiled random input { [1024][2048]f32 [4096][2048]f32 } auto output
-- compiled random input { [1024][2039]f32 [4096][2039]f32 } auto output
-- compiled random input { [1021][2048]f32 [4093][2048]f32 } auto output
-- compiled random input { [1021][2039]f32 [4093][2039]f32 } auto output
-- compiled random input { [1][2048]f32 [1][2048]f32 } auto output
-- compiled random input { [1024][1]f32 [4096][1]f32 } auto output

-- ==
-- entry: qa_bq_ab qa_bq_ab_NO_EPILOGUE
-- compiled random input { [2048][1024]f32 [4096][2048]f32 } auto output
-- compiled random input { [2039][1024]f32 [4096][2039]f32 } auto output
-- compiled random input { [2048][1021]f32 [4093][2048]f32 } auto output
-- compiled random input { [2039][1021]f32 [4093][2039]f32 } auto output
-- compiled random input { [2048][1]f32 [1][2048]f32 } auto output
-- compiled random input { [1][1024]f32 [4096][1]f32 } auto output

-- ==
-- entry: aq_qb_ab aq_qb_ab_NO_EPILOGUE
-- compiled random input { [1024][2048]f32 [2048][4096]f32 } auto output
-- compiled random input { [1024][2039]f32 [2039][4096]f32 } auto output
-- compiled random input { [1021][2048]f32 [2048][4093]f32 } auto output
-- compiled random input { [1021][2039]f32 [2039][4093]f32 } auto output
-- compiled random input { [1][2048]f32 [2048][1]f32 } auto output
-- compiled random input { [1024][1]f32 [1][4096]f32 } auto output

-- ==
-- entry: qa_qb_ab qa_qb_ab_NO_EPILOGUE
-- compiled random input { [2048][1024]f32 [2048][4096]f32 } auto output
-- compiled random input { [2039][1024]f32 [2039][4096]f32 } auto output
-- compiled random input { [2048][1021]f32 [2048][4093]f32 } auto output
-- compiled random input { [2039][1021]f32 [2039][4093]f32 } auto output
-- compiled random input { [2048][1]f32 [2048][1]f32 } auto output
-- compiled random input { [1][1024]f32 [1][4096]f32 } auto output
