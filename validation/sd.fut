def dotprod_f32 = \xs ys -> map2 (*) xs ys |> f32.sum

entry icaq_qbjk_abcijk
  [a][b][c][i][j][k][q]
  (xssss': [i][c][a][q]f32)
  (yssss': [q][b][j][k]f32)
  let xssss =
    xssss'
    |> transpose
    |> map transpose
    |> transpose
  let yssss =
    yssss'
    |> transpose
    |> map transpose
    |> map (map transpose)
  in
    map (\xsss ->
      map (\ysss ->
        map (\xss ->
          map (\xs ->
            map (\yss ->
              map (\ys ->
                dotprod_f32 xs ys
              ) yss
            ) ysss
          ) xss
        ) xsss
      ) yssss
    ) xssss
entry icaq_qbjk_abcijk_NO_EPILOGUE
  [a][b][c][i][j][k][q]
  (xssss': [i][c][a][q]f32)
  (yssss': [q][b][j][k]f32)
  let xssss =
    xssss'
    |> transpose
    |> map transpose
    |> transpose
  let yssss =
    yssss'
    |> transpose
    |> map transpose
    |> map (map transpose)
  in
    #[no_epilogue]
    map (\xsss ->
      map (\ysss ->
        map (\xss ->
          map (\xs ->
            map (\yss ->
              map (\ys ->
                dotprod_f32 xs ys
              ) yss
            ) ysss
          ) xss
        ) xsss
      ) yssss
    ) xssss

entry kiaq_bcjq_abcijk
  [a][b][c][i][j][k][q]
  (xssss': [k][i][a][q]f32)
  (yssss: [b][c][j][q]f32)
  let xssss =
    xssss'
    |> transpose
    |> map transpose
    |> transpose
  in
    map (\xsss ->
      map (\ysss ->
        map (\yss ->
          map (\xss ->
            map (\ys ->
              map (\xs ->
                dotprod_f32 xs ys
              ) xss
            ) yss
          ) xsss
        ) ysss
      ) yssss
    ) xssss

entry kiaq_bcjq_abcijk_NO_EPILOGUE
  [a][b][c][i][j][k][q]
  (xssss': [k][i][a][q]f32)
  (yssss: [b][c][j][q]f32)
  let xssss =
    xssss'
    |> transpose
    |> map transpose
    |> transpose
  in
    #[no_epilogue]
    map (\xsss ->
      map (\ysss ->
        map (\yss ->
          map (\xss ->
            map (\ys ->
              map (\xs ->
                dotprod_f32 xs ys
              ) xss
            ) yss
          ) xsss
        ) ysss
      ) yssss
    ) xssss


-- below test cases cover:
--   no partial tiles
--   partial tile in reduction dim
--   partial tiles in outer dims
--   partial tiles in all dims
--   unit reduction dim
--   unit outer dims
--   different size outer dims


-- ==
-- entry: icaq_qbjk_abcijk icaq_qbjk_abcijk_NO_EPILOGUE
-- compiled random input { [32][32][32][32]f32 [32][32][32][32]f32 }
-- compiled random input { [32][32][32][31]f32 [31][32][32][32]f32 }
-- compiled random input { [31][31][31][32]f32 [32][31][31][31]f32 }
-- compiled random input { [31][31][31][31]f32 [31][31][31][31]f32 }
-- compiled random input { [32][32][32][1]f32 [1][32][32][32]f32 }
-- compiled random input { [1][1][1][4096]f32 [4096][1][1][1]f32 }
-- compiled random input { [20][24][28][32]f32 [32][36][40][44]f32 }

-- ==
-- entry: kiaq_bcjq_abcijk kiaq_bcjq_abcijk_NO_EPILOGUE 
-- compiled random input { [32][32][32][32]f32 [32][32][32][32]f32 }
-- compiled random input { [32][32][32][31]f32 [32][32][32][31]f32 }
-- compiled random input { [31][31][31][32]f32 [31][31][31][32]f32 }
-- compiled random input { [31][31][31][31]f32 [31][31][31][31]f32 }
-- compiled random input { [32][32][32][1]f32 [32][32][32][1]f32 }
-- compiled random input { [1][1][1][4096]f32 [1][1][1][4096]f32 }
-- compiled random input { [20][24][28][32]f32 [36][40][44][32]f32 }
