#!/usr/bin/bash

mm_configs_file=./mm_configs.txt
sd_configs_file=./sd_configs.txt

mm_entry_points="aq_bq_ab aq_bq_ab_NO_EPILOGUE qa_bq_ab qa_bq_ab_NO_EPILOGUE \
                 aq_qb_ab aq_qb_ab_NO_EPILOGUE qa_qb_ab qa_qb_ab_NO_EPILOGUE"
sd_entry_points="icaq_qbjk_abcijk icaq_qbjk_abcijk_NO_EPILOGUE \
                 kiaq_bcjq_abcijk kiaq_bcjq_abcijk_NO_EPILOGUE"

for entry_point in $mm_entry_points; do
  echo Testing entry point "$entry_point"
  while read line; do
    config=($line)
    Ta=${config[0]}
    Tb=${config[1]}
    Q=${config[2]}
    Ra=${config[3]}
    Rb=${config[4]}

    # We use futhark-bench because it offers the --entry-point flag, which
    # futhark-test does not for som reason.
    futhark bench mm.fut                           \
      --backend=cuda                               \
      --entry-point=$entry_point                   \
      --pass-option="--param=$entry_point.T_a=$Ta" \
      --pass-option="--param=$entry_point.T_b=$Tb" \
      --pass-option="--param=$entry_point.T_q=$Q"  \
      --pass-option="--param=$entry_point.R_a=$Ra" \
      --pass-option="--param=$entry_point.R_b=$Rb" \
      --pass-option="--param=$entry_point.R_b=$Rb"

  done < $mm_configs_file
done


for entry_point in $sd_entry_points; do
  echo Testing entry point "$entry_point"
  while read line; do
    config=($line)
    Ta=${config[0]}
    Tb=${config[1]}
    Tc=${config[2]}
    Ti=${config[3]}
    Tj=${config[4]}
    Tk=${config[5]}
    Q=${config[6]}
    Ra=${config[7]}
    Rb=${config[8]}
    Rc=${config[9]}
    Ri=${config[10]}
    Rj=${config[11]}
    Rk=${config[12]}

    # We use futhark-bench because it offers the --entry-point flag, which
    # futhark-test does not for som reason.
    futhark bench mm.fut                           \
      --backend=cuda                               \
      --entry-point=$entry_point                   \
      --pass-option="--param=$entry_point.T_a=$Ta" \
      --pass-option="--param=$entry_point.T_b=$Tb" \
      --pass-option="--param=$entry_point.T_c=$Tc" \
      --pass-option="--param=$entry_point.T_i=$Ti" \
      --pass-option="--param=$entry_point.T_j=$Tj" \
      --pass-option="--param=$entry_point.T_k=$Tk" \
      --pass-option="--param=$entry_point.T_q=$Q"  \
      --pass-option="--param=$entry_point.R_a=$Ra" \
      --pass-option="--param=$entry_point.R_b=$Rb" \
      --pass-option="--param=$entry_point.R_c=$Rc" \
      --pass-option="--param=$entry_point.R_i=$Ri" \
      --pass-option="--param=$entry_point.R_j=$Rj" \
      --pass-option="--param=$entry_point.R_k=$Rk"

  done < $sd_configs_file
done

