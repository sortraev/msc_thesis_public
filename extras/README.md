# Extras

## `gen_futhark_tc.py`

Used to generate Futhark code for tensor contractions given the operand indices
and free indices as argument.

Usage:

```
$ python gen_futhark_tc.py <A indices> <B indices> <free indices>
```

Example: for the contraction `Z_abcijk = X_icaq * Y_qbjk`, use:

```
$ python gen_futhark_tc.py icaq qbjk abcijk
```
