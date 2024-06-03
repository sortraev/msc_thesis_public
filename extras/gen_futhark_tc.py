#!/usr/bin/env python3
import sys
import numpy as np

def indent_and_join(ss, ind_level):
    return "\n".join("  " * ind_level + s for s in ss)

def gen_signature(dims_A, dims_B, dims_C, common_dim, ind = 0):

    fun_definition = f"def {dims_A}_{dims_B}_{dims_C}"
    size_type_params = "  " + "".join([f"[{dim}]" for dim in dims_C + common_dim])
    type_params = "  't_x 't_y 't_z"

    type_A = "".join(f"[{dim}]" for dim in dims_A) + "t_x"
    type_B = "".join(f"[{dim}]" for dim in dims_B) + "t_y"
    name_A = f"x{len(dims_A) * 's'}'"
    name_B = f"y{len(dims_B) * 's'}'"

    param_redomap = f"  (redomap: [{common_dim}]t_x -> [{common_dim}]t_y -> t_z)"
    param_A = f"  ({name_A}: {type_A})"
    param_B = f"  ({name_B}: {type_B})"

    ret_type = "".join(f"[{dim}]" for dim in dims_C) + "t_z"
    ret_decl = f"  : {ret_type} ="
    return indent_and_join(
        [fun_definition, type_params, size_type_params,
         param_redomap, param_A, param_B, ret_decl],
        ind
    )

def gen_rearrangements(dims_A, dims_B, dims_C, common_dim, ind = 0):
    def find_swaps(inv_perm):
        # find sequence of adjacent element swaps needed to obtain
        # permutation, given corresponding inverse permutation `inv_perm`
        # (using bubble sort, recording each swap along the way).
        xs = inv_perm.copy()
        swaps = []
        n = len(xs)
        for i in range(n):
            for j in range(n - i - 1):
                if xs[j] > xs[j + 1]:
                    xs[j], xs[j + 1] = xs[j + 1], xs[j]
                    swaps.append(j)
        return swaps

    def rearrangement(swaps):
        def f(n):
            return "transpose"     if n == 0 else \
                   "map transpose" if n == 1 else \
                   f"map ({f(n - 1)})"
        return [f"  |> {f(swap)}" for swap in swaps]

    dims_C_K = dims_C + common_dim

    # inverse permutations using argsort.
    inv_perm_A = np.argsort([dims_A.index(c) for c in dims_C_K if c in dims_A])
    inv_perm_B = np.argsort([dims_B.index(c) for c in dims_C_K if c in dims_B])

    swaps_A = find_swaps(inv_perm_A)
    swaps_B = find_swaps(inv_perm_B)

    A_str = "x" + "s" * len(dims_A)
    B_str = "y" + "s" * len(dims_B)

    rearrangement_A = [f"let {A_str} =", f"  {A_str}'"] + rearrangement(swaps_A)
    rearrangement_B = [f"let {B_str} =", f"  {B_str}'"] + rearrangement(swaps_B)

    return indent_and_join(rearrangement_A + rearrangement_B, ind)

def gen_mapnest(dims_A, dims_B, dims_C, ind = 0):
    # for each dim in dims_C, does this dim come from A or B?
    # this obviously assumes a dim is in B if it is not in A.
    flags = [c in dims_A for c in dims_C]
    n_A, n_B = len(dims_A), len(dims_B)

    def f(flags, n_A, n_B, ind = 0):
        indent = lambda s: ind * " " + s
        if len(flags) == 0:
            return [indent("redomap xs ys")]

        is_A = flags[0]
        arr_str = ("x" if is_A else "y") + "s" * (n_A if is_A else n_B)
        map_body = f(flags[1:],
                     n_A - is_A,
                     n_B - (not is_A),
                     ind + 2)
        return (
            [indent(f"map (\\{arr_str[:-1]} ->"),
            *map_body,
            indent(f") {arr_str}")]
        )

    return indent_and_join(f(flags, n_A, n_B), ind)


def gen_tc(dims_A, dims_B, dims_C):

    dims_A_s = set(dims_A)
    dims_B_s = set(dims_B)
    K = list(dims_A_s.intersection(dims_B_s))

    if len(K) != 1:
        print("gen_tc() error: Expected exactly one contraced mode!")
        return None

    if dims_A_s.union(dims_B_s).difference(K) != set(dims_C):
        print("gen_tc() error: dims_C does not match free modes in operand tensors!")
        return None

    k = K[0]

    fn_declaration = gen_signature(dims_A, dims_B, dims_C, k)
    rearrangements = gen_rearrangements(dims_A, dims_B, dims_C, k, ind = 1)
    mapnest = gen_mapnest(dims_A, dims_B, dims_C, ind = 2)
    return indent_and_join([fn_declaration, rearrangements, "  in", mapnest], 0)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <A indices> <B indices> <free indices>")
    elif (res := gen_tc(sys.argv[1], sys.argv[2], sys.argv[3])) is not None:
        print(res)
