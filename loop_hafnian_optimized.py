"""
Optimized Loop Hafnian Batch Gamma Functions

"""

from __future__ import annotations

import numpy as np
import numba
from numba import prange

from thewalrus import charpoly
from thewalrus._hafnian import (
    precompute_binoms,
    matched_reps,
    find_kept_edges,
    get_submatrices,
    get_submatrix_batch_odd0,
    get_Dsubmatrices,
)
from thewalrus.loop_hafnian_batch import add_batch_edges_odd, add_batch_edges_even


# =============================================================================
# OPTIMIZATION 1: Powertrace-cached versions of f_loop and f_loop_odd
# =============================================================================

@numba.jit(nopython=True, cache=True)
def powertrace(H, n):  # pragma: no cover
    """Calculates the powertraces of the matrix ``H`` up to power ``n-1``.

    Args:
        H (array): square matrix
        n (int): required order

    Returns:
        (array): list of power traces from ``0`` to ``n-1``
    """
    m = len(H)
    min_val = min(n, m)
    pow_traces = [m, np.trace(H)]
    A = H
    for _ in range(min_val - 2):
        A = A @ H
        pow_traces.append(np.trace(A))
    if n <= m:
        return np.array(pow_traces, dtype=H.dtype)
    char_pol = charpoly.charpoly(H)
    for _ in range(min_val, n):
        ssum = 0
        for k in range(m):
            ssum -= char_pol[k] * pow_traces[-k - 1]
        pow_traces.append(ssum)
    return np.array(pow_traces, dtype=H.dtype)


@numba.jit(nopython=True, cache=True) # type: ignore
def f_loop_with_powtrace(AX_S: np.ndarray, XD_S: np.ndarray, D_S: np.ndarray,
                         n: int, powtrace_arr: np.ndarray) -> np.ndarray:
    """
    Evaluate polynomial coefficients using pre-computed powertrace.

    OPTIMIZATION 1: Instead of computing powertrace(AX) internally,
    we accept it as a parameter to avoid redundant computation.

    Args:
        AX_S: AX_S with weights given by repetitions and excluded rows removed
        XD_S: diagonal multiplied by X
        D_S: diagonal
        n: number of polynomial coefficients to compute
        powtrace_arr: pre-computed power traces of AX matrix

    Returns:
        array: polynomial coefficients
    """
    count = 0
    comb = np.zeros((2, n // 2 + 1), dtype=np.complex128)
    comb[0, 0] = 1

    # Use pre-computed powertrace instead of computing it
    XD_S_local = XD_S.copy()

    for i in range(1, n // 2 + 1):
        factor = powtrace_arr[i] / (2 * i) + (XD_S_local @ D_S) / 2
        XD_S_local = XD_S_local @ AX_S
        powfactor = 1.0
        count = 1 - count
        comb[count, :] = comb[1 - count, :]
        for j in range(1, n // (2 * i) + 1):
            powfactor *= factor / j
            for k in range(i * j + 1, n // 2 + 2):
                comb[count, k - 1] += comb[1 - count, k - i * j - 1] * powfactor
    return comb[count, :]


@numba.jit(nopython=True, cache=True) # type: ignore
def f_loop_odd_with_powtrace(AX_S: np.ndarray, XD_S: np.ndarray, D_S: np.ndarray,
                              n: int, oddloop: complex, oddVX_S: np.ndarray,
                              powtrace_arr: np.ndarray) -> np.ndarray:
    """
    Evaluate polynomial coefficients for odd case using pre-computed powertrace.

    OPTIMIZATION 1: Instead of computing powertrace(AX) internally,
    we accept it as a parameter to avoid redundant computation when
    both f_loop and f_loop_odd need the same powertrace.

    Args:
        AX_S: AX_S with weights given by repetitions and excluded rows removed
        XD_S: diagonal multiplied by X
        D_S: diagonal
        n: number of polynomial coefficients to compute
        oddloop: weight of self-edge
        oddVX_S: vector corresponding to matrix at the index of the self-edge
        powtrace_arr: pre-computed power traces of AX matrix

    Returns:
        array: polynomial coefficients
    """
    count = 0
    comb = np.zeros((2, n + 1), dtype=np.complex128)
    comb[0, 0] = 1

    # Use pre-computed powertrace
    D_S_local = D_S.copy()
    XD_S_local = XD_S.copy()

    for i in range(1, n + 1):
        if i == 1:
            factor = oddloop
        elif i % 2 == 0:
            factor = powtrace_arr[i // 2] / i + (XD_S_local @ D_S_local) / 2
        else:
            factor = oddVX_S @ D_S_local
            D_S_local = AX_S @ D_S_local

        powfactor = 1.0
        count = 1 - count
        comb[count, :] = comb[1 - count, :]
        for j in range(1, n // i + 1):
            powfactor *= factor / j
            for k in range(i * j + 1, n + 2):
                comb[count, k - 1] += comb[1 - count, k - i * j - 1] * powfactor

    return comb[count, :]


# =============================================================================
# OPTIMIZATION 2: Batch operations - compute powertrace once per j iteration
# =============================================================================

@numba.jit(nopython=True, cache=True, parallel=True) # type: ignore
def _calc_loop_hafnian_batch_gamma_even_optimized(
    A: np.ndarray,
    D: np.ndarray,
    fixed_edge_reps: np.ndarray,
    batch_max: int,
    odd_cutoff: int,
    glynn: bool = True
) -> np.ndarray:
    """
    Optimized version of _calc_loop_hafnian_batch_gamma_even.

    Key optimizations:
    1. Compute powertrace ONCE per j iteration (outside the k loop)
    2. Use pre-computed powertrace in both f_loop and f_loop_odd calls

    This eliminates redundant O(m³) matrix multiplications that were being
    performed n_D times (once per displacement vector) in the original code.

    Args:
        A: input matrix
        D: vector to find loop hafnian batch
        fixed_edge_reps: fixed number of edge repetition
        batch_max: maximum number of photons for m mode
        odd_cutoff: cutoff for unpaired modes
        glynn: determines the method used to evaluate the loop hafnian batch

    Returns:
        H_batch: matrix that contains batched loop hafnian with threshold detectors
    """
    oddloop = D[:, 0]
    oddV = A[0, :]

    n = A.shape[0]
    N_fixed = 2 * fixed_edge_reps.sum()

    N_max = N_fixed + 2 * batch_max + odd_cutoff

    edge_reps = np.concatenate((np.array([batch_max]), fixed_edge_reps))
    steps = np.prod(edge_reps + 1)

    # Precompute binomial coefficients
    max_binom = edge_reps.max() + odd_cutoff
    binoms = precompute_binoms(max_binom) # type: ignore
    n_D = D.shape[0]

    H_batch = np.zeros((n_D, 2 * batch_max + odd_cutoff + 1), dtype=np.complex128)

    for j in prange(steps):
        Hnew = np.zeros((n_D, 2 * batch_max + odd_cutoff + 1), dtype=np.complex128)

        kept_edges = find_kept_edges(j, edge_reps) # type: ignore
        edges_sum = kept_edges.sum()

        binom_prod = 1.0
        for i in range(1, n // 2):
            binom_prod *= binoms[edge_reps[i], kept_edges[i]]

        delta = 2 * kept_edges - edge_reps if glynn else kept_edges # type: ignore

        AX_S, XD_S, D_S, oddVX_S = get_submatrices(delta, A, D[0, :], oddV) # type: ignore
        AX_S_copy = AX_S.copy()

        # OPTIMIZATION 2: Compute powertrace ONCE per j iteration (outside k loop)
        # This was being computed n_D times in the original code!
        powtrace_len = N_max // 2 + 2
        powtrace_arr = powertrace(AX_S_copy, powtrace_len)

        for k in range(n_D):
            XD_S, D_S = get_Dsubmatrices(delta, D[k, :]) # type: ignore

            # OPTIMIZATION 1: Use pre-computed powertrace for both calls
            f_even = f_loop_with_powtrace(AX_S, XD_S, D_S, N_max, powtrace_arr) # type: ignore
            f_odd = f_loop_odd_with_powtrace(AX_S, XD_S, D_S, N_max, oddloop[k], oddVX_S, powtrace_arr) # type: ignore

            for N_det in range(2 * kept_edges[0], 2 * batch_max + odd_cutoff + 1):
                N = N_fixed + N_det
                plus_minus = (-1.0) ** (N // 2 - edges_sum)

                n_det_binom_prod = binoms[N_det // 2, kept_edges[0]] * binom_prod

                if N_det % 2 == 0:
                    Hnew[k, N_det] += n_det_binom_prod * plus_minus * f_even[N // 2]
                else:
                    Hnew[k, N_det] += n_det_binom_prod * plus_minus * f_odd[N]
        H_batch += Hnew

    if glynn:
        for j in range(H_batch.shape[1]):
            x = N_fixed + j
            H_batch[:, j] *= 0.5 ** (x // 2)

    return H_batch


@numba.jit(nopython=True, cache=True, parallel=True) # type: ignore
def _calc_loop_hafnian_batch_gamma_odd_optimized(
    A: np.ndarray,
    D: np.ndarray,
    fixed_edge_reps: np.ndarray,
    batch_max: int,
    even_cutoff: int,
    glynn: bool = True
) -> np.ndarray:
    """
    Optimized version of _calc_loop_hafnian_batch_gamma_odd.

    Same optimizations as the even version:
    1. Compute powertrace ONCE per j iteration (outside the k loop)
    2. Use pre-computed powertrace in both f_loop and f_loop_odd calls

    Args:
        A: input matrix
        D: vector to find loop hafnian batch
        fixed_edge_reps: fixed number of edge repetition
        batch_max: maximum number of photons for m mode
        even_cutoff: cutoff for paired modes
        glynn: determines the method used to evaluate the loop hafnian batch

    Returns:
        H_batch: matrix that contains batched loop hafnian with threshold detectors
    """
    oddloop = D[:, 0]
    oddV = A[0, :]

    oddloop0 = D[:, 1]
    oddV0 = A[1, :]

    n = A.shape[0]
    N_fixed = 2 * fixed_edge_reps.sum() + 1
    N_max = N_fixed + 2 * batch_max + even_cutoff + 1

    n_D = D.shape[0]

    edge_reps = np.concatenate((np.array([batch_max, 1]), fixed_edge_reps))
    steps = np.prod(edge_reps + 1)

    # Precompute binomial coefficients
    max_binom = edge_reps.max() + even_cutoff
    binoms = precompute_binoms(max_binom) # type: ignore

    H_batch = np.zeros((n_D, 2 * batch_max + even_cutoff + 2), dtype=np.complex128)

    for j in prange(steps):
        Hnew = np.zeros((n_D, 2 * batch_max + even_cutoff + 2), dtype=np.complex128)

        kept_edges = find_kept_edges(j, edge_reps) # type: ignore
        edges_sum = kept_edges.sum()

        binom_prod = 1.0
        for i in range(1, n // 2):
            binom_prod *= binoms[edge_reps[i], kept_edges[i]]

        delta = 2 * kept_edges - edge_reps if glynn else kept_edges # type: ignore

        AX_S, XD_S, D_S, oddVX_S = get_submatrices(delta, A, D[0, :], oddV) # type: ignore
        AX_S_copy = AX_S.copy()

        # OPTIMIZATION 2: Compute powertrace ONCE per j iteration
        powtrace_len = N_max // 2 + 2
        powtrace_arr = powertrace(AX_S_copy, powtrace_len)

        for k in range(n_D):
            XD_S, D_S = get_Dsubmatrices(delta, D[k, :]) # type: ignore

            if kept_edges[0] == 0 and kept_edges[1] == 0:
                oddVX_S0 = get_submatrix_batch_odd0(delta, oddV0)
                plus_minus = (-1) ** (N_fixed // 2 - edges_sum)
                f = f_loop_odd_with_powtrace(AX_S, XD_S, D_S, N_fixed, oddloop0[k], oddVX_S0, powtrace_arr)[N_fixed] # type: ignore
                Hnew[k, 0] += binom_prod * plus_minus * f

            # OPTIMIZATION 1: Use pre-computed powertrace
            f_even = f_loop_with_powtrace(AX_S, XD_S, D_S, N_max, powtrace_arr) # type: ignore
            f_odd = f_loop_odd_with_powtrace(AX_S, XD_S, D_S, N_max, oddloop[k], oddVX_S, powtrace_arr) # type: ignore

            for N_det in range(2 * kept_edges[0] + 1, 2 * batch_max + even_cutoff + 2):
                N = N_fixed + N_det
                plus_minus = (-1) ** (N // 2 - edges_sum)

                n_det_binom_prod = binoms[(N_det - 1) // 2, kept_edges[0]] * binom_prod

                if N % 2 == 0:
                    Hnew[k, N_det] += n_det_binom_prod * plus_minus * f_even[N // 2]
                else:
                    Hnew[k, N_det] += n_det_binom_prod * plus_minus * f_odd[N]

        H_batch += Hnew

    if glynn:
        for j in range(H_batch.shape[1]):
            x = N_fixed + j
            H_batch[:, j] *= 0.5 ** (x // 2)

    return H_batch


# =============================================================================
# Main optimized function (drop-in replacement)
# =============================================================================

def loop_hafnian_batch_gamma_optimized(
    A: np.ndarray,
    D: np.ndarray,
    fixed_reps: np.ndarray,
    N_cutoff: int,
    glynn: bool = True
) -> np.ndarray:
    """
    Optimized loop hafnian batch gamma computation.

    This is a drop-in replacement for thewalrus.loop_hafnian_batch_gamma.loop_hafnian_batch_gamma
    with the following optimizations:

    1. Powertrace caching: Compute powertrace(AX) once and share between f_loop and f_loop_odd
    2. Batch optimization: Hoist powertrace computation outside the displacement vector loop

    Args:
        A: input matrix
        D: vector to find loop hafnian batch
        fixed_reps: fixed number of edge repetition
        N_cutoff: max number of photons for m mode
        glynn: determines the method used to evaluate the loop hafnian batch

    Returns:
        loop hafnian batch gamma: matrix that contains the batched loop hafnian
        with threshold detectors
    """
    # Input validation
    n = A.shape[0]
    assert A.shape[1] == n
    assert D.shape[1] == n
    assert len(fixed_reps) == n - 1

    nz = np.nonzero(list(fixed_reps) + [1])[0]
    Anz = A[np.ix_(nz, nz)]
    Dnz = D[:, nz]

    fixed_reps = np.asarray(fixed_reps)
    fixed_reps_nz = fixed_reps[nz[:-1]]

    fixed_edges, fixed_m_reps, oddmode = matched_reps(fixed_reps_nz)

    if oddmode is None:
        batch_max = N_cutoff // 2
        odd_cutoff = N_cutoff % 2
        edges = add_batch_edges_even(fixed_edges)
        Ax = Anz[np.ix_(edges, edges)].astype(np.complex128)
        Dx = Dnz[:, edges].astype(np.complex128)
        return _calc_loop_hafnian_batch_gamma_even_optimized(
            Ax, Dx, fixed_m_reps, batch_max, odd_cutoff, glynn=glynn
        )

    edges = add_batch_edges_odd(fixed_edges, oddmode)
    Ax = Anz[np.ix_(edges, edges)].astype(np.complex128)
    Dx = Dnz[:, edges].astype(np.complex128)
    batch_max = (N_cutoff - 1) // 2
    even_cutoff = 1 - (N_cutoff % 2)
    return _calc_loop_hafnian_batch_gamma_odd_optimized(
        Ax, Dx, fixed_m_reps, batch_max, even_cutoff, glynn=glynn
    )
