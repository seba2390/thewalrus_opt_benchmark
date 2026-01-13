# Loop Hafnian Batch Gamma Optimization

## Summary

Eliminate redundant `powertrace(AX)` computations in `loop_hafnian_batch_gamma` by hoisting the calculation outside the displacement vector loop.

## Files

| File | Description |
|------|-------------|
| `loop_hafnian_optimized.py` | Optimized implementation (drop-in replacement) |
| `benchmark_hafnian_isolated.py` | Benchmark script comparing original vs optimized |
| `README.md` | Detailed proposal for thewalrus PR |

## Quick Start

```bash
# Create venv and install deps
python3.12.2 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Run benchmark
.venv/bin/python benchmark_hafnian_isolated.py
```

## The Problem

In `_calc_loop_hafnian_batch_gamma_even` and `_calc_loop_hafnian_batch_gamma_odd`, the matrix `AX = A @ X` and its powertrace are identical across all displacement vectors `gamma[k]` in a batch. However, the current implementation recomputes `powertrace(AX)` inside `f_loop` and `f_loop_odd` for every `k` in `range(n_d)`.

**Current code path:**
```
for k in prange(n_d):        # Loop over displacement vectors
    ...
    f_loop(A, ...)           # Calls powertrace(AX) internally
    f_loop_odd(A, ...)       # Calls powertrace(AX) again
```

The powertrace computation is O(n²) per call and dominates runtime for large matrices.

## The Fix

Compute `powertrace(AX)` once before the loop, pass it to modified `f_loop` and `f_loop_odd` functions.

**Optimized code path:**
```
AX = A @ X
powtrace = powertrace(AX, n+1)   # Compute once

for k in prange(n_d):
    ...
    f_loop_with_powtrace(A, ..., powtrace)      # Reuse
    f_loop_odd_with_powtrace(A, ..., powtrace)  # Reuse
```

## Changes Required

1. Add `powertrace` function (extract from existing `f_loop`)
2. Add `f_loop_with_powtrace(A, AX, XD, D_diag, n, powtrace)`
3. Add `f_loop_odd_with_powtrace(A, AX, XD, D_diag, n, powtrace, oddloop, oddVX)`
4. Modify `_calc_loop_hafnian_batch_gamma_even` to hoist powertrace
5. Modify `_calc_loop_hafnian_batch_gamma_odd` to hoist powertrace

## Benchmark Results

### Test Environment

- **Hardware:** MacBook Pro (Nov 2023), Apple M3 Pro chip, 18 GB RAM
- **OS:** macOS Sonoma 14.4


### Test Conditions

- Random complex symmetric matrices
- Random complex displacement vectors (batch)
- Realistic detection patterns (geometric distribution, 0-5 photons)
- Multiple trials per configuration for statistical robustness

### Matrix Size Scaling (batch_size=10)

| n  | Trials | Original (ms)     | Optimized (ms)    | Speedup | Rel Error |
|----|--------|-------------------|-------------------|---------|-----------|
| 4  | 50     | 0.75 ± 2.22       | 0.77 ± 2.40       | 0.98x   | 7.60e-14  |
| 6  | 50     | 0.48 ± 0.12       | 0.44 ± 0.05       | 1.09x   | 1.40e-13  |
| 8  | 40     | 0.59 ± 0.20       | 0.50 ± 0.09       | 1.18x   | 5.49e-14  |
| 10 | 30     | 0.90 ± 0.69       | 0.60 ± 0.21       | 1.50x   | 1.99e-13  |
| 12 | 25     | 1.93 ± 1.82       | 0.94 ± 0.61       | 2.05x   | 9.13e-14  |
| 14 | 20     | 3.60 ± 3.88       | 1.41 ± 1.05       | 2.56x   | 2.89e-13  |
| 16 | 15     | 9.29 ± 12.09      | 3.26 ± 2.71       | 2.85x   | 3.13e-12  |
| 18 | 12     | 12.02 ± 7.44      | 3.82 ± 2.15       | 3.15x   | 5.83e-13  |
| 20 | 10     | 27.52 ± 24.58     | 7.19 ± 6.35       | 3.83x   | 3.38e-12  |
| 22 | 8      | 132.59 ± 148.94   | 28.41 ± 28.75     | 4.67x   | 3.02e-12  |
| 24 | 6      | 243.74 ± 195.30   | 50.47 ± 37.36     | 4.83x   | 1.02e-11  |
| 26 | 5      | 1015.31 ± 1111.92 | 169.25 ± 166.43   | 6.00x   | 2.59e-10  |
| 28 | 4      | 1418.64 ± 677.98  | 253.07 ± 105.70   | 5.61x   | 5.73e-12  |
| 30 | 3      | 1530.38 ± 751.73  | 277.27 ± 119.13   | 5.52x   | 7.89e-12  |

**Average speedup: 3.27x**

### Batch Size Scaling (speedup by n and batch_size)

| n  | b=1   | b=5   | b=10  | b=20  | b=50  |
|----|-------|-------|-------|-------|-------|
| 10 | 0.92x | 1.29x | 1.71x | 1.95x | 2.36x |
| 12 | 1.14x | 1.69x | 2.23x | 2.73x | 3.46x |
| 14 | 1.21x | 2.41x | 3.13x | 3.52x | 3.90x |
| 16 | 1.19x | 2.75x | 3.06x | 4.03x | 4.30x |
| 18 | 1.52x | 2.82x | 3.32x | 3.89x | 4.27x |
| 20 | 1.19x | 3.00x | 3.55x | 4.17x | 4.25x |
| 22 | 1.45x | 3.36x | 3.71x | 3.70x | 4.56x |
| 24 | 1.41x | 2.91x | 3.72x | 4.24x | 4.56x |

**Key insight:** Speedup scales with both matrix size (n) and batch size. The optimization has minimal effect for batch_size=1 (no redundant computation to eliminate) but achieves **4-6x speedup** for larger batches.

## Correctness

All relative errors are at machine precision (~1e-10 to 1e-14). Verified by running identical random seeds through both implementations and comparing outputs.
