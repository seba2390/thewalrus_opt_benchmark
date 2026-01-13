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
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

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
powtrace = powertrace(AX, ...)   # Compute once

for k in prange(n_d):
    ...
    f_loop_with_powtrace(..., powtrace)      # Reuse
    f_loop_odd_with_powtrace(..., powtrace)  # Reuse
```


## Benchmark Results

See [RESULTS.md](RESULTS.md) for detailed performance benchmarks showing **3-6x speedup** for typical use cases.
