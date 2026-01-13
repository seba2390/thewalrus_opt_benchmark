# Benchmark Results

## Test Conditions

- Random complex symmetric matrices
- Random complex displacement vectors (batch)
- Realistic detection patterns (geometric distribution, 0-5 photons)
- Multiple trials per configuration for statistical robustness

---

## Results on Apple M3 Pro

**Test Environment:**
- **Hardware:** MacBook Pro (Nov 2023), Apple M3 Pro chip, 18 GB RAM
- **OS:** macOS Sonoma 14.4

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

---

## Results on Apple M1 Pro

**Test Environment:**
- **Hardware:** MacBook Pro, Apple M1 Pro chip, 16 GB RAM
- **OS:** macOS Sequoia 15.6.1
- **Python:** 3.12.9

### Matrix Size Scaling (batch_size=10)

| n  | Trials | Original (ms)     | Optimized (ms)    | Speedup | Rel Error |
|----|--------|-------------------|-------------------|---------|-----------|
| 4  | 50     | 0.98 ± 2.92       | 1.01 ± 3.22       | 0.97x   | 7.54e-14  |
| 6  | 50     | 0.60 ± 0.12       | 0.56 ± 0.07       | 1.07x   | 1.45e-13  |
| 8  | 40     | 0.74 ± 0.21       | 0.64 ± 0.13       | 1.15x   | 5.95e-14  |
| 10 | 30     | 1.02 ± 0.79       | 0.77 ± 0.31       | 1.32x   | 2.10e-13  |
| 12 | 25     | 2.13 ± 1.97       | 1.23 ± 0.78       | 1.73x   | 9.77e-14  |
| 14 | 20     | 3.73 ± 4.07       | 1.93 ± 1.43       | 1.93x   | 2.97e-13  |
| 16 | 15     | 10.17 ± 16.91     | 3.93 ± 4.78       | 2.59x   | 3.38e-12  |
| 18 | 12     | 12.97 ± 7.78      | 5.82 ± 3.52       | 2.23x   | 5.79e-13  |
| 20 | 10     | 29.84 ± 28.86     | 11.58 ± 9.05      | 2.58x   | 3.17e-12  |
| 22 | 8      | 144.62 ± 166.13   | 46.24 ± 47.18     | 3.13x   | 3.16e-12  |
| 24 | 6      | 276.44 ± 226.30   | 86.00 ± 63.35     | 3.21x   | 8.69e-12  |
| 26 | 5      | 1093.52 ± 1177.86 | 265.16 ± 278.42   | 4.12x   | 2.60e-10  |
| 28 | 4      | 1487.47 ± 581.19  | 404.07 ± 170.50   | 3.68x   | 5.70e-12  |
| 30 | 3      | 1601.35 ± 699.34  | 448.58 ± 205.77   | 3.57x   | 8.33e-12  |

**Average speedup: 2.38x**

### Batch Size Scaling (speedup by n and batch_size)

| n  | b=1   | b=5   | b=10  | b=20  | b=50  |
|----|-------|-------|-------|-------|-------|
| 10 | 0.92x | 1.25x | 1.54x | 1.66x | 2.11x |
| 12 | 1.11x | 1.68x | 1.84x | 2.38x | 2.66x |
| 14 | 1.14x | 1.86x | 2.43x | 2.54x | 2.81x |
| 16 | 1.13x | 1.97x | 2.24x | 2.05x | 2.83x |
| 18 | 1.17x | 2.04x | 2.70x | 2.83x | 2.69x |
| 20 | 1.36x | 2.43x | 2.64x | 2.41x | 2.39x |
| 22 | 1.37x | 2.15x | 2.46x | 2.52x | 2.82x |
| 24 | 1.33x | 2.36x | 2.28x | 2.55x | 3.01x |

---

## Correctness

All relative errors are at machine precision (~1e-10 to 1e-14). Verified by running identical random seeds through both implementations and comparing outputs.
