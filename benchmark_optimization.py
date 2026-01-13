"""
Isolated Benchmark: loop_hafnian_batch_gamma
============================================

Minimal benchmark that tests the hafnian function directly with synthetic inputs.
No external dependencies beyond thewalrus and numpy.

Usage:
    python benchmark_hafnian_isolated.py
"""

from __future__ import annotations

import sys
import time
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from thewalrus.loop_hafnian_batch_gamma import loop_hafnian_batch_gamma
from optimized.loop_hafnian_batch_gamma import loop_hafnian_batch_gamma as loop_hafnian_batch_gamma_optimized


class BenchmarkResult(TypedDict):
    """Type definition for benchmark results."""
    orig_mean: float
    orig_std: float
    opt_mean: float
    opt_std: float
    speedup: float
    max_rel_diff: float
    correct: bool


def generate_random_symmetric_matrix(n: int, seed: int) -> NDArray[np.complex128]:
    """Generate random complex symmetric matrix."""
    np.random.seed(seed)
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    return (A + A.T) / 2


def generate_random_gamma_batch(n: int, batch_size: int, seed: int) -> NDArray[np.complex128]:
    """Generate batch of random displacement vectors."""
    np.random.seed(seed + 1000)
    return np.random.randn(batch_size, n) + 1j * np.random.randn(batch_size, n)


def generate_random_det_pattern(n: int, seed: int) -> NDArray[np.int64]:
    """Generate random detection pattern (photon counts per mode).

    Uses geometric distribution to mimic realistic GBS detection:
    - 0 is most common (no photon)
    - Higher counts exponentially less likely
    - Capped at 5 for computational tractability
    """
    np.random.seed(seed + 2000)
    # Geometric-like distribution: p=0.5 gives ~50% zeros, ~25% ones, etc.
    pattern = np.random.geometric(p=0.5, size=n) - 1  # Shift to start at 0
    return np.clip(pattern, 0, 5).astype(np.int64)


def benchmark_single(
    n: int,
    batch_size: int,
    cutoff: int,
    n_trials: int,
    seed_base: int
) -> BenchmarkResult:
    """Benchmark both implementations for given parameters."""
    original_times: list[float] = []
    optimized_times: list[float] = []
    max_rel_diff: float = 0.0

    for trial in range(n_trials):
        seed = seed_base + trial * 100

        # Generate inputs (same for both)
        A = generate_random_symmetric_matrix(n, seed)
        gamma = generate_random_gamma_batch(n, batch_size, seed)
        det = generate_random_det_pattern(n - 1, seed) if n > 1 else np.array([], dtype=int)

        # Benchmark original
        start = time.perf_counter()
        result_orig = loop_hafnian_batch_gamma(A, gamma, det, cutoff)
        original_times.append(time.perf_counter() - start)

        # Benchmark optimized (same inputs)
        start = time.perf_counter()
        result_opt = loop_hafnian_batch_gamma_optimized(A, gamma, det, cutoff)
        optimized_times.append(time.perf_counter() - start)

        # Check correctness (relative error)
        rel_diff = np.max(np.abs(result_orig - result_opt) / (np.abs(result_orig) + 1e-15))
        max_rel_diff = max(max_rel_diff, rel_diff)

    return {
        "orig_mean": float(np.mean(original_times) * 1000),
        "orig_std": float(np.std(original_times) * 1000),
        "opt_mean": float(np.mean(optimized_times) * 1000),
        "opt_std": float(np.std(optimized_times) * 1000),
        "speedup": float(np.mean(original_times) / np.mean(optimized_times)),
        "max_rel_diff": max_rel_diff,
        "correct": max_rel_diff < 1e-9  # Relative error tolerance (machine precision for large matrices)
    }


def warmup() -> None:
    """JIT warmup."""
    print("JIT warmup...", end=" ")
    sys.stdout.flush()
    for _ in range(10):
        A = generate_random_symmetric_matrix(6, 999)
        gamma = generate_random_gamma_batch(6, 10, 999)
        det = generate_random_det_pattern(5, 999)
        loop_hafnian_batch_gamma(A, gamma, det, 1)
        loop_hafnian_batch_gamma_optimized(A, gamma, det, 1)
    print("Done\n")


def main() -> None:
    print("=" * 75)
    print("ISOLATED BENCHMARK: loop_hafnian_batch_gamma")
    print("=" * 75)
    print("\nTest conditions:")
    print("  - Random complex symmetric matrices")
    print("  - Random complex displacement vectors")
    print("  - Random detection patterns")
    print("  - Multiple trials per configuration for statistical robustness")
    print()

    warmup()

    # Test configurations: (n, batch_size, cutoff, n_trials)
    configs = [
        (4,  10, 1, 50),
        (6,  10, 1, 50),
        (8,  10, 1, 40),
        (10, 10, 1, 30),
        (12, 10, 1, 25),
        (14, 10, 1, 20),
        (16, 10, 1, 15),
        (18, 10, 1, 12),
        (20, 10, 1, 10),
        (22, 10, 1, 8),
        (24, 10, 1, 6),
        (26, 10, 1, 5),
        (28, 10, 1, 4),
        (30, 10, 1, 3),
    ]

    print(f"{'n':>4} | {'batch':>5} | {'trials':>6} | {'Original (ms)':>18} | {'Optimized (ms)':>18} | {'Speedup':>8} | {'Rel Err':>10}")
    print("-" * 85)

    results = []
    for n, batch_size, cutoff, n_trials in configs:
        r = benchmark_single(n, batch_size, cutoff, n_trials, seed_base=42)
        results.append((n, r))

        orig_str = f"{r['orig_mean']:7.2f} +- {r['orig_std']:6.2f}"
        opt_str = f"{r['opt_mean']:7.2f} +- {r['opt_std']:6.2f}"

        print(f"{n:>4} | {batch_size:>5} | {n_trials:>6} | {orig_str:>18} | {opt_str:>18} | {r['speedup']:>7.2f}x | {r['max_rel_diff']:>10.2e}")

    print("-" * 85)

    # Summary
    avg_speedup = np.mean([r['speedup'] for _, r in results])
    all_correct = all(r['correct'] for _, r in results)

    print(f"\nAverage speedup: {avg_speedup:.2f}x")
    print(f"All correctness checks: {'PASSED' if all_correct else 'FAILED'}")

    # Additional test: vary batch size across multiple n values
    print("\n" + "=" * 90)
    print("BATCH SIZE SCALING (varying n and batch_size)")
    print("=" * 90)

    batch_sizes = [1, 5, 10, 20, 50]
    n_values = [10, 12, 14, 16, 18, 20, 22, 24]

    # Header
    header = f"{'n':>4} |"
    for bs in batch_sizes:
        header += f" {'b='+str(bs):>8}"
    print(header)
    print("-" * 55)

    for n in n_values:
        row = f"{n:>4} |"
        n_trials = max(3, 20 - n)  # Fewer trials for larger n
        for batch_size in batch_sizes:
            r = benchmark_single(n, batch_size, 1, n_trials, seed_base=123)
            row += f" {r['speedup']:>7.2f}x"
        print(row)

    print("-" * 55)
    print("(Values show speedup: Original / Optimized)")


if __name__ == "__main__":
    main()
