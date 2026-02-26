#!/usr/bin/env python3
"""
ERDŐS 710 — Z30: EXHAUSTIVE HOPCROFT-KARP VERIFICATION

Exhaustive Hall verification at EVERY integer n ∈ [4, n₀].
Uses optimized Python HK with adjacency lists.

For each n:
  1. Compute L = (2/√e + ε)·n·√(ln n / ln ln n), M = L - n
  2. Build bipartite graph G = (V, H) where V = {1,...,⌊n/2⌋}, H = (2n, n+L]
  3. Run Hopcroft-Karp to find maximum matching
  4. By König's theorem: max matching = |V| ⟺ Hall for ALL subsets

This is THE gold standard — no sampling, no heuristics, mathematically airtight.

Usage:
  python3 hpc_z30_exhaustive_hk.py --n_start 4 --n_end 10000
  python3 hpc_z30_exhaustive_hk.py --n_start 10001 --n_end 50000 --checkpoint_interval 1000
  python3 hpc_z30_exhaustive_hk.py --n_start 4 --n_end 100000 --parallel 8
"""

import sys
import time
import argparse
import os
from math import log, sqrt, exp, floor
from collections import deque

C_TARGET = 2 / sqrt(exp(1))
EPS = 0.05


def target_L(n, eps=EPS):
    """Compute L = (2/√e + ε)·n·√(ln n / ln ln n)."""
    if n < 3:
        return 3 * n
    return int((C_TARGET + eps) * n * sqrt(log(n) / log(log(n))))


def hopcroft_karp_fast(n_val):
    """
    Optimized Hopcroft-Karp for the divisibility bipartite graph.

    Left vertices: V = {1, 2, ..., N} where N = floor(n/2)
    Right vertices: H = {2n+1, 2n+2, ..., n+L}
    Edges: k -> m iff k divides m

    Returns: (max_matching_size, N, M) where M = L - n = |H|.
    """
    L = target_L(n_val)
    if L <= n_val:
        # Trivial case: M <= 0
        return (0, n_val // 2, 0)

    M = L - n_val
    N = n_val // 2
    lo_h = 2 * n_val + 1  # first element of H
    hi_h = n_val + L       # last element of H

    if N == 0:
        return (0, 0, M)

    # Build adjacency list: for each k in V, list of target indices in H
    # Targets indexed 0..M-1 where index i = target (2n + 1 + i)
    adj = [[] for _ in range(N + 1)]  # adj[k] = list of target indices

    for k in range(1, N + 1):
        # Multiples of k in (2n, n+L] = {k*j : j = ceil((2n+1)/k) to floor((n+L)/k)}
        j_start = (2 * n_val) // k + 1
        j_end = (n_val + L) // k
        for j in range(j_start, j_end + 1):
            m = k * j
            idx = m - lo_h  # target index 0-based
            if 0 <= idx < M:
                adj[k].append(idx)

    # Hopcroft-Karp with BFS + DFS
    match_l = [0] * (N + 1)     # match_l[k] = target index + 1 (0 = unmatched)
    match_r = [0] * M            # match_r[idx] = k (0 = unmatched)
    INF = N + M + 1

    dist = [0] * (N + 1)

    def bfs():
        queue = deque()
        for k in range(1, N + 1):
            if match_l[k] == 0:
                dist[k] = 0
                queue.append(k)
            else:
                dist[k] = INF
        found = False
        while queue:
            k = queue.popleft()
            if dist[k] < INF:
                for idx in adj[k]:
                    w = match_r[idx]
                    if w == 0:
                        found = True
                    elif dist[w] == INF:
                        dist[w] = dist[k] + 1
                        queue.append(w)
        return found

    def dfs(k):
        for idx in adj[k]:
            w = match_r[idx]
            if w == 0 or (dist[w] == dist[k] + 1 and dfs(w)):
                match_l[k] = idx + 1
                match_r[idx] = k
                return True
        dist[k] = INF
        return False

    matching = 0
    while bfs():
        for k in range(1, N + 1):
            if match_l[k] == 0:
                if dfs(k):
                    matching += 1

    return (matching, N, M)


def verify_single(n_val):
    """Verify Hall's condition for a single n."""
    matching, N, M = hopcroft_karp_fast(n_val)
    hall = (matching == N)
    deficit = N - matching
    return {
        'n': n_val,
        'N': N,
        'M': M,
        'matching': matching,
        'deficit': deficit,
        'hall': hall,
    }


def run_exhaustive(n_start, n_end, checkpoint_interval=5000, output_dir=None):
    """
    Run exhaustive HK verification for every integer n in [n_start, n_end].
    """
    total = n_end - n_start + 1
    failures = []
    checked = 0
    t_start = time.time()
    last_checkpoint = time.time()

    print(f"=" * 70)
    print(f"  Z30: EXHAUSTIVE HOPCROFT-KARP VERIFICATION")
    print(f"  Range: n = {n_start:,} to {n_end:,} ({total:,} values)")
    print(f"=" * 70)

    # Progress tracking
    for n_val in range(n_start, n_end + 1):
        result = verify_single(n_val)
        checked += 1

        if not result['hall']:
            failures.append(result)
            print(f"  *** FAILURE at n = {n_val}: matching = {result['matching']}, "
                  f"N = {result['N']}, deficit = {result['deficit']} ***")

        # Progress output
        now = time.time()
        if now - last_checkpoint >= 10 or n_val == n_end:
            elapsed = now - t_start
            rate = checked / elapsed if elapsed > 0 else 0
            remaining = (total - checked) / rate if rate > 0 else 0
            print(f"  n = {n_val:>8,}  ({checked:>8,}/{total:,}  {100*checked/total:>5.1f}%)"
                  f"  {rate:>6.0f}/s  ETA: {remaining:>6.0f}s"
                  f"  failures: {len(failures)}", flush=True)
            last_checkpoint = now

        # Checkpoint file
        if checkpoint_interval and checked % checkpoint_interval == 0:
            _write_checkpoint(n_start, n_val, checked, failures, t_start, output_dir)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE: {checked:,} values verified in {elapsed:.1f}s ({checked/elapsed:.0f}/s)")
    print(f"  Failures: {len(failures)}")
    if failures:
        for f in failures[:10]:
            print(f"    n={f['n']}: deficit={f['deficit']}")
    else:
        print(f"  *** ALL PASS — Hall verified for EVERY n in [{n_start}, {n_end}] ***")
    print(f"{'=' * 70}")

    # Final checkpoint
    _write_checkpoint(n_start, n_end, checked, failures, t_start, output_dir, final=True)

    return failures


def _write_checkpoint(n_start, n_current, checked, failures, t_start, output_dir, final=False):
    """Write checkpoint file."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_file = os.path.join(output_dir, 'z30_checkpoint.txt')
    elapsed = time.time() - t_start
    with open(ckpt_file, 'w') as f:
        f.write(f"Z30 Exhaustive HK Verification\n")
        f.write(f"Range: [{n_start}, {n_current}]\n")
        f.write(f"Checked: {checked}\n")
        f.write(f"Failures: {len(failures)}\n")
        f.write(f"Elapsed: {elapsed:.1f}s\n")
        f.write(f"Rate: {checked/elapsed:.0f}/s\n")
        f.write(f"Final: {final}\n")
        if failures:
            f.write(f"Failure details:\n")
            for fl in failures:
                f.write(f"  n={fl['n']}: deficit={fl['deficit']}\n")


def _worker(args):
    """Worker function for parallel verification (must be top-level for pickling)."""
    ns, ne = args
    fails = []
    count = 0
    for n_val in range(ns, ne + 1):
        result = verify_single(n_val)
        count += 1
        if not result['hall']:
            fails.append(result)
    return (ns, ne, count, fails)


def run_parallel(n_start, n_end, n_workers, checkpoint_interval=5000):
    """
    Run exhaustive HK using multiprocessing.
    Splits range into chunks and runs in parallel.
    """
    from multiprocessing import Pool, cpu_count

    if n_workers <= 0:
        n_workers = cpu_count()

    total = n_end - n_start + 1
    chunk_size = max(1, total // n_workers)

    # Split into ranges
    ranges = []
    for i in range(n_workers):
        start = n_start + i * chunk_size
        end = min(n_start + (i + 1) * chunk_size - 1, n_end)
        if start > n_end:
            break
        ranges.append((start, end))
    # Fix last chunk
    if ranges:
        ranges[-1] = (ranges[-1][0], n_end)

    print(f"=" * 70)
    print(f"  Z30: PARALLEL EXHAUSTIVE HK ({n_workers} workers)")
    print(f"  Range: n = {n_start:,} to {n_end:,} ({total:,} values)")
    print(f"  Chunks: {len(ranges)}")
    print(f"=" * 70)

    t_start = time.time()

    with Pool(n_workers) as pool:
        results = pool.map(_worker, ranges)

    elapsed = time.time() - t_start
    total_checked = sum(r[2] for r in results)
    all_failures = []
    for _, _, _, fails in results:
        all_failures.extend(fails)

    print(f"\n{'=' * 70}")
    print(f"  COMPLETE: {total_checked:,} values verified in {elapsed:.1f}s "
          f"({total_checked/elapsed:.0f}/s)")
    print(f"  Failures: {len(all_failures)}")
    if all_failures:
        all_failures.sort(key=lambda x: x['n'])
        for f in all_failures[:10]:
            print(f"    n={f['n']}: deficit={f['deficit']}")
    else:
        print(f"  *** ALL PASS — Hall verified for EVERY n in [{n_start}, {n_end}] ***")
    print(f"{'=' * 70}")

    return all_failures


def main():
    parser = argparse.ArgumentParser(description="Z30: Exhaustive HK verification")
    parser.add_argument('--n_start', type=int, default=4)
    parser.add_argument('--n_end', type=int, default=10000)
    parser.add_argument('--checkpoint_interval', type=int, default=5000)
    parser.add_argument('--parallel', type=int, default=0,
                        help='Number of parallel workers (0=serial)')
    args = parser.parse_args()

    if args.parallel > 0:
        failures = run_parallel(args.n_start, args.n_end, args.parallel,
                                args.checkpoint_interval)
    else:
        failures = run_exhaustive(args.n_start, args.n_end,
                                  args.checkpoint_interval)

    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
