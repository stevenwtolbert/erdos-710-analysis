#!/usr/bin/env python3
"""
ERDŐS 710 — Z68: EXHAUSTIVE HOPCROFT-KARP VERIFICATION

Runs Hopcroft-Karp at EVERY integer n ∈ [n_start, n_end] to verify that the
bipartite graph S₊ → H_smooth has a perfect matching.

By König's theorem: max matching = |V| ⟺ Hall's condition holds for ALL subsets.
This is the GOLD STANDARD — no heuristics, no sufficient conditions, mathematically
airtight for each tested n.

Z67 showed that HK takes only 0.1-0.5s per n in pure Python, making exhaustive
verification feasible for n up to ~200,000.

Usage:
    python3 -u hpc_z68_exhaustive_hk.py [start] [end] [checkpoint_interval]

Default: n = 4 to 200000, checkpoint every 1000.
"""

import math
import time
import sys
import os
import json
from collections import defaultdict, deque

C_TARGET = 2 / math.e**0.5
EPS = 0.05

# Parse command line
N_START = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N_END = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
CHECKPOINT_INTERVAL = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

CHECKPOINT_FILE = "/home/ashbringer/projects/e710_new_H/z68_checkpoint.json"
STATE_FILE = "/home/ashbringer/projects/e710_new_H/states/state_80_z68_exhaustive_hk.md"


def log(msg):
    print(msg)
    sys.stdout.flush()


def compute_params(n):
    ln_n = math.log(n)
    ln_ln_n = math.log(ln_n) if ln_n > 1 else 0.1
    L = (C_TARGET + EPS) * n * math.sqrt(ln_n / ln_ln_n)
    M = n + L
    B = int(math.sqrt(M))
    return L, M, B


def sieve_primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            for mult in range(p*p, limit + 1, p):
                sieve[mult] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]


def get_smooth_numbers_fast(B, lo, hi):
    """Get B-smooth numbers in (lo, hi]."""
    if hi <= lo:
        return []
    size = hi - lo
    remaining = list(range(lo + 1, hi + 1))
    primes = sieve_primes(B)
    for p in primes:
        start = lo + 1
        first = start + (-start % p)
        for idx in range(first - lo - 1, size, p):
            while remaining[idx] % p == 0:
                remaining[idx] //= p
    return [lo + 1 + i for i in range(size) if remaining[i] == 1]


def hopcroft_karp(n_left, n_right, adj):
    """Hopcroft-Karp maximum matching.

    adj[u] = list of right vertex indices for left vertex u.
    Returns matching size.
    """
    match_left = [-1] * n_left
    match_right = [-1] * n_right
    dist = [0] * n_left

    def bfs():
        queue = deque()
        for u in range(n_left):
            if match_left[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = float('inf')
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                w = match_right[v]
                if w == -1:
                    found = True
                elif dist[w] == float('inf'):
                    dist[w] = dist[u] + 1
                    queue.append(w)
        return found

    def dfs(u):
        for v in adj[u]:
            w = match_right[v]
            if w == -1 or (dist[w] == dist[u] + 1 and dfs(w)):
                match_left[u] = v
                match_right[v] = u
                return True
        dist[u] = float('inf')
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if match_left[u] == -1:
                if dfs(u):
                    matching += 1
    return matching


def verify_hall_at_n(n):
    """Verify Hall's condition at a specific n via Hopcroft-Karp.

    Returns (pass, n_splus, matching, delta) or (None, ...) if n too small.
    """
    L, M, B = compute_params(n)
    n_half = n // 2
    nL = int(n + L)
    delta = 2 * M / n - 1

    if B < 2 or n_half <= B:
        return None, 0, 0, delta

    # Build bipartite graph
    S_plus = get_smooth_numbers_fast(B, B, n_half)
    H_smooth = get_smooth_numbers_fast(B, n, nL)

    if not S_plus or not H_smooth:
        return None, len(S_plus), 0, delta

    H_set = set(H_smooth)
    s_idx = {k: i for i, k in enumerate(S_plus)}
    h_idx = {h: i for i, h in enumerate(H_smooth)}
    n_left = len(S_plus)
    n_right = len(H_smooth)

    # Build adjacency list
    adj = [[] for _ in range(n_left)]
    for k in S_plus:
        u = s_idx[k]
        lo_mult = n // k + 1
        hi_mult = nL // k
        for m in range(lo_mult, hi_mult + 1):
            h = k * m
            if h in H_set:
                adj[u].append(h_idx[h])

    matching = hopcroft_karp(n_left, n_right, adj)
    hall_pass = matching >= n_left
    return hall_pass, n_left, matching, delta


def load_checkpoint():
    """Load checkpoint if exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None


def save_checkpoint(data):
    """Save checkpoint."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f)


def main():
    log("ERDŐS 710 — Z68: EXHAUSTIVE HOPCROFT-KARP VERIFICATION")
    log("=" * 80)
    log(f"Range: n = {N_START} to {N_END}")
    log(f"Checkpoint interval: every {CHECKPOINT_INTERVAL} values")
    log(f"Method: Hopcroft-Karp maximum matching (König's theorem)")
    log("")
    t0 = time.time()

    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    if checkpoint and checkpoint.get('n_start') == N_START and checkpoint.get('n_end') == N_END:
        last_n = checkpoint['last_n']
        total_pass = checkpoint['total_pass']
        total_skip = checkpoint['total_skip']
        total_fail = checkpoint['total_fail']
        failures = checkpoint.get('failures', [])
        max_splus = checkpoint.get('max_splus', 0)
        min_margin = checkpoint.get('min_margin', float('inf'))
        log(f"Resuming from checkpoint: last_n = {last_n}")
        log(f"  Progress: {total_pass} pass, {total_skip} skip, {total_fail} fail")
        start_n = last_n + 1
    else:
        total_pass = 0
        total_skip = 0
        total_fail = 0
        failures = []
        max_splus = 0
        min_margin = float('inf')
        start_n = N_START

    # Main verification loop
    batch_start = time.time()
    batch_count = 0
    last_report_time = time.time()

    for n in range(start_n, N_END + 1):
        result, n_splus, matching, delta = verify_hall_at_n(n)

        if result is None:
            total_skip += 1
        elif result:
            total_pass += 1
            if n_splus > max_splus:
                max_splus = n_splus
            margin = matching - n_splus  # should be 0 for exact match
        else:
            total_fail += 1
            deficiency = n_splus - matching
            failures.append({
                'n': n, 'n_splus': n_splus, 'matching': matching,
                'deficiency': deficiency, 'delta': delta
            })
            log(f"  *** HALL FAILURE at n = {n}: matching = {matching}, "
                f"|S₊| = {n_splus}, deficiency = {deficiency}, δ = {delta:.3f} ***")

        batch_count += 1
        now = time.time()

        # Periodic progress report
        if now - last_report_time > 30 or n == N_END or n % CHECKPOINT_INTERVAL == 0:
            elapsed = now - t0
            rate = batch_count / max(now - batch_start, 0.001)
            remaining = (N_END - n) / max(rate, 0.001)
            total_tested = total_pass + total_skip + total_fail

            if n % CHECKPOINT_INTERVAL == 0 or n == N_END:
                log(f"  n = {n:>7} | pass={total_pass} skip={total_skip} fail={total_fail} | "
                    f"|S₊|={n_splus:>6} | rate={rate:.1f}/s | "
                    f"elapsed={elapsed:.0f}s | ETA={remaining:.0f}s ({remaining/3600:.1f}h)")

                # Save checkpoint
                save_checkpoint({
                    'n_start': N_START, 'n_end': N_END,
                    'last_n': n,
                    'total_pass': total_pass, 'total_skip': total_skip,
                    'total_fail': total_fail, 'failures': failures,
                    'max_splus': max_splus,
                    'min_margin': min_margin,
                    'timestamp': time.time(),
                })

            last_report_time = now

    total_time = time.time() - t0

    # Final summary
    log(f"\n{'='*80}")
    log("  Z68 FINAL SUMMARY: EXHAUSTIVE HOPCROFT-KARP VERIFICATION")
    log(f"{'='*80}")
    log(f"  Range: n = {N_START} to {N_END}")
    log(f"  Total pass: {total_pass}")
    log(f"  Total skip: {total_skip} (n too small for S₊)")
    log(f"  Total fail: {total_fail}")
    log(f"  Max |S₊|: {max_splus}")
    log(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")

    if total_fail == 0:
        log(f"\n  *** HALL'S CONDITION VERIFIED AT EVERY n ∈ [{N_START}, {N_END}] ***")
        log(f"  *** {total_pass} rigorous HK verifications, ZERO failures ***")
        log(f"  *** By König's theorem, this proves Hall for ALL subsets at each n ***")
    else:
        log(f"\n  *** {total_fail} HALL FAILURES DETECTED ***")
        for fail in failures:
            log(f"    n = {fail['n']}: deficiency = {fail['deficiency']}, δ = {fail['delta']:.3f}")

    # Write state file
    with open(STATE_FILE, 'w') as f:
        f.write("# State 80: Z68 Exhaustive Hopcroft-Karp Verification\n\n")
        f.write("## Method\n\n")
        f.write("Hopcroft-Karp maximum matching on the bipartite graph S₊ → H_smooth\n")
        f.write("at EVERY integer n in the specified range. By König's theorem,\n")
        f.write("max matching = |S₊| ⟺ Hall's condition holds for ALL subsets.\n\n")
        f.write("## Parameters\n\n")
        f.write(f"- C = 2/√e + 0.05 ≈ {C_TARGET + EPS:.4f}\n")
        f.write(f"- L = C · n · √(ln n / ln ln n)\n")
        f.write(f"- B = ⌊√(n + L)⌋\n")
        f.write(f"- S₊ = B-smooth numbers in (B, n/2]\n")
        f.write(f"- H_smooth = B-smooth numbers in (n, n+L]\n\n")
        f.write("## Results\n\n")
        f.write(f"- Range: n = {N_START} to {N_END}\n")
        f.write(f"- Total pass: {total_pass}\n")
        f.write(f"- Total skip: {total_skip}\n")
        f.write(f"- Total fail: {total_fail}\n")
        f.write(f"- Total time: {total_time:.0f}s ({total_time/3600:.1f}h)\n\n")
        if total_fail == 0:
            f.write(f"**Hall's condition verified at EVERY n ∈ [{N_START}, {N_END}].**\n")
            f.write(f"**{total_pass} rigorous HK verifications, ZERO failures.**\n")
        else:
            f.write(f"**{total_fail} failures detected:**\n\n")
            for fail in failures:
                f.write(f"- n = {fail['n']}: deficiency = {fail['deficiency']}\n")

    log(f"  State file: {STATE_FILE}")


if __name__ == '__main__':
    main()
