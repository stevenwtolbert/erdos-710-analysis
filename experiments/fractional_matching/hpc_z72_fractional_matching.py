#!/usr/bin/env python3
"""
ERDŐS 710 — Z72: FRACTIONAL MATCHING APPROACH

The LP relaxation of bipartite matching:
  For each edge (k,h), assign weight w(k,h) ≥ 0
  Left constraint:  Σ_h w(k,h) = 1   for all k ∈ S₊
  Right constraint: Σ_k w(k,h) ≤ 1   for all h ∈ H

By total unimodularity: fractional matching exists ⟺ integer matching exists.

The UNIFORM fractional matching: w(k,h) = 1/deg(k) for each edge (k,h).
  Left constraints: Σ_h 1/deg(k) = deg(k)/deg(k) = 1  ✓ automatically
  Right constraints: load(h) = Σ_{k: k|h, k∈S₊} 1/deg(k) ≤ 1  ← CHECK THIS

If max_h load(h) ≤ 1, we have a valid fractional perfect matching.

We also check WEIGHTED variants:
  w(k,h) = c_k for each edge, where c_k is chosen to balance loads.
"""

import math, time, sys
from collections import Counter

C_TARGET = 2 / math.e**0.5
EPS = 0.05

def log(msg):
    print(msg); sys.stdout.flush()

def compute_params(n):
    ln_n = math.log(n)
    ln_ln_n = math.log(ln_n) if ln_n > 1 else 0.1
    L = (C_TARGET + EPS) * n * math.sqrt(ln_n / ln_ln_n)
    M = n + L
    B = int(math.sqrt(M))
    return L, M, B

def sieve_primes(limit):
    if limit < 2: return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            for mult in range(p*p, limit + 1, p):
                sieve[mult] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def get_smooth_numbers_fast(B, lo, hi):
    if hi <= lo: return []
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


def analyze_fractional_matching(n):
    """Check if uniform fractional matching works."""
    L, M, B = compute_params(n)
    n_half = n // 2
    nL = int(n + L)
    delta = 2 * M / n - 1

    S_plus = get_smooth_numbers_fast(B, B, n_half)
    H_smooth = get_smooth_numbers_fast(B, n, nL)

    if not S_plus or not H_smooth:
        return None

    H_set = set(H_smooth)
    S_set = set(S_plus)

    # Compute left degrees
    left_deg = {}
    for k in S_plus:
        lo_mult = n // k + 1
        hi_mult = nL // k
        deg = 0
        for m in range(lo_mult, hi_mult + 1):
            h = k * m
            if h in H_set:
                deg += 1
        left_deg[k] = deg

    # For each target h, compute load(h) = Σ_{k|h, k∈S₊} 1/deg(k)
    # Need to find smooth divisors of h in S_plus
    primes_list = sieve_primes(B)

    loads = []
    max_load = 0
    max_load_h = 0
    max_load_details = []

    for h in H_smooth:
        # Factorize h
        factors = {}
        temp = h
        for p in primes_list:
            if p * p > temp:
                break
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp //= p
        if temp > 1:
            if temp <= B:
                factors[temp] = factors.get(temp, 0) + 1
            else:
                loads.append(0)
                continue

        # Enumerate divisors
        divisors = [1]
        for p, e in factors.items():
            new_divisors = []
            pe = 1
            for i in range(e + 1):
                for d in divisors:
                    new_divisors.append(d * pe)
                pe *= p
            divisors = new_divisors

        # Find which divisors are in S_plus
        load = 0
        contributors = []
        for d in divisors:
            if B < d <= n_half and d in S_set:
                deg_d = left_deg[d]
                if deg_d > 0:
                    load += 1.0 / deg_d
                    contributors.append((d, deg_d))

        loads.append(load)
        if load > max_load:
            max_load = load
            max_load_h = h
            max_load_details = contributors

    loads_sorted = sorted(loads)
    avg_load = sum(loads) / len(loads) if loads else 0
    p99_load = loads_sorted[min(len(loads_sorted)-1, 99*len(loads_sorted)//100)]
    p999_load = loads_sorted[min(len(loads_sorted)-1, 999*len(loads_sorted)//1000)]

    # Count how many targets have load > 1
    over_1 = sum(1 for l in loads if l > 1.0)

    return {
        'n': n, 'B': B, 'delta': delta,
        'n_left': len(S_plus), 'n_right': len(H_smooth),
        'max_load': max_load, 'avg_load': avg_load,
        'p99_load': p99_load, 'p999_load': p999_load,
        'over_1': over_1,
        'max_load_h': max_load_h,
        'max_load_details': max_load_details,
        'min_left_deg': min(left_deg.values()),
        'max_left_deg': max(left_deg.values()),
    }


def main():
    test_ns = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]

    log("ERDŐS 710 — Z72: FRACTIONAL MATCHING (UNIFORM 1/deg(k))")
    log("=" * 80)
    log("Check: load(h) = Σ_{k|h, k∈S₊} 1/deg(k) ≤ 1 for all h")
    log("If max load ≤ 1 ⟹ valid fractional matching ⟹ perfect integer matching")
    log("")

    for n in test_ns:
        t1 = time.time()
        result = analyze_fractional_matching(n)
        dt = time.time() - t1

        if result is None:
            log(f"  n={n}: SKIP")
            continue

        cond = "✓ PASS" if result['max_load'] <= 1.0 else "✗ FAIL"

        log(f"  n = {n:>7} | δ = {result['delta']:.2f} | B = {result['B']}")
        log(f"    |S₊| = {result['n_left']}, |H| = {result['n_right']}")
        log(f"    Left deg range: [{result['min_left_deg']}, {result['max_left_deg']}]")
        log(f"    Load: max = {result['max_load']:.4f}, avg = {result['avg_load']:.4f}, "
            f"P99 = {result['p99_load']:.4f}, P99.9 = {result['p999_load']:.4f}")
        log(f"    Targets with load > 1: {result['over_1']}")
        log(f"    Uniform fractional matching: {cond}")
        if result['max_load'] > 1.0 and result['max_load_details']:
            log(f"    Worst h = {result['max_load_h']}: load = {result['max_load']:.4f}")
            top_contribs = sorted(result['max_load_details'], key=lambda x: -1/x[1])[:5]
            for k, deg in top_contribs:
                log(f"      k={k} (deg={deg}, contrib={1/deg:.4f})")
        log(f"    ({dt:.1f}s)")
        log("")

    log("Done.")


if __name__ == '__main__':
    main()
