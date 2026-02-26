#!/usr/bin/env python3
"""
Z31b: Refined degree analysis.

Key findings from Z31a:
  - min_L/max_R -> 0 (degree condition fails badly)
  - avg_L/avg_R -> 2+ (on average, massive expansion)
  - max_R targets are highly composite (e.g. 2520 = 2^3*3^2*5*7)
  - min_L sources are near n/2 (large, close to boundary)

Questions:
  1. What percentile of right-degree do we need to trim to get min_L >= trimmed_max_R?
  2. What is the right-degree distribution tail? (How many targets have high degree?)
  3. For the min-degree sources near n/2, what are their targets?
  4. If we remove the top p% of right-degree targets, what is the resulting ratio?
  5. Fractional relaxation: what is avg_L / avg_R when restricted to percentile buckets?
"""

import math
import time
from collections import Counter, defaultdict

C_TARGET = 2.0 / math.sqrt(math.e) + 0.05

def compute_L(n):
    ln_n = math.log(n)
    ln_ln_n = math.log(ln_n)
    return C_TARGET * n * math.sqrt(ln_n / ln_ln_n)

def sieve_smooth_fast(lo, hi, B):
    if lo > hi or B < 2:
        return set()
    is_prime_arr = [True] * (B + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(math.sqrt(B)) + 1):
        if is_prime_arr[i]:
            for j in range(i*i, B + 1, i):
                is_prime_arr[j] = False
    primes = [i for i in range(2, B + 1) if is_prime_arr[i]]
    
    size = hi - lo + 1
    residual = list(range(lo, hi + 1))
    for p in primes:
        start = ((lo + p - 1) // p) * p
        for idx in range(start - lo, size, p):
            while residual[idx] % p == 0:
                residual[idx] //= p
    return {lo + i for i in range(size) if residual[i] == 1}

def get_divisors_in_range(h, lo, hi):
    divisors = []
    sq = int(math.sqrt(h))
    for d in range(1, sq + 1):
        if h % d == 0:
            if lo <= d <= hi:
                divisors.append(d)
            c = h // d
            if c != d and lo <= c <= hi:
                divisors.append(c)
    return divisors

def run_refined(n):
    t0 = time.time()
    L = compute_L(n)
    L_int = int(math.floor(L))
    B = int(math.floor(math.sqrt(n + L_int)))
    half_n = n // 2
    
    S_plus = sieve_smooth_fast(B + 1, half_n, B)
    target_lo = 2 * n + 1
    target_hi = n + L_int
    H_smooth = sieve_smooth_fast(target_lo, target_hi, B)
    
    S_plus_sorted = sorted(S_plus)
    H_smooth_sorted = sorted(H_smooth)
    
    # Compute all degrees
    left_deg = {}
    for k in S_plus_sorted:
        m_lo = (target_lo + k - 1) // k
        m_hi = target_hi // k
        count = 0
        for m in range(m_lo, m_hi + 1):
            if k * m in H_smooth:
                count += 1
        left_deg[k] = count
    
    right_deg = {}
    for h in H_smooth_sorted:
        divs = get_divisors_in_range(h, B + 1, half_n)
        count = sum(1 for d in divs if d in S_plus)
        right_deg[h] = count
    
    min_left = min(left_deg.values())
    
    # Right degree distribution analysis
    right_vals = sorted(right_deg.values(), reverse=True)
    size_H = len(right_vals)
    
    print(f"\nn = {n}, B = {B}, |S_+| = {len(S_plus)}, |H_smooth| = {size_H}")
    print(f"  min_left_deg = {min_left}")
    print(f"  Right degree percentiles:")
    for pct in [100, 99.9, 99.5, 99, 98, 95, 90, 80, 50]:
        idx = max(0, int((100 - pct) / 100 * size_H))
        if idx < size_H:
            print(f"    top {100-pct:.1f}% trimmed -> max_R = {right_vals[idx]}, "
                  f"min_L/max_R = {min_left/right_vals[idx]:.4f}" if right_vals[idx] > 0 
                  else f"    top {100-pct:.1f}% trimmed -> max_R = 0")
    
    # How many targets have right_deg > min_left?
    exceeding = sum(1 for v in right_vals if v > min_left)
    print(f"  Targets with right_deg > min_left ({min_left}): {exceeding} ({100*exceeding/size_H:.2f}%)")
    
    # What fraction of edges go to high-degree targets?
    total_edges = sum(right_deg.values())
    edges_to_high = sum(v for v in right_deg.values() if v > min_left)
    print(f"  Edge fraction to high-deg targets: {edges_to_high/total_edges:.4f}")
    
    # Left degree distribution by position (k near B vs k near n/2)
    quartiles = [[], [], [], []]
    for k in S_plus_sorted:
        rel_pos = (k - B) / (half_n - B)  # 0 = near B, 1 = near n/2
        q = min(3, int(rel_pos * 4))
        quartiles[q].append(left_deg[k])
    
    print(f"  Left degree by position quartile (0=near B, 3=near n/2):")
    for q in range(4):
        if quartiles[q]:
            avg_q = sum(quartiles[q]) / len(quartiles[q])
            min_q = min(quartiles[q])
            max_q = max(quartiles[q])
            print(f"    Q{q}: n={len(quartiles[q])}, min={min_q}, avg={avg_q:.1f}, max={max_q}")
    
    # Deficiency analysis: for each k with small degree, what are the multipliers?
    print(f"  Sources with degree <= {min_left + 2}:")
    shown = 0
    for k in S_plus_sorted:
        if left_deg[k] <= min_left + 2 and shown < 5:
            m_lo = (target_lo + k - 1) // k
            m_hi = target_hi // k
            targets = []
            for m in range(m_lo, m_hi + 1):
                h = k * m
                is_smooth = h in H_smooth
                targets.append((m, h, is_smooth))
            total_mult = m_hi - m_lo + 1
            smooth_mult = sum(1 for _, _, s in targets if s)
            print(f"    k={k}: deg={left_deg[k]}, multiplier range [{m_lo},{m_hi}] "
                  f"({total_mult} total, {smooth_mult} smooth)")
            shown += 1
    
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")
    
    return {
        'n': n, 'min_left': min_left,
        'right_vals': right_vals,
        'exceeding': exceeding,
        'size_H': size_H,
        'size_S': len(S_plus),
    }

def main():
    print("=" * 90)
    print("Z31b: Refined Degree Structure Analysis")
    print("=" * 90)
    
    test_ns = [1000, 5000, 10000, 50000, 100000, 200000]
    results = []
    for n in test_ns:
        r = run_refined(n)
        results.append(r)
    
    # Summary: threshold analysis
    print("\n" + "=" * 90)
    print("THRESHOLD ANALYSIS: What right-degree cap c gives min_L >= c?")
    print("=" * 90)
    print(f"{'n':>8s}  {'min_L':>6s}  {'max_R':>6s}  {'#(R>minL)':>10s}  {'%(R>minL)':>10s}  "
          f"{'#(R>2)':>8s}  {'%(R>2)':>8s}")
    print("-" * 75)
    for r in results:
        exc = r['exceeding']
        exc_pct = 100 * exc / r['size_H']
        gt2 = sum(1 for v in r['right_vals'] if v > 2)
        gt2_pct = 100 * gt2 / r['size_H']
        print(f"{r['n']:>8d}  {r['min_left']:>6d}  {r['right_vals'][0]:>6d}  "
              f"{exc:>10d}  {exc_pct:>9.2f}%  {gt2:>8d}  {gt2_pct:>7.2f}%")

if __name__ == '__main__':
    main()
