#!/usr/bin/env python3
"""
Z31c: Stratified Hall analysis.

Key insight from Z31b: The bipartite graph has extreme heterogeneity.
- Sources near B have degree 40-400+ (grows with n)
- Sources near n/2 have degree 3-5 (constant!)
- High-degree targets (highly composite) have deg up to 112

The min-degree sources are near n/2 with multiplier range [7,12].
These sources ONLY connect to targets = k*m for m in {7,8,...,12}.
The targets they connect to are NOT the high-degree targets.

Key question: For sources near n/2 (Q3), what is the max right-degree 
of their neighbors? If Q3's neighbor's max_right_deg is small (say <= 5),
then Hall holds for Q3 locally.

Also: Does the bipartite graph decompose into "layers" where each layer
satisfies Hall independently?

Plan:
1. For each quartile Q_i of sources, compute the neighbor set N(Q_i)
2. For each N(Q_i), compute max_right_deg restricted to Q_i
3. Check if min_left(Q_i) >= max_right_restricted(Q_i)
"""

import math
import time
from collections import defaultdict

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

def run_stratified(n):
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
    
    # Build adjacency: source -> set of targets, target -> set of sources
    src_to_tgt = defaultdict(set)
    tgt_to_src = defaultdict(set)
    
    for k in S_plus_sorted:
        m_lo = (target_lo + k - 1) // k
        m_hi = target_hi // k
        for m in range(m_lo, m_hi + 1):
            h = k * m
            if h in H_smooth:
                src_to_tgt[k].add(h)
                tgt_to_src[h].add(k)
    
    # Partition sources into quartiles by position
    num_q = 8  # Use octiles for finer resolution
    quartiles = [[] for _ in range(num_q)]
    for k in S_plus_sorted:
        rel_pos = (k - B) / (half_n - B)
        q = min(num_q - 1, int(rel_pos * num_q))
        quartiles[q].append(k)
    
    print(f"\nn = {n}, B = {B}, |S_+| = {len(S_plus)}, |H_smooth| = {len(H_smooth)}")
    print(f"  Target interval: ({target_lo}, {target_hi}]")
    print(f"  Multiplier range for k near n/2: [{(target_lo + half_n - 1)//half_n}, {target_hi//half_n}]")
    print()
    
    # For each octile, compute:
    # - Size
    # - min/avg/max left degree
    # - Neighbor set size
    # - max right degree restricted to this octile
    # - Whether restricted Hall holds
    
    print(f"  {'Oct':>4s}  {'k range':>18s}  {'|Q|':>6s}  {'minL':>5s}  {'avgL':>6s}  "
          f"{'maxL':>5s}  {'|N(Q)|':>7s}  {'maxR_Q':>7s}  {'minL/mRQ':>9s}  {'Hall?':>6s}")
    print("  " + "-" * 100)
    
    for q in range(num_q):
        Q = quartiles[q]
        if not Q:
            continue
        Q_set = set(Q)
        
        # Left degrees
        left_degs = [len(src_to_tgt[k]) for k in Q]
        min_l = min(left_degs)
        max_l = max(left_degs)
        avg_l = sum(left_degs) / len(left_degs)
        
        # Neighbor set of Q
        NH_Q = set()
        for k in Q:
            NH_Q.update(src_to_tgt[k])
        
        # Restricted right degree: for each h in NH_Q, count how many sources in Q divide h
        max_r_restricted = 0
        for h in NH_Q:
            restricted_deg = len(tgt_to_src[h] & Q_set)
            max_r_restricted = max(max_r_restricted, restricted_deg)
        
        ratio = min_l / max_r_restricted if max_r_restricted > 0 else float('inf')
        hall = "YES" if ratio >= 1 else "NO"
        
        k_lo = min(Q)
        k_hi = max(Q)
        
        print(f"  Q{q:>2d}  [{k_lo:>7d},{k_hi:>7d}]  {len(Q):>6d}  {min_l:>5d}  {avg_l:>6.1f}  "
              f"{max_l:>5d}  {len(NH_Q):>7d}  {max_r_restricted:>7d}  {ratio:>9.4f}  {hall:>6s}")
    
    # Now check: for the LAST quartile (sources near n/2), what do their targets look like?
    Q_last = quartiles[num_q - 1]
    if Q_last:
        Q_last_set = set(Q_last)
        print(f"\n  Detail for last octile (sources near n/2, Q{num_q-1}):")
        print(f"    |Q| = {len(Q_last)}")
        
        # Multiplier distribution
        mult_counts = defaultdict(int)
        for k in Q_last:
            for h in src_to_tgt[k]:
                m = h // k
                mult_counts[m] += 1
        
        print(f"    Multiplier distribution:")
        for m in sorted(mult_counts.keys()):
            print(f"      m={m}: {mult_counts[m]} edges")
        
        # Target right degree distribution (restricted to Q_last)
        NH_last = set()
        for k in Q_last:
            NH_last.update(src_to_tgt[k])
        
        restricted_degs = []
        for h in NH_last:
            rd = len(tgt_to_src[h] & Q_last_set)
            restricted_degs.append(rd)
        
        from collections import Counter
        rd_hist = Counter(restricted_degs)
        print(f"    Restricted right-degree distribution (|N(Q_last)|={len(NH_last)}):")
        for deg in sorted(rd_hist.keys()):
            print(f"      deg={deg}: {rd_hist[deg]} targets")
        
        # How about Hall ratio for Q_last?
        print(f"    Hall ratio |N(Q_last)|/|Q_last| = {len(NH_last)/len(Q_last):.4f}")
    
    # Cross-octile analysis: do different octiles share many targets?
    print(f"\n  Target overlap matrix (fraction of N(Q_i) in N(Q_j)):")
    neighbor_sets = []
    for q in range(num_q):
        Q = quartiles[q]
        NH = set()
        for k in Q:
            NH.update(src_to_tgt[k])
        neighbor_sets.append(NH)
    
    # Print overlap as percentage
    header = "       " + "".join(f"  Q{q:d}   " for q in range(num_q))
    print(f"  {header}")
    for i in range(num_q):
        row = f"  Q{i:d} "
        for j in range(num_q):
            if len(neighbor_sets[i]) > 0:
                overlap = len(neighbor_sets[i] & neighbor_sets[j]) / len(neighbor_sets[i])
                row += f" {100*overlap:5.1f}%"
            else:
                row += "     -"
        print(row)
    
    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")

def main():
    print("=" * 110)
    print("Z31c: Stratified Hall Analysis - Octile Decomposition")
    print("=" * 110)
    
    test_ns = [1000, 5000, 10000, 50000, 100000]
    for n in test_ns:
        run_stratified(n)

if __name__ == '__main__':
    main()
