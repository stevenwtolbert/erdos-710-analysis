#!/usr/bin/env python3
"""
Z31f: Fine partition analysis.

The dyadic partition (factor-of-2 intervals) fails for 2-3 middle intervals.
Question: Does a finer partition (factor of sqrt(2), or factor of 1.5) fix ALL intervals?

Also: what is the critical quantity min_L_j / max_R_j as a function of the 
interval width ratio? As the interval narrows, max_R_j drops (fewer sources in 
a narrow interval can share a target), but min_L_j also drops (sources in a 
narrow interval have fewer multiples).

The sweet spot: narrow enough that max_R_j is small, wide enough that min_L_j 
is still >= max_R_j.
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

def analyze_partition(n, ratio_name, ratio_val):
    """Partition S_+ into intervals [a, a*ratio) and check degree condition."""
    L = compute_L(n)
    L_int = int(math.floor(L))
    B = int(math.floor(math.sqrt(n + L_int)))
    half_n = n // 2
    
    S_plus = sieve_smooth_fast(B + 1, half_n, B)
    target_lo = 2 * n + 1
    target_hi = n + L_int
    H_smooth = sieve_smooth_fast(target_lo, target_hi, B)
    
    S_plus_sorted = sorted(S_plus)
    
    # Build adjacency
    src_to_tgt = defaultdict(list)
    tgt_to_src = defaultdict(list)
    for k in S_plus_sorted:
        m_lo = (target_lo + k - 1) // k
        m_hi = target_hi // k
        for m in range(m_lo, m_hi + 1):
            h = k * m
            if h in H_smooth:
                src_to_tgt[k].append(h)
                tgt_to_src[h].append(k)
    
    # Create intervals
    intervals = []
    lo = B + 1
    while lo <= half_n:
        hi = min(int(lo * ratio_val), half_n)
        if hi == lo:
            hi = lo  # at least one element wide
        intervals.append((lo, hi))
        lo = hi + 1
    
    # Analyze each interval
    all_pass = True
    worst_ratio = float('inf')
    n_intervals = 0
    n_nonempty = 0
    n_failing = 0
    
    for (lo_j, hi_j) in intervals:
        S_j = [k for k in S_plus_sorted if lo_j <= k <= hi_j]
        if not S_j:
            continue
        n_nonempty += 1
        S_j_set = set(S_j)
        
        left_degs = [len(src_to_tgt[k]) for k in S_j]
        if not left_degs:
            continue
        min_l = min(left_degs)
        
        NH_j = set()
        for k in S_j:
            NH_j.update(src_to_tgt[k])
        
        max_r = 0
        for h in NH_j:
            rd = sum(1 for k in tgt_to_src[h] if k in S_j_set)
            max_r = max(max_r, rd)
        
        ratio = min_l / max_r if max_r > 0 else float('inf')
        if ratio < 1:
            all_pass = False
            n_failing += 1
        worst_ratio = min(worst_ratio, ratio)
    
    return all_pass, worst_ratio, n_nonempty, n_failing

def main():
    print("=" * 90)
    print("Z31f: Fine Partition Analysis - Finding Optimal Interval Width")
    print("=" * 90)
    
    test_ns = [1000, 5000, 10000, 50000, 100000, 200000]
    
    # Test different partition ratios
    partition_ratios = [
        ("2.0 (dyadic)", 2.0),
        ("sqrt(2)~1.41", math.sqrt(2)),
        ("1.3", 1.3),
        ("1.2", 1.2),
        ("1.15", 1.15),
        ("1.1", 1.1),
        ("1.05", 1.05),
    ]
    
    print(f"\n{'':>10s}", end="")
    for name, _ in partition_ratios:
        print(f"  {name:>16s}", end="")
    print()
    print("-" * (10 + 18 * len(partition_ratios)))
    
    for n in test_ns:
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
        
        # Build adjacency once
        src_to_tgt = defaultdict(list)
        tgt_to_src = defaultdict(list)
        for k in S_plus_sorted:
            m_lo = (target_lo + k - 1) // k
            m_hi = target_hi // k
            for m in range(m_lo, m_hi + 1):
                h = k * m
                if h in H_smooth:
                    src_to_tgt[k].append(h)
                    tgt_to_src[h].append(k)
        
        row = f"n={n:>6d}"
        for name, ratio_val in partition_ratios:
            # Create intervals
            intervals = []
            lo = B + 1
            while lo <= half_n:
                hi = min(int(lo * ratio_val), half_n)
                if hi < lo:
                    hi = lo
                intervals.append((lo, hi))
                lo = hi + 1
            
            all_pass = True
            worst_r = float('inf')
            n_fail = 0
            n_nonempty = 0
            
            for (lo_j, hi_j) in intervals:
                S_j = [k for k in S_plus_sorted if lo_j <= k <= hi_j]
                if not S_j:
                    continue
                n_nonempty += 1
                S_j_set = set(S_j)
                
                left_degs = [len(src_to_tgt[k]) for k in S_j]
                min_l = min(left_degs)
                
                NH_j = set()
                for k in S_j:
                    NH_j.update(src_to_tgt[k])
                
                max_r = 0
                for h in NH_j:
                    rd = sum(1 for k in tgt_to_src[h] if k in S_j_set)
                    max_r = max(max_r, rd)
                
                ratio = min_l / max_r if max_r > 0 else float('inf')
                if ratio < 1:
                    all_pass = False
                    n_fail += 1
                worst_r = min(worst_r, ratio)
            
            status = "ALL PASS" if all_pass else f"{n_fail} FAIL"
            row += f"  {worst_r:>5.2f}({status:>8s})"
        
        elapsed = time.time() - t0
        print(f"{row}  [{elapsed:.1f}s]")
    
    # Detailed analysis of the FAILING intervals for ratio=1.1 at n=200000
    print(f"\n{'='*90}")
    print("Detailed failing intervals for ratio=1.1 at various n:")
    print(f"{'='*90}")
    
    for n in [50000, 100000, 200000]:
        L = compute_L(n)
        L_int = int(math.floor(L))
        B = int(math.floor(math.sqrt(n + L_int)))
        half_n = n // 2
        
        S_plus = sieve_smooth_fast(B + 1, half_n, B)
        target_lo = 2 * n + 1
        target_hi = n + L_int
        H_smooth = sieve_smooth_fast(target_lo, target_hi, B)
        
        S_plus_sorted = sorted(S_plus)
        
        src_to_tgt = defaultdict(list)
        tgt_to_src = defaultdict(list)
        for k in S_plus_sorted:
            m_lo = (target_lo + k - 1) // k
            m_hi = target_hi // k
            for m in range(m_lo, m_hi + 1):
                h = k * m
                if h in H_smooth:
                    src_to_tgt[k].append(h)
                    tgt_to_src[h].append(k)
        
        ratio_val = 1.1
        intervals = []
        lo = B + 1
        while lo <= half_n:
            hi = min(int(lo * ratio_val), half_n)
            if hi < lo:
                hi = lo
            intervals.append((lo, hi))
            lo = hi + 1
        
        print(f"\nn = {n}:")
        for (lo_j, hi_j) in intervals:
            S_j = [k for k in S_plus_sorted if lo_j <= k <= hi_j]
            if not S_j:
                continue
            S_j_set = set(S_j)
            
            left_degs = [len(src_to_tgt[k]) for k in S_j]
            min_l = min(left_degs)
            avg_l = sum(left_degs) / len(left_degs)
            
            NH_j = set()
            for k in S_j:
                NH_j.update(src_to_tgt[k])
            
            max_r = 0
            for h in NH_j:
                rd = sum(1 for k in tgt_to_src[h] if k in S_j_set)
                max_r = max(max_r, rd)
            
            ratio = min_l / max_r if max_r > 0 else float('inf')
            if ratio < 1.5:  # Show any borderline interval
                nhall = len(NH_j) / len(S_j)
                print(f"  [{lo_j:>7d},{hi_j:>7d}] |S|={len(S_j):>5d} minL={min_l:>3d} avgL={avg_l:>6.1f} "
                      f"maxR={max_r:>3d} ratio={ratio:>5.2f} |NH|/|S|={nhall:.2f} "
                      f"{'FAIL' if ratio < 1 else 'pass'}")

if __name__ == '__main__':
    main()
