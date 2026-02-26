#!/usr/bin/env python3
"""
Z31: Degree structure of smooth bipartite subgraph for Erdős 710.

For each n, compute:
  B = floor(sqrt(n+L)),  L = (2/sqrt(e) + 0.05) * n * sqrt(ln n / ln ln n)
  S_plus = {k : k is B-smooth, B < k <= n/2}
  H_smooth = {h in (2n, n+L] : h is B-smooth}
  
  Edge: k in S_plus connected to h in H_smooth iff k | h

Outputs:
  1. |H_smooth| / |S_plus|
  2. min left degree (min over k in S_plus of deg_smooth(k))
  3. max right degree (max over h in H_smooth of #divisors in S_plus)
  4. Whether min_left_deg >= max_right_deg
"""

import math
import time
from collections import defaultdict

C_TARGET = 2.0 / math.sqrt(math.e) + 0.05  # ~1.2631

def compute_L(n):
    ln_n = math.log(n)
    ln_ln_n = math.log(ln_n)
    return C_TARGET * n * math.sqrt(ln_n / ln_ln_n)

def sieve_smooth(lo, hi, B):
    """
    Return set of B-smooth numbers in [lo, hi].
    Uses a sieve: start with residual = each number, divide out all primes <= B.
    A number is B-smooth iff its residual becomes 1.
    """
    if lo > hi:
        return set()
    
    size = hi - lo + 1
    residual = list(range(lo, hi + 1))  # residual[i] corresponds to number lo+i
    
    # Sieve out all primes <= B
    p = 2
    while p <= B:
        # Find first multiple of p >= lo
        start = ((lo + p - 1) // p) * p
        for idx in range(start - lo, size, p):
            while residual[idx] % p == 0:
                residual[idx] //= p
        # Next prime (simple increment)
        p += 1
        # Skip non-primes efficiently
        if p > 2:
            while p <= B:
                is_prime = True
                for d in range(2, int(math.sqrt(p)) + 1):
                    if p % d == 0:
                        is_prime = False
                        break
                if is_prime:
                    break
                p += 1
    
    result = set()
    for i in range(size):
        if residual[i] == 1:
            result.add(lo + i)
    return result

def sieve_smooth_fast(lo, hi, B):
    """
    Fast B-smooth sieve using actual prime sieve first.
    """
    if lo > hi:
        return set()
    
    # Generate primes up to B using sieve of Eratosthenes
    if B < 2:
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
    
    result = set()
    for i in range(size):
        if residual[i] == 1:
            result.add(lo + i)
    return result

def get_divisors_in_range(h, lo, hi):
    """Get all divisors of h that are in [lo, hi]."""
    divisors = []
    # Find all divisors up to sqrt(h)
    sq = int(math.sqrt(h))
    for d in range(1, sq + 1):
        if h % d == 0:
            if lo <= d <= hi:
                divisors.append(d)
            complement = h // d
            if complement != d and lo <= complement <= hi:
                divisors.append(complement)
    return divisors

def run_experiment(n):
    t0 = time.time()
    
    L = compute_L(n)
    L_int = int(math.floor(L))
    B = int(math.floor(math.sqrt(n + L_int)))
    half_n = n // 2
    
    # S_plus: B-smooth numbers in (B, n/2]
    S_plus = sieve_smooth_fast(B + 1, half_n, B)
    
    # H_smooth: B-smooth numbers in (2n, n + L]
    # Wait: the target interval is (2n, n+L] -- but that requires L > n, i.e., n+L > 2n.
    # Actually the standard setup: targets are in [n+1, n+L] that are multiples of sources.
    # Let me re-read the problem statement...
    # "h in (2n, n+L]" -- but 2n > n+L for small n. Let me check.
    # Actually I think the targets should be (n, n+L] and we need 2k > n, i.e., k > n/2.
    # No wait -- the problem says S_+ = {k : B < k <= n/2} and H_smooth in (2n, n+L].
    # But 2n > n+L when L < n. For n=1000, L ~ 1263*sqrt(6.9/1.93) ~ 1263*1.89 ~ 2387,
    # so n+L ~ 3387 and 2n = 2000. So 2n < n+L. Good.
    # The interval (2n, n+L] makes sense: these are targets h where h/k >= 2 for k <= n/2,
    # and h/k could be 2,3,4,... (the "top half" are in (n, 2n]).
    
    # Actually let me reconsider. In the Erdos 710 setup:
    # We have a set A of size f(n), and we need a subset of size n with all pairwise sums distinct.
    # The standard construction: A subset of [1, n+L] with the B-Sidon-like property.
    # 
    # Actually, for the bipartite graph in the proof:
    # Left vertices: S_+ = B-smooth numbers in (B, n/2]  (these are "sources")
    # Right vertices: targets that each source can "hit"
    # For source k, target h means that k*m = h for some multiplier m, with h in target range.
    #
    # In the Erdos 710 proof structure, the target range is typically [n+1, n+L].
    # Let me use the correct range from the proof outline.
    # 
    # From the memory: the bipartite graph G = ({1,...,floor(n/2)}, (2n, n+L])
    # So targets are in (2n, n+L]. For this to be non-empty we need L > n, which is true
    # when n is large enough.
    #
    # Hmm, but for n=1000: L ~ 2387, so n+L ~ 3387, 2n = 2000. (2n, n+L] = (2000, 3387]. OK.
    # For these targets h and sources k <= 500, the multiplier is h/k >= 2000/500 = 4.
    # 
    # Wait, the user says G = ({1,...,floor(n/2)}, (2n, n+L]).
    # So the FULL left side is {1,...,floor(n/2)}, but S_+ is the B-smooth subset.
    # And H_smooth is the B-smooth numbers in (2n, n+L].
    # Edge: k | h.
    
    target_lo = 2 * n + 1
    target_hi = n + L_int
    
    if target_lo > target_hi:
        print(f"  n={n}: target interval empty (2n+1={target_lo} > n+L={target_hi})")
        return None
    
    H_smooth = sieve_smooth_fast(target_lo, target_hi, B)
    
    S_plus_sorted = sorted(S_plus)
    H_smooth_sorted = sorted(H_smooth)
    
    size_S = len(S_plus_sorted)
    size_H = len(H_smooth_sorted)
    
    if size_S == 0 or size_H == 0:
        print(f"  n={n}: S_plus={size_S}, H_smooth={size_H} (one is empty)")
        return None
    
    # Compute left degrees: for each k in S_plus, count h in H_smooth with k | h
    # Efficient: for each k, iterate multiples of k in (target_lo, target_hi]
    left_deg = {}
    for k in S_plus_sorted:
        # Multiples of k in [target_lo, target_hi]
        m_lo = (target_lo + k - 1) // k  # ceiling(target_lo / k)
        m_hi = target_hi // k             # floor(target_hi / k)
        count = 0
        for m in range(m_lo, m_hi + 1):
            h = k * m
            if h in H_smooth:
                count += 1
        left_deg[k] = count
    
    # Compute right degrees: for each h in H_smooth, count divisors in S_plus
    # Efficient: enumerate divisors of h, check membership in S_plus set
    right_deg = {}
    S_plus_set = S_plus  # already a set
    for h in H_smooth_sorted:
        divs = get_divisors_in_range(h, B + 1, half_n)
        count = sum(1 for d in divs if d in S_plus_set)
        right_deg[h] = count
    
    min_left = min(left_deg.values())
    max_left = max(left_deg.values())
    avg_left = sum(left_deg.values()) / size_S
    
    min_right = min(right_deg.values())
    max_right = max(right_deg.values())
    avg_right = sum(right_deg.values()) / size_H
    
    # Count isolated vertices (degree 0)
    left_isolated = sum(1 for v in left_deg.values() if v == 0)
    right_isolated = sum(1 for v in right_deg.values() if v == 0)
    
    ratio = size_H / size_S if size_S > 0 else float('inf')
    hall_by_degree = min_left >= max_right
    
    elapsed = time.time() - t0
    
    return {
        'n': n,
        'B': B,
        'L': L_int,
        'size_S': size_S,
        'size_H': size_H,
        'ratio': ratio,
        'min_left': min_left,
        'max_left': max_left,
        'avg_left': avg_left,
        'min_right': min_right,
        'max_right': max_right,
        'avg_right': avg_right,
        'left_isolated': left_isolated,
        'right_isolated': right_isolated,
        'hall_by_degree': hall_by_degree,
        'elapsed': elapsed,
    }

def main():
    print("=" * 100)
    print("Z31: Degree Structure of Smooth Bipartite Subgraph")
    print("=" * 100)
    print()
    print(f"C_TARGET = 2/sqrt(e) + 0.05 = {C_TARGET:.4f}")
    print(f"L(n) = C_TARGET * n * sqrt(ln n / ln ln n)")
    print(f"B(n) = floor(sqrt(n + L))")
    print(f"S_+ = {{k : B-smooth, B < k <= n/2}}")
    print(f"H_smooth = {{h in (2n, n+L] : B-smooth}}")
    print(f"Edge: k | h")
    print()
    
    test_ns = [1000, 5000, 10000, 50000, 100000, 200000]
    results = []
    
    for n in test_ns:
        print(f"Computing n = {n}...", flush=True)
        r = run_experiment(n)
        if r:
            results.append(r)
            print(f"  Done in {r['elapsed']:.1f}s: |S_+|={r['size_S']}, |H_smooth|={r['size_H']}")
    
    print()
    print("=" * 100)
    print("RESULTS TABLE")
    print("=" * 100)
    print()
    
    # Table 1: Basic sizes
    print("Table 1: Set Sizes and Ratio")
    print("-" * 85)
    print(f"{'n':>8s}  {'B':>6s}  {'L':>8s}  {'|S_+|':>8s}  {'|H_sm|':>8s}  "
          f"{'|H|/|S|':>8s}  {'L_iso':>6s}  {'R_iso':>6s}")
    print("-" * 85)
    for r in results:
        print(f"{r['n']:>8d}  {r['B']:>6d}  {r['L']:>8d}  {r['size_S']:>8d}  {r['size_H']:>8d}  "
              f"{r['ratio']:>8.3f}  {r['left_isolated']:>6d}  {r['right_isolated']:>6d}")
    print()
    
    # Table 2: Degree statistics
    print("Table 2: Degree Statistics")
    print("-" * 100)
    print(f"{'n':>8s}  {'min_L':>7s}  {'avg_L':>8s}  {'max_L':>7s}  "
          f"{'min_R':>7s}  {'avg_R':>8s}  {'max_R':>7s}  "
          f"{'minL>=maxR':>10s}  {'gap':>8s}")
    print("-" * 100)
    for r in results:
        gap = r['min_left'] - r['max_right']
        hall_str = "YES" if r['hall_by_degree'] else "NO"
        print(f"{r['n']:>8d}  {r['min_left']:>7d}  {r['avg_left']:>8.2f}  {r['max_left']:>7d}  "
              f"{r['min_right']:>7d}  {r['avg_right']:>8.2f}  {r['max_right']:>7d}  "
              f"{hall_str:>10s}  {gap:>8d}")
    print()
    
    # Table 3: Ratios for analysis
    print("Table 3: Key Ratios")
    print("-" * 80)
    print(f"{'n':>8s}  {'min_L/max_R':>12s}  {'avg_L/avg_R':>12s}  {'min_L':>7s}  {'max_R':>7s}  {'delta':>8s}")
    print("-" * 80)
    for r in results:
        ln_n = math.log(r['n'])
        ln_ln_n = math.log(ln_n)
        delta = 2 * (r['n'] + r['L']) / r['n'] - 1  # approximate
        ratio_lr = r['min_left'] / r['max_right'] if r['max_right'] > 0 else float('inf')
        ratio_avg = r['avg_left'] / r['avg_right'] if r['avg_right'] > 0 else float('inf')
        print(f"{r['n']:>8d}  {ratio_lr:>12.4f}  {ratio_avg:>12.4f}  {r['min_left']:>7d}  "
              f"{r['max_right']:>7d}  {delta:>8.3f}")
    print()
    
    # Detailed degree distributions for smallest case
    if results:
        r0 = results[0]
        n = r0['n']
        print(f"Detailed degree distribution for n={n}:")
        # Re-run to get distributions
        L = compute_L(n)
        L_int = int(math.floor(L))
        B = int(math.floor(math.sqrt(n + L_int)))
        half_n = n // 2
        
        S_plus = sieve_smooth_fast(B + 1, half_n, B)
        H_smooth = sieve_smooth_fast(2 * n + 1, n + L_int, B)
        S_plus_set = S_plus
        
        # Left degree distribution
        left_degs = []
        for k in sorted(S_plus):
            target_lo = 2 * n + 1
            target_hi = n + L_int
            m_lo = (target_lo + k - 1) // k
            m_hi = target_hi // k
            count = 0
            for m in range(m_lo, m_hi + 1):
                h = k * m
                if h in H_smooth:
                    count += 1
            left_degs.append((k, count))
        
        # Right degree distribution
        right_degs = []
        for h in sorted(H_smooth):
            divs = get_divisors_in_range(h, B + 1, half_n)
            count = sum(1 for d in divs if d in S_plus_set)
            right_degs.append((h, count))
        
        # Print left deg histogram
        from collections import Counter
        left_hist = Counter(d for _, d in left_degs)
        print(f"  Left degree histogram (|S_+|={len(left_degs)}):")
        for deg in sorted(left_hist.keys()):
            print(f"    deg={deg}: {left_hist[deg]} vertices")
        
        right_hist = Counter(d for _, d in right_degs)
        print(f"  Right degree histogram (|H_smooth|={len(right_degs)}):")
        for deg in sorted(right_hist.keys()):
            print(f"    deg={deg}: {right_hist[deg]} vertices")
        
        # Show the sources with min degree
        min_d = min(d for _, d in left_degs)
        print(f"\n  Sources with minimum degree {min_d}:")
        for k, d in left_degs:
            if d == min_d:
                print(f"    k={k}")
                if sum(1 for k2, d2 in left_degs if d2 == min_d) > 10:
                    print(f"    ... (showing first few)")
                    break
        
        # Show the targets with max degree
        max_d = max(d for _, d in right_degs)
        print(f"\n  Targets with maximum degree {max_d}:")
        count_shown = 0
        for h, d in right_degs:
            if d == max_d:
                # Show factorization
                hh = h
                factors = []
                pp = 2
                while pp * pp <= hh:
                    while hh % pp == 0:
                        factors.append(pp)
                        hh //= pp
                    pp += 1
                if hh > 1:
                    factors.append(hh)
                print(f"    h={h} = {'*'.join(map(str, factors))}")
                count_shown += 1
                if count_shown >= 10:
                    print(f"    ... (showing first 10)")
                    break

    print()
    print("=" * 100)
    print("ANALYSIS")
    print("=" * 100)
    if results:
        # Check trend
        print()
        print("Key question: Does min_left_deg / max_right_deg -> infinity?")
        for r in results:
            ratio_lr = r['min_left'] / r['max_right'] if r['max_right'] > 0 else float('inf')
            trend = "HALL (degree)" if ratio_lr >= 1 else "no Hall by degree"
            print(f"  n={r['n']:>6d}: min_L/max_R = {ratio_lr:.4f}  [{trend}]")
        
        print()
        print("If min_left >= max_right, then Hall's condition holds because:")
        print("  For any T subset S_+, |NH(T)| >= |T| follows from a simple")
        print("  double-counting argument: each source has >= min_L neighbors,")
        print("  each target has <= max_R sources, so |NH(T)| >= |T|*min_L/max_R >= |T|.")


if __name__ == '__main__':
    main()
