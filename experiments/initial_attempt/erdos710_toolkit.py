#!/usr/bin/env python3
"""
Erdős 710 — Computational Toolkit for Gap Closure
Focused on: large-prime stratification, violator analysis, augmenting paths.

Usage: python erdos710_toolkit.py [--full]
  Default: quick mode (n ≤ 1000)
  --full: extended mode (n ≤ 5000, slower)
"""

import sys, time, math, random
from math import gcd, log, sqrt, exp, floor, ceil
from collections import defaultdict, Counter
from itertools import combinations

random.seed(42)

# ═══════════════════════════════════════════════════════════════
# CORE ARITHMETIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def largest_prime_factor(k, _cache={}):
    """P⁺(k): largest prime factor of k."""
    if k in _cache: return _cache[k]
    if k <= 1: _cache[k] = 0; return 0
    r = 0; t = k; d = 2
    while d * d <= t:
        while t % d == 0: r = max(r, d); t //= d
        d += 1
    if t > 1: r = max(r, t)
    _cache[k] = r; return r

def smallest_prime_factor(k):
    if k <= 1: return 0
    if k % 2 == 0: return 2
    d = 3
    while d * d <= k:
        if k % d == 0: return d
        d += 2
    return k

def factorize(k):
    """Return list of (prime, exponent) pairs."""
    if k <= 1: return []
    factors = []; d = 2
    while d * d <= k:
        e = 0
        while k % d == 0: e += 1; k //= d
        if e > 0: factors.append((d, e))
        d += 1
    if k > 1: factors.append((k, 1))
    return factors

def omega_big(k):
    """Ω(k): number of prime factors with multiplicity."""
    return sum(e for _, e in factorize(k))

def smooth_part(k, B):
    """B-smooth part of k: product of prime powers p^a dividing k with p ≤ B."""
    s = 1
    for p, e in factorize(k):
        if p <= B: s *= p ** e
    return s

def is_smooth(k, B):
    return largest_prime_factor(k) <= B

def targets(k, n, L):
    """Multiples of k in (n, n+L]."""
    j0 = n // k + 1
    j1 = (n + L) // k
    return set(k * j for j in range(j0, j1 + 1))

def divisors_up_to(m, bound):
    """All divisors of m that are ≤ bound."""
    divs = []
    d = 1
    while d * d <= m:
        if m % d == 0:
            if d <= bound: divs.append(d)
            if m // d <= bound and m // d != d: divs.append(m // d)
        d += 1
    return divs

# ═══════════════════════════════════════════════════════════════
# MATCHING (Hopcroft-Karp)
# ═══════════════════════════════════════════════════════════════

def hopcroft_karp(srcs, adj):
    ml = {}; mr = {}
    def bfs():
        q = []; d = {}
        for s in srcs:
            if s not in ml: d[s] = 0; q.append(s)
            else: d[s] = float('inf')
        found = False; h = 0
        while h < len(q):
            s = q[h]; h += 1
            for t in adj.get(s, []):
                s2 = mr.get(t)
                if s2 is None: found = True
                elif d.get(s2, float('inf')) == float('inf'):
                    d[s2] = d[s] + 1; q.append(s2)
        return found, d
    def dfs(s, d):
        for t in adj.get(s, []):
            s2 = mr.get(t)
            if s2 is None or (d.get(s2, float('inf')) == d[s] + 1 and dfs(s2, d)):
                ml[s] = t; mr[t] = s; return True
        d[s] = float('inf'); return False
    while True:
        found, d = bfs()
        if not found: break
        for s in list(srcs):
            if s not in ml: dfs(s, d)
    return ml, mr

def compute_fn(n):
    """Compute f(n) exactly by binary search + Hopcroft-Karp."""
    def has_matching(n, L):
        srcs = list(range(1, n + 1))
        adj = {k: list(targets(k, n, L)) for k in srcs}
        ml, _ = hopcroft_karp(srcs, adj)
        return len(ml) == n
    lo, hi = n, 4 * n
    while lo < hi:
        mid = (lo + hi) // 2
        if has_matching(n, mid): hi = mid
        else: lo = mid + 1
    return lo

# ═══════════════════════════════════════════════════════════════
# ASYMPTOTIC PARAMETERS
# ═══════════════════════════════════════════════════════════════

C_TARGET = 2 / sqrt(exp(1))  # ≈ 1.2131

def get_params(n):
    """Return (u, c, B, A_set, R_set) for asymptotic analysis."""
    u = sqrt(exp(1) * log(n) / log(log(n)))
    c = exp(-1 / (2 * u))
    B = int(n ** (1 / u))
    cn = int(c * n)
    A = sorted([k for k in range(cn, n + 1) if is_smooth(k, B)])
    R = sorted([k for k in range(1, n + 1) if k not in set(A)])
    return u, c, B, A, R

def target_L(n, eps=0.05):
    """L = (2/√e + ε) · n · √(ln n / ln ln n)"""
    return int((C_TARGET + eps) * n * sqrt(log(n) / log(log(n))))

# ═══════════════════════════════════════════════════════════════
# TEST 1: LARGE PRIME STRATIFICATION
# ═══════════════════════════════════════════════════════════════

def test_large_prime_stratification(n, eps=0.05):
    """
    Split R into R_big (P⁺(k) > √n) and R_small_p (B < P⁺(k) ≤ √n).
    Check exclusivity and Hall properties for each.
    """
    u, c, B, A, R = get_params(n)
    L = target_L(n, eps)
    A_set = set(A)
    sqn = sqrt(n)

    # Compute N(A)
    NA = set()
    for k in A:
        NA |= targets(k, n, L)

    # Partition R
    R_big = [k for k in R if largest_prime_factor(k) > sqn]
    R_small_p = [k for k in R if B < largest_prime_factor(k) <= sqn]
    R_very_small = [k for k in R if largest_prime_factor(k) <= B and k not in A_set]
    # R_very_small are smooth but outside [cn, n]

    # Check exclusivity for R_big
    big_exclusive_count = 0
    big_total_targets = 0
    big_exclusive_targets = 0
    for k in R_big:
        tgts = targets(k, n, L)
        excl = tgts - NA
        big_total_targets += len(tgts)
        big_exclusive_targets += len(excl)
        if excl == tgts:
            big_exclusive_count += 1

    # Check Hall for R_big in exclusive graph
    R_big_adj = {}
    for k in R_big:
        excl = list(targets(k, n, L) - NA)
        if excl:
            R_big_adj[k] = excl

    if R_big:
        ml_big, _ = hopcroft_karp(R_big, R_big_adj)
        big_matched = len(ml_big)
    else:
        big_matched = 0

    # Check Hall for R_small_p in exclusive graph (minus targets used by R_big)
    big_used = set(ml_big.values()) if R_big else set()
    R_sp_adj = {}
    for k in R_small_p:
        excl = list((targets(k, n, L) - NA) - big_used)
        if excl:
            R_sp_adj[k] = excl

    if R_small_p:
        ml_sp, _ = hopcroft_karp(R_small_p, R_sp_adj)
        sp_matched = len(ml_sp)
    else:
        sp_matched = 0

    # Also handle R_very_small
    sp_used = set(ml_sp.values()) if R_small_p else set()
    R_vs_adj = {}
    for k in R_very_small:
        excl = list((targets(k, n, L) - NA) - big_used - sp_used)
        if excl:
            R_vs_adj[k] = excl

    if R_very_small:
        ml_vs, _ = hopcroft_karp(R_very_small, R_vs_adj)
        vs_matched = len(ml_vs)
    else:
        vs_matched = 0

    # JOINT matching: all of R into Excl simultaneously
    Excl = set(range(n + 1, n + L + 1)) - NA
    R_all_adj = {}
    for k in R:
        excl_tgts = list(targets(k, n, L) & Excl)
        if excl_tgts:
            R_all_adj[k] = excl_tgts
    ml_all, _ = hopcroft_karp(R, R_all_adj)
    joint_matched = len(ml_all)

    # Min exclusive degree
    min_excl_deg = min((len(R_all_adj.get(k, [])) for k in R), default=0)
    zero_deg = sum(1 for k in R if len(R_all_adj.get(k, [])) == 0)

    return {
        'n': n, 'L': L, '|A|': len(A), '|R|': len(R),
        '|R_big|': len(R_big), '|R_small_p|': len(R_small_p),
        '|R_very_small|': len(R_very_small),
        'R_big_fully_exclusive': big_exclusive_count,
        'R_big_exclusive_frac': big_exclusive_targets / max(1, big_total_targets),
        'R_big_Hall': big_matched == len(R_big),
        'R_sp_Hall': sp_matched == len(R_small_p),
        'R_vs_Hall': vs_matched == len(R_very_small),
        'seq_matched': big_matched + sp_matched + vs_matched,
        'joint_matched': joint_matched,
        'joint_Hall': joint_matched == len(R),
        'min_excl_deg': min_excl_deg,
        'zero_excl_deg': zero_deg,
    }

# ═══════════════════════════════════════════════════════════════
# TEST 2: HYPOTHETICAL VIOLATOR ANALYSIS
# ═══════════════════════════════════════════════════════════════

def test_violator_structure(n, eps=0.05):
    """
    Find the tightest subsets of V = {1,...,n/2} mapping into H = (2n, n+L].
    Analyze their arithmetic structure.
    """
    L = target_L(n, eps)
    M = L - n  # |H|
    V = list(range(1, n // 2 + 1))

    # Build adjacency for V → H
    adj = {}
    for k in V:
        tgts = []
        j0 = (2 * n) // k + 1
        j1 = (n + L) // k
        for j in range(j0, j1 + 1):
            tgts.append(k * j)
        adj[k] = tgts

    # Find tightest singleton (worst ratio)
    worst_ratio = float('inf')
    worst_k = None
    for k in V:
        r = len(adj.get(k, []))
        if r > 0 and r < worst_ratio:
            worst_ratio = r
            worst_k = k

    # Find tightest pairs
    worst_pair_ratio = float('inf')
    worst_pair = None
    sample_pairs = random.sample(list(combinations(V[-50:], 2)), min(500, len(V[-50:]) * (len(V[-50:]) - 1) // 2)) if len(V) >= 50 else list(combinations(V[-20:], 2))
    for k1, k2 in sample_pairs:
        NS = set(adj.get(k1, [])) | set(adj.get(k2, []))
        r = len(NS) / 2
        if r < worst_pair_ratio:
            worst_pair_ratio = r
            worst_pair = (k1, k2)

    # Analyze GCD structure of top elements
    top = V[-min(30, len(V)):]
    gcd_matrix = {}
    for i, k1 in enumerate(top):
        for j, k2 in enumerate(top):
            if i < j:
                g = gcd(k1, k2)
                gcd_matrix[(k1, k2)] = g

    avg_gcd = sum(gcd_matrix.values()) / max(1, len(gcd_matrix))
    max_gcd = max(gcd_matrix.values()) if gcd_matrix else 0
    frac_coprime = sum(1 for g in gcd_matrix.values() if g == 1) / max(1, len(gcd_matrix))

    # For top elements, compute overlap: |N({k1}) ∩ N({k2})| / min(|N(k1)|, |N(k2)|)
    overlaps = []
    for (k1, k2), g in gcd_matrix.items():
        n1 = set(adj.get(k1, []))
        n2 = set(adj.get(k2, []))
        if n1 and n2:
            overlaps.append(len(n1 & n2) / min(len(n1), len(n2)))

    return {
        'n': n, 'M': M, '|V|': len(V),
        'worst_singleton_k': worst_k,
        'worst_singleton_deg': worst_ratio,
        'worst_pair': worst_pair,
        'worst_pair_ratio': worst_pair_ratio,
        'top_avg_gcd': avg_gcd,
        'top_max_gcd': max_gcd,
        'top_frac_coprime': frac_coprime,
        'top_avg_overlap': sum(overlaps) / max(1, len(overlaps)) if overlaps else 0,
        'top_max_overlap': max(overlaps) if overlaps else 0,
    }

# ═══════════════════════════════════════════════════════════════
# TEST 3: FIRST-TARGET COLLISION ANALYSIS
# ═══════════════════════════════════════════════════════════════

def test_first_target_collisions(n, eps=0.05):
    """
    Analyze the collision structure of the first-target map k ↦ m_k.
    """
    u, c, B, A, R = get_params(n)
    L = target_L(n, eps)
    A_set = set(A)
    NA = set()
    for k in A:
        NA |= targets(k, n, L)

    # First target: smallest multiple of k exceeding n
    first_target = {}
    for k in R:
        m_k = k * ceil((n + 1) / k)
        if m_k <= n + L:
            first_target[k] = m_k

    # Check exclusivity of first targets
    ft_exclusive = sum(1 for k, m in first_target.items() if m not in NA)
    ft_total = len(first_target)

    # Count collisions
    target_to_sources = defaultdict(list)
    for k, m in first_target.items():
        target_to_sources[m].append(k)

    collision_counts = Counter(len(v) for v in target_to_sources.values())
    max_collision = max(collision_counts.keys()) if collision_counts else 0
    unique_targets = sum(1 for v in target_to_sources.values() if len(v) == 1)

    # For colliding elements, check if they have alternative exclusive targets
    colliding_k = [k for m, ks in target_to_sources.items() if len(ks) > 1 for k in ks]
    alt_available = 0
    for k in colliding_k:
        excl = targets(k, n, L) - NA - {first_target.get(k, -1)}
        if excl:
            alt_available += 1

    return {
        'n': n, '|R|': len(R),
        'ft_total': ft_total,
        'ft_exclusive': ft_exclusive,
        'ft_exclusive_frac': ft_exclusive / max(1, ft_total),
        'unique_targets': unique_targets,
        'collision_dist': dict(collision_counts),
        'max_collision': max_collision,
        'colliding_elements': len(colliding_k),
        'colliding_with_alternatives': alt_available,
        'alt_frac': alt_available / max(1, len(colliding_k)),
    }

# ═══════════════════════════════════════════════════════════════
# TEST 4: INTERVAL HALL WITH SURPLUS
# ═══════════════════════════════════════════════════════════════

def test_interval_surplus(n, eps=0.05):
    """Check Hall surplus for the full smooth interval A."""
    u, c, B, A, R = get_params(n)
    L = target_L(n, eps)
    NA = set()
    for k in A:
        NA |= targets(k, n, L)
    Excl = set(range(n + 1, n + L + 1)) - NA

    return {
        'n': n, 'u': u, 'c': c, 'B': B, '|A|': len(A),
        '|N(A)|': len(NA), 'surplus': len(NA) - len(A),
        'surplus_pct': (len(NA) - len(A)) / max(1, len(A)) * 100,
        '|Excl|': len(Excl), 'L': L,
        'excl_pct': len(Excl) / L * 100,
    }

# ═══════════════════════════════════════════════════════════════
# TEST 5: FULL MATCHING VERIFICATION
# ═══════════════════════════════════════════════════════════════

def test_full_matching(n, eps=0.05):
    """Verify full matching exists at target L."""
    L = target_L(n, eps)
    srcs = list(range(1, n + 1))
    adj = {k: list(targets(k, n, L)) for k in srcs}
    ml, mr = hopcroft_karp(srcs, adj)
    return {
        'n': n, 'eps': eps, 'L': L,
        'matched': len(ml), 'target': n,
        'success': len(ml) == n,
    }

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    full_mode = '--full' in sys.argv
    t0 = time.time()

    print("=" * 75)
    print("  ERDŐS 710 — GAP CLOSURE TOOLKIT")
    print("=" * 75)

    # ─── Test 1: Large Prime Stratification ───
    print("\n▸ TEST 1: Large Prime Stratification")
    print(f"  {'n':>6} {'|R_big|':>7} {'|R_sp|':>6} {'|R_vs|':>6} "
          f"{'big_excl%':>9} {'joint':>6} {'0-deg':>5} {'min_d':>5}")

    ns1 = [200, 500, 1000] + ([2000, 3000] if full_mode else [])
    for n in ns1:
        r = test_large_prime_stratification(n)
        print(f"  {r['n']:>6} {r['|R_big|']:>7} {r['|R_small_p|']:>6} {r['|R_very_small|']:>6} "
              f"{r['R_big_exclusive_frac']*100:>8.1f}% "
              f"{'✓' if r['joint_Hall'] else '✗':>6} "
              f"{r['zero_excl_deg']:>5} "
              f"{r['min_excl_deg']:>5}")

    # ─── Test 2: Violator Structure ───
    print("\n▸ TEST 2: Hypothetical Violator Structure (top elements of V)")
    print(f"  {'n':>6} {'worst_k':>7} {'deg':>4} {'pair_ratio':>10} "
          f"{'avg_gcd':>8} {'coprime%':>8} {'avg_ovlp':>9}")

    ns2 = [200, 500, 1000] + ([2000] if full_mode else [])
    for n in ns2:
        r = test_violator_structure(n)
        print(f"  {r['n']:>6} {r['worst_singleton_k']:>7} {r['worst_singleton_deg']:>4} "
              f"{r['worst_pair_ratio']:>10.3f} "
              f"{r['top_avg_gcd']:>8.2f} {r['top_frac_coprime']*100:>7.1f}% "
              f"{r['top_avg_overlap']:>9.4f}")

    # ─── Test 3: First-Target Collisions ───
    print("\n▸ TEST 3: First-Target Collision Analysis")
    print(f"  {'n':>6} {'|R|':>5} {'excl%':>6} {'unique':>6} {'max_col':>7} "
          f"{'colliding':>9} {'alt_avail%':>10}")

    ns3 = [200, 500, 1000] + ([2000] if full_mode else [])
    for n in ns3:
        r = test_first_target_collisions(n)
        print(f"  {r['n']:>6} {r['|R|']:>5} {r['ft_exclusive_frac']*100:>5.1f}% "
              f"{r['unique_targets']:>6} {r['max_collision']:>7} "
              f"{r['colliding_elements']:>9} {r['alt_frac']*100:>9.1f}%")

    # ─── Test 4: Interval Surplus ───
    print("\n▸ TEST 4: Smooth Interval Surplus")
    print(f"  {'n':>6} {'u':>5} {'|A|':>5} {'|N(A)|':>7} {'surplus':>8} {'surplus%':>8} {'excl%':>6}")

    ns4 = [200, 500, 1000] + ([2000, 3000] if full_mode else [])
    for n in ns4:
        r = test_interval_surplus(n)
        print(f"  {r['n']:>6} {r['u']:>5.2f} {r['|A|']:>5} {r['|N(A)|']:>7} "
              f"{r['surplus']:>+8} {r['surplus_pct']:>7.1f}% {r['excl_pct']:>5.1f}%")

    # ─── Test 5: Full Matching ───
    print("\n▸ TEST 5: Full Matching Verification")
    ns5 = [200, 500, 1000] + ([1500] if full_mode else [])
    for n in ns5:
        for eps in [0.05, 0.10]:
            r = test_full_matching(n, eps)
            status = "✓" if r['success'] else "✗"
            print(f"  n={n}, ε={eps:.2f}: L={r['L']}, matched={r['matched']}/{r['target']} {status}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")

if __name__ == '__main__':
    main()
