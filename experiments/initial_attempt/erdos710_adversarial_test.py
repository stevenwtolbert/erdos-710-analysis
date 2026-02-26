#!/usr/bin/env python3
"""
Erdos 710 -- Adversarial testing of Shearer and Cauchy-Schwarz bounds.

Tests whether the Shearer bound (prod |N_k|)^{1/Delta} >= s can fail
for adversarially chosen Case B subsets (e.g., divisors of highly composite
targets in H).

Also tests the Cauchy-Schwarz bound:
  |N_H(S)| >= (sum |N_k|)^2 / sum_{k1,k2} |N_{k1} cap N_{k2}|

Usage: python erdos710_adversarial_test.py
"""

import sys, math, time
from math import gcd, log, floor, ceil
from collections import defaultdict
from itertools import combinations

from erdos710_toolkit import target_L, C_TARGET, divisors_up_to, factorize

def build_graph(n, L):
    """Build V -> H bipartite graph. V = {1,...,n//2}, H = (2n, n+L]."""
    V = list(range(1, n // 2 + 1))
    adj = {}
    for k in V:
        j0 = (2 * n) // k + 1
        j1 = (n + L) // k
        adj[k] = set(k * j for j in range(j0, j1 + 1))
    return V, adj

def lcm(a, b):
    return a * b // gcd(a, b)

def shearer_bound(S, adj):
    """Compute Shearer bound (prod |N_k|)^{1/Delta} and actual |N_H(S)|."""
    NH = set()
    tau = defaultdict(int)
    for k in S:
        for m in adj.get(k, set()):
            tau[m] += 1
            NH.add(m)

    if not NH:
        return 0, 0, 0, len(NH)

    Delta = max(tau.values())
    log_sum = sum(log(len(adj[k])) for k in S if len(adj[k]) > 0)

    # Handle zero-degree vertices
    if any(len(adj.get(k, set())) == 0 for k in S):
        return 0, Delta, 0, len(NH)

    shearer = math.exp(log_sum / Delta) if Delta > 0 else 0
    return shearer, Delta, log_sum, len(NH)

def cauchy_schwarz_bound(S, adj, M):
    """Compute Cauchy-Schwarz bound (sum |N_k|)^2 / sum tau_S(m)^2."""
    NH = set()
    tau = defaultdict(int)
    for k in S:
        for m in adj.get(k, set()):
            tau[m] += 1
            NH.add(m)

    if not NH:
        return 0, len(NH)

    sum_deg = sum(len(adj.get(k, set())) for k in S)
    sum_tau2 = sum(t * t for t in tau.values())

    cs_bound = sum_deg * sum_deg / sum_tau2 if sum_tau2 > 0 else 0
    return cs_bound, len(NH)

def find_hc_targets(n, L):
    """Find the most highly composite targets in H = (2n, n+L]."""
    H_start = 2 * n + 1
    H_end = n + L

    best = []
    for m in range(H_start, H_end + 1):
        # Count divisors of m in V = {1,...,n//2}
        divs = divisors_up_to(m, n // 2)
        tau_v = len(divs)
        if tau_v >= 10:
            best.append((tau_v, m, divs))

    best.sort(reverse=True)
    return best[:20]

def adversarial_subset(m, divs, M, s, n):
    """Build adversarial Case B subset from divisors of m.

    Pick divisors of m in (M/(s+1), n/2] as many as possible,
    then fill remaining slots with elements near n/2.
    """
    threshold = M / (s + 1)
    # Divisors of m in the Case B range
    case_b_divs = sorted([d for d in divs if threshold < d <= n // 2])

    S = list(case_b_divs[:s])

    # Fill remaining with non-divisor elements near n/2
    used = set(S)
    k = n // 2
    while len(S) < s and k > threshold:
        if k not in used:
            S.append(k)
        k -= 1

    S.sort()
    return S

def test_adversarial(n, eps=0.05):
    """Test Shearer and CS bounds against adversarial Case B subsets."""
    L = target_L(n, eps)
    M = L - n
    V, adj = build_graph(n, L)

    print(f"\n{'='*70}")
    print(f"  n = {n}, L = {L}, M = {M}, 2M/n = {2*M/n:.4f}")
    print(f"  H = ({2*n}, {n+L}]")
    print(f"{'='*70}")

    # Find highly composite targets
    hc_targets = find_hc_targets(n, L)

    if not hc_targets:
        print("  No highly composite targets found.")
        return

    print(f"\n  Top 5 highly composite targets in H:")
    for tau_v, m, divs in hc_targets[:5]:
        fac = factorize(m)
        fac_str = " * ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in fac)
        print(f"    m = {m} = {fac_str}, tau_V(m) = {tau_v}")

    # Test adversarial subsets for each HC target
    print(f"\n  Adversarial subset tests:")
    print(f"  {'m':>6} {'s':>4} {'#div_m':>7} {'Delta':>6} "
          f"{'Shearer':>9} {'CS':>9} {'actual':>8} {'Sh>=s':>6} {'CS>=s':>6}")

    any_shearer_fail = False
    any_cs_fail = False
    any_hall_fail = False

    for tau_v, m, divs in hc_targets[:10]:
        for s in [5, 10, 15, 20, 30, 50, min(100, n // 4)]:
            if s > n // 2:
                continue

            S = adversarial_subset(m, divs, M, s, n)
            if len(S) != s:
                continue

            # Check Case B condition
            threshold = M / (s + 1)
            if min(S) <= threshold:
                continue

            sh_bound, Delta, log_sum, actual_N = shearer_bound(S, adj)
            cs_bound, _ = cauchy_schwarz_bound(S, adj, M)

            sh_ok = sh_bound >= s
            cs_ok = cs_bound >= s
            hall_ok = actual_N >= s

            if not sh_ok:
                any_shearer_fail = True
            if not cs_ok:
                any_cs_fail = True
            if not hall_ok:
                any_hall_fail = True

            # Count how many divisors of m are in S
            div_in_S = sum(1 for k in S if m % k == 0)

            sh_str = f"{sh_bound:.2f}" if sh_bound < 1e6 else "large"
            cs_str = f"{cs_bound:.2f}" if cs_bound < 1e6 else "large"
            sh_mark = "ok" if sh_ok else "FAIL"
            cs_mark = "ok" if cs_ok else "FAIL"

            print(f"  {m:>6} {s:>4} {div_in_S:>7} {Delta:>6} "
                  f"{sh_str:>9} {cs_str:>9} {actual_N:>8} "
                  f"{sh_mark:>6} {cs_mark:>6}"
                  + ("  ***HALL FAIL***" if not hall_ok else ""))

    # Also test random Case B subsets with many elements sharing divisors
    print(f"\n  Random high-overlap Case B subsets:")
    import random
    random.seed(42)

    for trial in range(20):
        s = random.choice([5, 10, 20, 50, min(100, n // 4)])
        if s > n // 2:
            continue
        threshold = M / (s + 1)

        # Pick elements that share many common factors
        # Strategy: pick multiples of small d in (threshold, n/2]
        for d in [2, 3, 4, 6, 12]:
            candidates = [k for k in range(1, n // 2 + 1)
                         if k % d == 0 and k > threshold]
            if len(candidates) < s:
                continue
            S = sorted(random.sample(candidates, s))

            sh_bound, Delta, log_sum, actual_N = shearer_bound(S, adj)
            cs_bound, _ = cauchy_schwarz_bound(S, adj, M)

            sh_ok = sh_bound >= s
            cs_ok = cs_bound >= s

            if not sh_ok or not cs_ok:
                sh_mark = "ok" if sh_ok else "FAIL"
                cs_mark = "ok" if cs_ok else "FAIL"
                print(f"    d={d}, s={s}: Delta={Delta}, "
                      f"Shearer={sh_bound:.2f} ({sh_mark}), "
                      f"CS={cs_bound:.2f} ({cs_mark}), "
                      f"actual={actual_N}")
                if not sh_ok:
                    any_shearer_fail = True
                if not cs_ok:
                    any_cs_fail = True

    print(f"\n  Summary for n={n}:")
    print(f"    Shearer failures:       {'YES' if any_shearer_fail else 'none'}")
    print(f"    Cauchy-Schwarz failures: {'YES' if any_cs_fail else 'none'}")
    print(f"    Hall failures:          {'YES' if any_hall_fail else 'none'}")

    return any_shearer_fail, any_cs_fail, any_hall_fail

def test_exhaustive_small(n, eps=0.05):
    """For small n, test ALL Case B subsets up to size s_max."""
    L = target_L(n, eps)
    M = L - n
    V, adj = build_graph(n, L)
    s_max = min(8, n // 4)

    print(f"\n{'='*70}")
    print(f"  EXHAUSTIVE test: n = {n}, s_max = {s_max}")
    print(f"{'='*70}")

    worst_shearer_ratio = float('inf')
    worst_cs_ratio = float('inf')
    worst_S_shearer = None
    worst_S_cs = None

    V_top = [k for k in V if k > n // 4]  # Focus on top quarter

    for s in range(2, s_max + 1):
        threshold = M / (s + 1)
        candidates = [k for k in V_top if k > threshold]

        if len(candidates) < s:
            continue

        count = 0
        max_combos = 5000  # Cap for tractability

        for S in combinations(candidates, s):
            S = list(S)
            count += 1
            if count > max_combos:
                break

            sh_bound, Delta, _, actual_N = shearer_bound(S, adj)
            cs_bound, _ = cauchy_schwarz_bound(S, adj, M)

            if s > 0:
                sh_ratio = sh_bound / s
                cs_ratio = cs_bound / s

                if sh_ratio < worst_shearer_ratio:
                    worst_shearer_ratio = sh_ratio
                    worst_S_shearer = (s, list(S), Delta, sh_bound, actual_N)

                if cs_ratio < worst_cs_ratio:
                    worst_cs_ratio = cs_ratio
                    worst_S_cs = (s, list(S), Delta, cs_bound, actual_N)

        print(f"  s={s}: tested {min(count, max_combos)} subsets")

    if worst_S_shearer:
        s, S, Delta, bound, actual = worst_S_shearer
        status = "FAIL" if bound < s else "ok"
        print(f"\n  Worst Shearer ratio: {worst_shearer_ratio:.4f} ({status})")
        print(f"    S = {S[:8]}{'...' if len(S) > 8 else ''}")
        print(f"    s={s}, Delta={Delta}, bound={bound:.2f}, actual={actual}")

    if worst_S_cs:
        s, S, Delta, bound, actual = worst_S_cs
        status = "FAIL" if bound < s else "ok"
        print(f"\n  Worst CS ratio: {worst_cs_ratio:.4f} ({status})")
        print(f"    S = {S[:8]}{'...' if len(S) > 8 else ''}")
        print(f"    s={s}, Delta={Delta}, bound={bound:.2f}, actual={actual}")

def main():
    t0 = time.time()

    print("=" * 70)
    print("  ERDOS 710 — ADVERSARIAL BOUND TESTING")
    print("  Testing Shearer and Cauchy-Schwarz against worst-case subsets")
    print("=" * 70)

    # Exhaustive for small n
    for n in [100, 200]:
        test_exhaustive_small(n)

    # Adversarial for larger n
    for n in [200, 500, 1000, 2000]:
        test_adversarial(n)

    print(f"\nTotal time: {time.time() - t0:.1f}s")

if __name__ == '__main__':
    main()
