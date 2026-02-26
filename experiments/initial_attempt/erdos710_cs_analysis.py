#!/usr/bin/env python3
"""
Analyze the Cauchy-Schwarz bound structure for Case B subsets.
Goal: understand what ratio P/D looks like and what bounds are provable.

CS bound: |N_H(S)| >= D^2 / (D + 2P)
where D = sum|N_k|, P = sum_{k1<k2} |N_{k1} ∩ N_{k2}|

Key quantities to compute:
- D, P, T2 = D + 2P, CS = D^2/T2
- sigma = sum 1/k, G = sum 1/lcm(k1,k2) (so P ~ M*G)
- Ratios: P/D, G/sigma, G/sigma^2
"""
import math
from math import gcd, log, floor
from collections import defaultdict
from erdos710_toolkit import target_L, factorize, divisors_up_to

def lcm(a, b):
    return a * b // gcd(a, b)

def build_graph(n, L):
    adj = {}
    for k in range(1, n // 2 + 1):
        j0 = (2 * n) // k + 1
        j1 = (n + L) // k
        adj[k] = set(k * j for j in range(j0, j1 + 1))
    return adj

def analyze_cs_structure(n, eps=0.05):
    L = target_L(n, eps)
    M = L - n
    nL = n + L
    alpha = 2 * M / n
    adj = build_graph(n, L)

    # Find HC targets
    best_m, best_tau, best_divs = 0, 0, []
    for m in range(2 * n + 1, n + L + 1):
        divs = divisors_up_to(m, n // 2)
        if len(divs) > best_tau:
            best_tau = len(divs)
            best_m = m
            best_divs = divs

    fac = factorize(best_m)
    fac_str = " * ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in fac)

    print(f"\n{'='*75}")
    print(f"  n={n}, M={M}, n+L={nL}, alpha=2M/n={alpha:.4f}")
    print(f"  Best HC target: m={best_m} = {fac_str}, tau_V={best_tau}")
    print(f"{'='*75}")

    header = (f"  {'type':>12} {'s':>4} {'D':>6} {'P':>8} {'T2':>8} "
              f"{'CS':>7} {'CS/s':>6} {'sigma':>7} {'G':>8} "
              f"{'G/sig':>7} {'P/D':>6} {'nz%':>5}")
    print(header)

    results = []

    for label, S_builder in [
        ("HC-divs", lambda s: _build_hc_subset(best_m, best_divs, M, s, n)),
        ("consec", lambda s: _build_consecutive(M, s, n)),
        ("even", lambda s: _build_multiples(2, M, s, n)),
        ("mult-6", lambda s: _build_multiples(6, M, s, n)),
        ("random", lambda s: _build_random(M, s, n)),
    ]:
        for s in [5, 10, 20, 50, 100, 200, min(n // 4, 500)]:
            if s > n // 2:
                continue
            S = S_builder(s)
            if S is None or len(S) != s:
                continue

            # Verify Case B
            threshold = M / (s + 1)
            if min(S) <= threshold:
                continue

            r = compute_cs_data(S, adj, M, nL)
            cs_over_s = r['cs'] / s if s > 0 else 0
            nz_pct = 100 * r['nz_pairs'] / r['total_pairs'] if r['total_pairs'] > 0 else 0

            print(f"  {label:>12} {s:>4} {r['D']:>6} {r['P']:>8} {r['T2']:>8} "
                  f"{r['cs']:>7.1f} {cs_over_s:>6.2f} {r['sigma']:>7.4f} {r['G']:>8.5f} "
                  f"{r['G_over_sig']:>7.3f} {r['P_over_D']:>6.3f} {nz_pct:>5.1f}")
            results.append((label, s, r))

    # Summary statistics
    print(f"\n  Key observations for n={n}:")
    for label, s, r in results:
        cs_over_s = r['cs'] / s
        if cs_over_s < 2.0:
            print(f"    TIGHT: {label} s={s}: CS/s = {cs_over_s:.3f}, P/D = {r['P_over_D']:.3f}")

    return results

def compute_cs_data(S, adj, M, nL):
    s = len(S)
    # Degrees
    degs = [len(adj.get(k, set())) for k in S]
    D = sum(degs)
    sigma = sum(1.0 / k for k in S)

    # Pairwise intersections
    P = 0
    G = 0.0
    total_pairs = 0
    nz_pairs = 0

    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            k1, k2 = S[i], S[j]
            l = lcm(k1, k2)
            total_pairs += 1
            if l <= nL:
                isect = len(adj[k1] & adj[k2])
                P += isect
                nz_pairs += 1
            G += gcd(k1, k2) / (k1 * k2)

    T2 = D + 2 * P
    cs = D * D / T2 if T2 > 0 else 0

    return {
        'D': D, 'P': P, 'T2': T2, 'cs': cs,
        'sigma': sigma, 'G': G,
        'G_over_sig': G / sigma if sigma > 0 else 0,
        'P_over_D': P / D if D > 0 else 0,
        'total_pairs': total_pairs,
        'nz_pairs': nz_pairs,
    }

def _build_hc_subset(m, divs, M, s, n):
    threshold = M / (s + 1)
    cb_divs = sorted([d for d in divs if threshold < d <= n // 2])
    S = list(cb_divs[:s])
    used = set(S)
    k = n // 2
    while len(S) < s and k > threshold:
        if k not in used:
            S.append(k)
        k -= 1
    S.sort()
    return S if len(S) == s else None

def _build_consecutive(M, s, n):
    threshold = M / (s + 1)
    start = max(int(threshold) + 1, n // 2 - s + 1)
    if start + s - 1 > n // 2:
        start = n // 2 - s + 1
    if start <= threshold:
        start = int(threshold) + 1
    S = list(range(start, start + s))
    if len(S) != s or max(S) > n // 2 or min(S) <= threshold:
        return None
    return S

def _build_multiples(d, M, s, n):
    threshold = M / (s + 1)
    candidates = [k for k in range(d, n // 2 + 1, d) if k > threshold]
    if len(candidates) < s:
        return None
    return sorted(candidates[-s:])

def _build_random(M, s, n):
    import random
    random.seed(42 + s + n)
    threshold = M / (s + 1)
    candidates = [k for k in range(int(threshold) + 1, n // 2 + 1)]
    if len(candidates) < s:
        return None
    return sorted(random.sample(candidates, s))

def check_universal_lcm_bound(n, eps=0.05):
    """Check sum 1/lcm for ALL of {1,...,n/2} vs (log n)^2."""
    L = target_L(n, eps)
    M = L - n
    N = n // 2

    # Compute sum_{a,b=1}^{N} 1/lcm(a,b) by summing
    # This is O(N^2) so only for small N
    if N > 300:
        print(f"\n  n={n}: N={N} too large for exhaustive lcm sum")
        return

    total = 0.0
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            total += 1.0 / lcm(a, b)
            if a != b:
                total += 1.0 / lcm(a, b)

    diag = sum(1.0 / k for k in range(1, N + 1))
    off_diag = (total - diag) / 2

    logN_sq = log(N) ** 2
    print(f"\n  n={n}, N={N}:")
    print(f"    sum_{{a,b}} 1/lcm = {total:.4f}")
    print(f"    diagonal sum 1/k = {diag:.4f}")
    print(f"    off-diag sum = {off_diag:.4f}")
    print(f"    (log N)^2 = {logN_sq:.4f}")
    print(f"    ratio to (log N)^2 = {total / logN_sq:.4f}")

if __name__ == '__main__':
    # Universal lcm sum check
    print("=" * 75)
    print("  UNIVERSAL 1/lcm SUMS")
    print("=" * 75)
    for n in [100, 200, 400, 600]:
        check_universal_lcm_bound(n)

    # CS structure analysis
    print("\n" + "=" * 75)
    print("  CAUCHY-SCHWARZ STRUCTURAL ANALYSIS")
    print("=" * 75)
    for n in [200, 500, 1000, 2000]:
        analyze_cs_structure(n)
