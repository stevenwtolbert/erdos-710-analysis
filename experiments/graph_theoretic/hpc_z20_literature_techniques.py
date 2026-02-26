#!/usr/bin/env python3
"""
ERDŐS 710 — EXPERIMENTS Z20a–Z20e: LITERATURE TECHNIQUE TESTS

Five experiments testing techniques from recent papers (2020–2025) on the
remaining analytic gap: Hall's condition for S₊ (large smooth numbers).

Z20a: Spectral Analysis (requires numpy)
  - Compute eigenvalues/singular values of smooth bipartite adjacency matrix
  - Check if spectral gap grows with n (Garland/Aharoni-Haxell matching)

Z20b: Green-Walker for Adversarial Subsets
  - GCD concentration in greedy-ordered prefixes
  - Test whether GW constant C_needed < 1.0 at D=3 for hardest prefixes

Z20c: Multiplication Table for Adversarial Subsets
  - Mehdizadeh expansion for adversarial subsets, not just full pool
  - Track per-element new-target contribution in greedy order

Z20d: Randomized Janson / Concentration
  - Compare greedy adversary with random subsets
  - Compute Janson ratio μ²/(2Δ̄) for greedy vs random

Z20e: Ford-Type Divisor Concentration
  - CS with artificially capped τ(m) — would sharper max_τ bounds suffice?
  - Ford ratio = log(max_τ) / √(log n)

Usage:
  python3 hpc_z20_literature_techniques.py --experiments Z20a,Z20b,Z20c,Z20d,Z20e
  python3 hpc_z20_literature_techniques.py --experiments Z20b,Z20c --n_values 500,1000,5000
"""

import sys
import time
import argparse
import random
from math import gcd, log, sqrt, exp, floor, ceil
from collections import defaultdict, Counter

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

C_TARGET = 2 / sqrt(exp(1))
EPS = 0.05


def target_L(n, eps=EPS):
    if n < 3:
        return 3 * n
    return int((C_TARGET + eps) * n * sqrt(log(n) / log(log(n))))


def lcm(a, b):
    return a * b // gcd(a, b)


def largest_prime_factor(x):
    if x <= 1:
        return 1
    p = 2
    lpf = 1
    while p * p <= x:
        while x % p == 0:
            lpf = p
            x //= p
        p += 1
    if x > 1:
        lpf = x
    return lpf


def is_smooth(x, bound):
    """Check if x is y-smooth (all prime factors <= bound)."""
    return largest_prime_factor(x) <= bound


def compute_targets(k, n, L):
    j0 = (2 * n) // k + 1
    j1 = (n + L) // k
    return set(k * j for j in range(j0, j1 + 1))


def compute_degree(k, n, L):
    j0 = (2 * n) // k + 1
    j1 = (n + L) // k
    return max(0, j1 - j0 + 1)


def build_greedy_minimizer(pool, targets, target_to_pool, s):
    """Build greedy |NH|-minimizer."""
    T = []
    NH = set()
    rem = set(pool)
    new_count = {k: len(targets[k]) for k in pool}
    for step in range(s):
        best_k, best_new = None, float('inf')
        for k in rem:
            nc = new_count[k]
            if nc < best_new or (nc == best_new and (best_k is None or k > best_k)):
                best_new, best_k = nc, k
        if best_k is None:
            break
        T.append(best_k)
        rem.discard(best_k)
        newly_covered = targets[best_k] - NH
        NH |= newly_covered
        for h in newly_covered:
            for k2 in target_to_pool.get(h, []):
                if k2 in rem:
                    new_count[k2] -= 1
    return T, NH


def hopcroft_karp(graph, U, V_set):
    """Hopcroft-Karp maximum matching."""
    from collections import deque

    pair_u = {u: None for u in U}
    pair_v = {v: None for v in V_set}
    dist = {}
    INF = float('inf')

    def bfs():
        queue = deque()
        for u in U:
            if pair_u[u] is None:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in graph.get(u, []):
                next_u = pair_v.get(v)
                if next_u is None:
                    found = True
                elif dist.get(next_u, INF) == INF:
                    dist[next_u] = dist[u] + 1
                    queue.append(next_u)
        return found

    def dfs(u):
        for v in graph.get(u, []):
            next_u = pair_v.get(v)
            if next_u is None or (dist.get(next_u, INF) == dist[u] + 1 and dfs(next_u)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in U:
            if pair_u[u] is None:
                if dfs(u):
                    matching += 1

    return matching, {u: v for u, v in pair_u.items() if v is not None}


def build_smooth_pool(n, L, sf):
    """Build smooth pool and greedy minimizer. Returns dict with all components."""
    M = L - n
    N = n // 2
    delta = 2 * M / n - 1
    nL = n + L
    sqrt_nL = nL ** 0.5
    smooth_bound = int(sqrt_nL)

    s = int(sf * N)
    if s < 10:
        return None
    alpha = M / (s + 1)
    pool = sorted(range(int(alpha) + 1, N + 1))
    if len(pool) < s:
        return None

    lpf_cache = {}
    for k in range(1, N + 1):
        lpf_cache[k] = largest_prime_factor(k)

    targets = {}
    for k in pool:
        targets[k] = compute_targets(k, n, L)
    target_to_pool = {}
    for k in pool:
        for h in targets[k]:
            if h not in target_to_pool:
                target_to_pool[h] = []
            target_to_pool[h].append(k)

    T, NH_full = build_greedy_minimizer(pool, targets, target_to_pool, s)
    t = len(T)

    # Split
    R = [k for k in T if lpf_cache[k] > sqrt_nL]
    S_smooth = [k for k in T if lpf_cache[k] <= sqrt_nL]

    # Smooth pool (all smooth elements from full pool)
    pool_smooth = [k for k in pool if lpf_cache[k] <= sqrt_nL]

    # Build smooth-only targets and reverse index
    smooth_targets = {}
    for k in pool_smooth:
        smooth_targets[k] = targets[k]
    smooth_t2p = {}
    for k in pool_smooth:
        for h in targets[k]:
            if h not in smooth_t2p:
                smooth_t2p[h] = []
            smooth_t2p[h].append(k)

    # Build smooth-only greedy minimizer
    s_smooth = min(s, len(pool_smooth))
    T_smooth_greedy, NH_smooth_greedy = build_greedy_minimizer(
        pool_smooth, smooth_targets, smooth_t2p, s_smooth)

    return {
        'n': n, 'L': L, 'M': M, 'N': N, 'delta': delta,
        'nL': nL, 'sqrt_nL': sqrt_nL, 'smooth_bound': smooth_bound,
        's': s, 'alpha': alpha,
        'pool': pool, 'pool_smooth': pool_smooth,
        'targets': targets, 'target_to_pool': target_to_pool,
        'smooth_targets': smooth_targets, 'smooth_t2p': smooth_t2p,
        'lpf_cache': lpf_cache,
        'T': T, 'NH_full': NH_full, 't': t,
        'R': R, 'S_smooth': S_smooth,
        'T_smooth_greedy': T_smooth_greedy,
        'NH_smooth_greedy': NH_smooth_greedy,
    }


def greedy_prefixes(T_ordered, targets, fracs):
    """Extract greedy-ordered prefixes at given size fractions."""
    t = len(T_ordered)
    result = []
    for f in fracs:
        size = max(1, int(f * t))
        if size > t:
            size = t
        T_prefix = T_ordered[:size]
        NH_prefix = set()
        for k in T_prefix:
            NH_prefix |= targets[k]
        result.append((f, T_prefix, NH_prefix))
    return result


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT Z20a: SPECTRAL ANALYSIS
# ═══════════════════════════════════════════════════════════════

def experiment_Z20a(n_values, sf=0.5):
    """
    Compute eigenvalues/singular values of smooth bipartite adjacency matrix.
    Check if spectral gap grows with n (would enable Garland/Aharoni-Haxell proof).
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z20a: SPECTRAL ANALYSIS OF SMOOTH BIPARTITE GRAPH")
    print("=" * 78)

    if not HAS_NUMPY:
        print("\n  *** SKIPPED: numpy not available ***")
        print("  Install numpy to run spectral analysis.")
        return []

    results = []

    for n in n_values:
        L = target_L(n)
        if L <= n:
            continue
        data = build_smooth_pool(n, L, sf)
        if data is None:
            continue

        T_smooth = data['T_smooth_greedy']
        targets = data['targets']
        smooth_bound = data['smooth_bound']
        n_smooth = len(T_smooth)

        if n_smooth < 3:
            print(f"  n={n}: too few smooth elements ({n_smooth}), skipping")
            continue

        # Build smooth-target bipartite graph
        smooth_graph = {}
        all_smooth_targets = set()
        for k in T_smooth:
            stgt = set()
            for m in targets[k]:
                if largest_prime_factor(m) <= smooth_bound:
                    stgt.add(m)
            smooth_graph[k] = stgt
            all_smooth_targets |= stgt

        smooth_tgt_list = sorted(all_smooth_targets)
        n_targets = len(smooth_tgt_list)
        tgt_idx = {m: i for i, m in enumerate(smooth_tgt_list)}

        print(f"\n  n={n}: |S_smooth|={n_smooth}, |smooth_targets|={n_targets}")

        # Build biadjacency matrix B (|S_smooth| x |smooth_targets|)
        B = np.zeros((n_smooth, n_targets), dtype=np.float64)
        for i, k in enumerate(T_smooth):
            for m in smooth_graph.get(k, []):
                j = tgt_idx.get(m)
                if j is not None:
                    B[i, j] = 1.0

        # Singular values of B
        sigma = np.linalg.svd(B, compute_uv=False)
        sigma1 = sigma[0] if len(sigma) > 0 else 0
        sigma2 = sigma[1] if len(sigma) > 1 else 0
        sigma_ratio = sigma2 / sigma1 if sigma1 > 0 else 1.0
        cheeger = (1 - sigma_ratio) / 2

        # Eigenvalues of BB^T
        BBT = B @ B.T
        eigvals = np.linalg.eigvalsh(BBT)
        eigvals = sorted(eigvals, reverse=True)
        lam1 = eigvals[0] if len(eigvals) > 0 else 0
        lam2 = eigvals[1] if len(eigvals) > 1 else 0
        spectral_gap = lam1 - lam2

        # Row/column statistics
        row_sums = B.sum(axis=1)
        col_sums = B.sum(axis=0)
        avg_row = row_sums.mean()
        avg_col = col_sums.mean()
        min_row = row_sums.min()
        max_col = col_sums.max()

        print(f"    Biadjacency: {n_smooth} x {n_targets}, nnz={int(B.sum())}")
        print(f"    Row sums (smooth deg): avg={avg_row:.2f}, min={min_row:.0f}")
        print(f"    Col sums (target tau):  avg={avg_col:.2f}, max={max_col:.0f}")
        print(f"    Singular values: σ₁={sigma1:.4f}, σ₂={sigma2:.4f}")
        print(f"    σ₂/σ₁ = {sigma_ratio:.4f}")
        print(f"    Cheeger bound = (1 - σ₂/σ₁)/2 = {cheeger:.4f}")
        print(f"    Eigenvalues of BBᵀ: λ₁={lam1:.2f}, λ₂={lam2:.2f}")
        print(f"    Spectral gap λ₁−λ₂ = {spectral_gap:.2f}")

        # Top 5 singular values
        top_sv = sigma[:min(5, len(sigma))]
        print(f"    Top 5 σ: {['%.3f' % s for s in top_sv]}")

        hit = sigma_ratio < 0.5 or cheeger > 0.3
        print(f"    HIT (σ₂/σ₁ < 0.5 or Cheeger > 0.3): {'YES' if hit else 'NO'}")

        rec = {
            'n': n, 'n_smooth': n_smooth, 'n_targets': n_targets,
            'sigma1': float(sigma1), 'sigma2': float(sigma2),
            'sigma_ratio': float(sigma_ratio), 'cheeger': float(cheeger),
            'lam1': float(lam1), 'lam2': float(lam2),
            'spectral_gap': float(spectral_gap),
            'avg_row': float(avg_row), 'min_row': float(min_row),
            'avg_col': float(avg_col), 'max_col': float(max_col),
            'hit': hit,
        }
        results.append(rec)

    # Summary
    print(f"\n{'='*78}")
    print(f"  Z20a SUMMARY")
    print(f"{'='*78}")
    if results:
        print(f"\n  {'n':>6} {'|S|':>5} {'|tgt|':>6} "
              f"{'σ₁':>7} {'σ₂':>7} {'σ₂/σ₁':>7} {'Cheeg':>6} "
              f"{'λ₁':>8} {'gap':>8} {'HIT':>4}")
        for r in results:
            print(f"  {r['n']:>6} {r['n_smooth']:>5} {r['n_targets']:>6} "
                  f"{r['sigma1']:>7.3f} {r['sigma2']:>7.3f} "
                  f"{r['sigma_ratio']:>7.4f} {r['cheeger']:>6.4f} "
                  f"{r['lam1']:>8.2f} {r['spectral_gap']:>8.2f} "
                  f"{'Y' if r['hit'] else 'N':>4}")

        any_hit = any(r['hit'] for r in results)
        print(f"\n  Spectral gap viable: {'YES — further investigation warranted' if any_hit else 'NO — spectral gap too small'}")
    else:
        print("  No results (numpy not available or insufficient data)")

    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT Z20b: GREEN-WALKER FOR ADVERSARIAL SUBSETS
# ═══════════════════════════════════════════════════════════════

def experiment_Z20b(n_values, sf=0.5):
    """
    Test Green-Walker GCD bound on adversarial (greedy-ordered) prefixes.
    For each prefix T': compute GCD distribution and C_needed for contradiction.
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z20b: GREEN-WALKER GCD BOUND ON ADVERSARIAL SUBSETS")
    print("=" * 78)

    prefix_fracs = [0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    D_values = [2, 3, 5, 10]
    MAX_PAIRS = 5000  # sample limit for large sets

    results = []

    for n in n_values:
        L = target_L(n)
        if L <= n:
            continue
        data = build_smooth_pool(n, L, sf)
        if data is None:
            continue

        T_smooth = data['T_smooth_greedy']
        targets = data['targets']
        N = data['N']
        nL = data['nL']
        delta = data['delta']
        n_smooth = len(T_smooth)

        if n_smooth < 5:
            continue

        print(f"\n{'='*70}")
        print(f"  n={n}, N={N}, δ={delta:.3f}, |T_smooth|={n_smooth}")

        prefixes = greedy_prefixes(T_smooth, targets, prefix_fracs)

        for frac, T_prefix, NH_prefix in prefixes:
            tp = len(T_prefix)
            nhp = len(NH_prefix)
            ratio = nhp / tp if tp > 0 else 0

            # Compute GCD distribution for pairs in T_prefix
            gcd_counts = {D: 0 for D in D_values}
            total_pairs = 0

            if tp <= 200:
                # Exact computation
                for i in range(tp):
                    for j in range(i + 1, tp):
                        g = gcd(T_prefix[i], T_prefix[j])
                        total_pairs += 1
                        for D in D_values:
                            if g >= D:
                                gcd_counts[D] += 1
            else:
                # Sample pairs
                total_pairs = MAX_PAIRS
                for _ in range(MAX_PAIRS):
                    i, j = random.sample(range(tp), 2)
                    g = gcd(T_prefix[i], T_prefix[j])
                    for D in D_values:
                        if g >= D:
                            gcd_counts[D] += 1

            # Compute C_needed = delta_D * D^2 * |T'|^2 / N^2
            # Green-Walker: |A||B| << delta^{-2-eps} * X*Y / D^2
            # For contradiction with Hall violation: need C_needed < 1
            # Equivalently: delta_D * D^2 * |T'|^2 / N^2 < 1
            print(f"\n  frac={frac:.2f}: |T'|={tp}, |NH|={nhp}, ratio={ratio:.3f}")
            c_needed_row = []
            for D in D_values:
                delta_D = gcd_counts[D] / max(1, total_pairs)
                c_needed = delta_D * D * D * tp * tp / (N * N) if N > 0 else float('inf')
                c_needed_row.append(c_needed)
                print(f"    D={D:>2}: δ_D={delta_D:.4f}, "
                      f"C_needed={c_needed:.6f}")

            rec = {
                'n': n, 'frac': frac, 'tp': tp, 'nhp': nhp, 'ratio': ratio,
                'total_pairs': total_pairs,
            }
            for i, D in enumerate(D_values):
                delta_D = gcd_counts[D] / max(1, total_pairs)
                rec[f'delta_{D}'] = delta_D
                rec[f'C_{D}'] = c_needed_row[i]
            results.append(rec)

    # Summary
    print(f"\n{'='*78}")
    print(f"  Z20b SUMMARY")
    print(f"{'='*78}")
    print(f"\n  {'n':>6} {'frac':>5} {'|T|':>5} {'ratio':>6} "
          f"{'δ_2':>6} {'C_2':>8} {'δ_3':>6} {'C_3':>8} "
          f"{'δ_5':>6} {'C_5':>8}")
    for r in results:
        print(f"  {r['n']:>6} {r['frac']:>5.2f} {r['tp']:>5} {r['ratio']:>6.3f} "
              f"{r.get('delta_2',0):>6.4f} {r.get('C_2',0):>8.5f} "
              f"{r.get('delta_3',0):>6.4f} {r.get('C_3',0):>8.5f} "
              f"{r.get('delta_5',0):>6.4f} {r.get('C_5',0):>8.5f}")

    # Check hardest prefix (50-90%)
    hardest = [r for r in results if 0.45 <= r['frac'] <= 0.95]
    if hardest:
        worst_C3 = max(r.get('C_3', 0) for r in hardest)
        print(f"\n  Hardest prefix (50-90%) worst C_3: {worst_C3:.6f}")
        hit = worst_C3 < 1.0
        print(f"  HIT (C_3 < 1.0 for hardest prefix): {'YES' if hit else 'NO'}")
    else:
        print("\n  No hardest-prefix data.")

    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT Z20c: MULTIPLICATION TABLE FOR ADVERSARIAL SUBSETS
# ═══════════════════════════════════════════════════════════════

def experiment_Z20c(n_values, sf=0.5):
    """
    Mehdizadeh's smooth multiplication table expansion for adversarial subsets.
    Track per-element new-target contribution in greedy order.
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z20c: MULTIPLICATION TABLE EXPANSION (ADVERSARIAL)")
    print("=" * 78)

    prefix_fracs = [0.10, 0.25, 0.50, 0.75, 0.90, 1.00]

    results = []

    for n in n_values:
        L = target_L(n)
        if L <= n:
            continue
        data = build_smooth_pool(n, L, sf)
        if data is None:
            continue

        T_smooth = data['T_smooth_greedy']
        targets = data['targets']
        smooth_bound = data['smooth_bound']
        delta = data['delta']
        N = data['N']
        n_smooth = len(T_smooth)

        if n_smooth < 5:
            continue

        print(f"\n{'='*70}")
        print(f"  n={n}, N={N}, δ={delta:.3f}, |T_smooth|={n_smooth}")

        # Track per-element new-target contribution
        NH_running = set()
        NH_smooth_running = set()
        new_per_element = []
        new_smooth_per_element = []

        for k in T_smooth:
            tgt = targets[k]
            new = tgt - NH_running
            new_per_element.append(len(new))
            NH_running |= tgt

            # Smooth targets only
            smooth_new = set()
            for m in tgt:
                if m not in NH_smooth_running and largest_prime_factor(m) <= smooth_bound:
                    smooth_new.add(m)
            new_smooth_per_element.append(len(smooth_new))
            NH_smooth_running |= smooth_new

        # Prefix analysis
        prefixes = greedy_prefixes(T_smooth, targets, prefix_fracs)

        print(f"\n  Per-element new targets (first/middle/last 5):")
        if n_smooth >= 15:
            print(f"    First 5: {new_per_element[:5]}")
            mid = n_smooth // 2
            print(f"    Middle 5 (around {mid}): {new_per_element[mid-2:mid+3]}")
            print(f"    Last 5: {new_per_element[-5:]}")
        else:
            print(f"    All: {new_per_element}")

        for frac, T_prefix, NH_prefix in prefixes:
            tp = len(T_prefix)
            nhp = len(NH_prefix)
            ratio = nhp / tp if tp > 0 else 0

            # Smooth targets in NH_prefix
            NH_smooth = set()
            for m in NH_prefix:
                if largest_prime_factor(m) <= smooth_bound:
                    NH_smooth.add(m)
            smooth_ratio = len(NH_smooth) / tp if tp > 0 else 0

            # Compute E1 for this prefix (sum of degrees)
            E1 = sum(len(targets[k]) for k in T_prefix)
            distinct_total = nhp
            inv_avg_tau = distinct_total / E1 if E1 > 0 else 0

            print(f"\n  frac={frac:.2f}: |T'|={tp}, |NH|={nhp}, ratio={ratio:.3f}")
            print(f"    |NH_smooth|={len(NH_smooth)}, smooth_ratio={smooth_ratio:.3f}")
            print(f"    E1={E1}, distinct/E1={inv_avg_tau:.4f}")
            print(f"    Avg new per elem in prefix: {sum(new_per_element[:tp])/tp:.2f}")

            rec = {
                'n': n, 'frac': frac, 'tp': tp, 'nhp': nhp, 'ratio': ratio,
                'nh_smooth': len(NH_smooth), 'smooth_ratio': smooth_ratio,
                'E1': E1, 'inv_avg_tau': inv_avg_tau,
                'avg_new': sum(new_per_element[:tp]) / tp,
            }
            results.append(rec)

    # Summary
    print(f"\n{'='*78}")
    print(f"  Z20c SUMMARY")
    print(f"{'='*78}")
    print(f"\n  {'n':>6} {'frac':>5} {'|T|':>5} {'|NH|':>6} {'ratio':>6} "
          f"{'sm_rat':>6} {'d/E1':>6} {'avg_new':>7}")
    for r in results:
        print(f"  {r['n']:>6} {r['frac']:>5.2f} {r['tp']:>5} {r['nhp']:>6} "
              f"{r['ratio']:>6.3f} {r['smooth_ratio']:>6.3f} "
              f"{r['inv_avg_tau']:>6.4f} {r['avg_new']:>7.2f}")

    # Check: ratio >= 1.0 for ALL prefixes at ALL n
    all_pass = all(r['ratio'] >= 1.0 for r in results)
    min_ratio = min(r['ratio'] for r in results) if results else 0
    print(f"\n  |NH|/|T'| ≥ 1.0 for ALL prefixes: {'YES' if all_pass else 'NO'}")
    print(f"  Min ratio across all: {min_ratio:.4f}")

    # Smooth ratio check
    all_smooth_pass = all(r['smooth_ratio'] >= 1.0 for r in results)
    min_smooth = min(r['smooth_ratio'] for r in results) if results else 0
    print(f"  |NH_smooth|/|T'| ≥ 1.0 for ALL: {'YES' if all_smooth_pass else 'NO'}")
    print(f"  Min smooth ratio: {min_smooth:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT Z20d: RANDOMIZED JANSON / CONCENTRATION
# ═══════════════════════════════════════════════════════════════

def experiment_Z20d(n_values, sf=0.5, n_trials=100):
    """
    Compare greedy adversary with random subsets.
    Compute Janson ratio for greedy vs random.
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z20d: RANDOMIZED JANSON / CONCENTRATION COMPARISON")
    print("=" * 78)

    prefix_fracs = [0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    MAX_CODEG_PAIRS = 3000  # sample limit for codegree sum

    results = []

    for n in n_values:
        L = target_L(n)
        if L <= n:
            continue
        data = build_smooth_pool(n, L, sf)
        if data is None:
            continue

        T_smooth = data['T_smooth_greedy']
        targets = data['targets']
        pool_smooth = data['pool_smooth']
        delta = data['delta']
        N = data['N']
        nL = data['nL']
        n_smooth = len(T_smooth)

        if n_smooth < 10:
            continue

        print(f"\n{'='*70}")
        print(f"  n={n}, N={N}, δ={delta:.3f}, |T_smooth|={n_smooth}, |pool_smooth|={len(pool_smooth)}")

        for frac in prefix_fracs:
            size = max(1, int(frac * n_smooth))
            if size > len(pool_smooth):
                size = len(pool_smooth)

            # Greedy prefix
            T_greedy = T_smooth[:size]
            NH_greedy = set()
            for k in T_greedy:
                NH_greedy |= targets[k]
            greedy_ratio = len(NH_greedy) / size

            # Greedy E1 and codeg (Δ̄)
            greedy_E1 = sum(len(targets[k]) for k in T_greedy)
            # Compute codegree sum (sample if large)
            greedy_codeg = 0
            if size <= 100:
                for i in range(size):
                    for j in range(i + 1, size):
                        g = len(targets[T_greedy[i]] & targets[T_greedy[j]])
                        greedy_codeg += g
            else:
                n_sample = MAX_CODEG_PAIRS
                for _ in range(n_sample):
                    i, j = random.sample(range(size), 2)
                    g = len(targets[T_greedy[i]] & targets[T_greedy[j]])
                    greedy_codeg += g
                # Scale to estimate full sum
                total_possible = size * (size - 1) // 2
                greedy_codeg = greedy_codeg * total_possible / n_sample

            greedy_janson = (greedy_E1 ** 2) / (2 * greedy_codeg) if greedy_codeg > 0 else float('inf')

            # Random trials
            random_ratios = []
            random_E1s = []
            random_codegs = []
            random_jansons = []

            actual_trials = min(n_trials, 100)
            for trial in range(actual_trials):
                T_rand = random.sample(pool_smooth, min(size, len(pool_smooth)))
                NH_rand = set()
                for k in T_rand:
                    NH_rand |= targets[k]
                rand_ratio = len(NH_rand) / len(T_rand)
                random_ratios.append(rand_ratio)

                rand_E1 = sum(len(targets[k]) for k in T_rand)
                random_E1s.append(rand_E1)

                # Codegree sum (sample)
                rand_codeg = 0
                rs = len(T_rand)
                if rs <= 100:
                    for i in range(rs):
                        for j in range(i + 1, rs):
                            g = len(targets[T_rand[i]] & targets[T_rand[j]])
                            rand_codeg += g
                else:
                    n_samp = min(MAX_CODEG_PAIRS, rs * (rs - 1) // 2)
                    for _ in range(n_samp):
                        i, j = random.sample(range(rs), 2)
                        g = len(targets[T_rand[i]] & targets[T_rand[j]])
                        rand_codeg += g
                    total_possible = rs * (rs - 1) // 2
                    rand_codeg = rand_codeg * total_possible / n_samp

                random_codegs.append(rand_codeg)
                rand_janson = (rand_E1 ** 2) / (2 * rand_codeg) if rand_codeg > 0 else float('inf')
                random_jansons.append(rand_janson)

            # Statistics
            min_rand = min(random_ratios)
            avg_rand = sum(random_ratios) / len(random_ratios)
            max_rand = max(random_ratios)
            std_rand = (sum((x - avg_rand)**2 for x in random_ratios) / len(random_ratios)) ** 0.5
            cv_rand = std_rand / avg_rand if avg_rand > 0 else 0  # coefficient of variation

            gap = min_rand - greedy_ratio
            finite_jansons = [j for j in random_jansons if j < float('inf')]
            avg_janson_rand = sum(finite_jansons) / len(finite_jansons) if finite_jansons else 0
            min_janson_rand = min(finite_jansons) if finite_jansons else 0

            print(f"\n  frac={frac:.2f}: size={size}")
            print(f"    GREEDY: |NH|/t={greedy_ratio:.4f}, E1={greedy_E1}, "
                  f"Δ̄={greedy_codeg:.0f}, Janson={greedy_janson:.3f}")
            print(f"    RANDOM ({actual_trials} trials): "
                  f"min={min_rand:.4f}, avg={avg_rand:.4f}, max={max_rand:.4f}")
            print(f"    std={std_rand:.4f}, CV={cv_rand:.4f}")
            print(f"    Gap (min_rand - greedy) = {gap:.4f}")
            print(f"    Janson random: avg={avg_janson_rand:.3f}, min={min_janson_rand:.3f}")

            rec = {
                'n': n, 'frac': frac, 'size': size,
                'greedy_ratio': greedy_ratio, 'greedy_E1': greedy_E1,
                'greedy_codeg': greedy_codeg, 'greedy_janson': greedy_janson,
                'min_rand': min_rand, 'avg_rand': avg_rand, 'max_rand': max_rand,
                'std_rand': std_rand, 'cv_rand': cv_rand,
                'gap': gap,
                'avg_janson_rand': avg_janson_rand, 'min_janson_rand': min_janson_rand,
            }
            results.append(rec)

    # Summary
    print(f"\n{'='*78}")
    print(f"  Z20d SUMMARY")
    print(f"{'='*78}")
    print(f"\n  {'n':>6} {'frac':>5} {'sz':>5} "
          f"{'greedy':>7} {'min_R':>7} {'avg_R':>7} "
          f"{'gap':>7} {'CV':>6} "
          f"{'J_grd':>7} {'J_rnd':>7}")
    for r in results:
        j_grd = f"{r['greedy_janson']:.2f}" if r['greedy_janson'] < 1e6 else "∞"
        j_rnd = f"{r['avg_janson_rand']:.2f}" if r['avg_janson_rand'] < 1e6 else "∞"
        print(f"  {r['n']:>6} {r['frac']:>5.2f} {r['size']:>5} "
              f"{r['greedy_ratio']:>7.4f} {r['min_rand']:>7.4f} {r['avg_rand']:>7.4f} "
              f"{r['gap']:>7.4f} {r['cv_rand']:>6.4f} "
              f"{j_grd:>7} {j_rnd:>7}")

    # Check: gap bounded and not growing with n
    if results:
        gaps_by_n = defaultdict(list)
        for r in results:
            if 0.45 <= r['frac'] <= 0.95:
                gaps_by_n[r['n']].append(r['gap'])

        print(f"\n  Gap trend (hardest 50-90% prefixes):")
        for nv in sorted(gaps_by_n):
            avg_gap = sum(gaps_by_n[nv]) / len(gaps_by_n[nv])
            print(f"    n={nv}: avg_gap={avg_gap:.4f}")

        all_gaps = [r['gap'] for r in results if 0.45 <= r['frac'] <= 0.95]
        max_gap = max(all_gaps) if all_gaps else 0
        hit = max_gap <= 0.15
        print(f"  Max gap at hardest prefix: {max_gap:.4f}")
        print(f"  HIT (gap ≤ 0.15): {'YES' if hit else 'NO'}")

    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT Z20e: FORD-TYPE DIVISOR CONCENTRATION
# ═══════════════════════════════════════════════════════════════

def experiment_Z20e(n_values, sf=0.5):
    """
    CS with artificially capped τ — would sharper max_τ bounds suffice?
    Ford ratio = log(max_τ) / √(log n).
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z20e: FORD-TYPE DIVISOR CONCENTRATION / CAPPED CS")
    print("=" * 78)

    prefix_fracs = [0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    cap_values = [2, 3, 5, 10]

    results = []

    for n in n_values:
        L = target_L(n)
        if L <= n:
            continue
        data = build_smooth_pool(n, L, sf)
        if data is None:
            continue

        T_smooth = data['T_smooth_greedy']
        targets = data['targets']
        smooth_bound = data['smooth_bound']
        delta = data['delta']
        N = data['N']
        n_smooth = len(T_smooth)

        if n_smooth < 5:
            continue

        print(f"\n{'='*70}")
        print(f"  n={n}, N={N}, δ={delta:.3f}, |T_smooth|={n_smooth}")

        prefixes = greedy_prefixes(T_smooth, targets, prefix_fracs)

        for frac, T_prefix, NH_prefix in prefixes:
            tp = len(T_prefix)
            if tp < 2:
                continue

            # Compute tau distribution on targets
            tau = Counter()
            for k in T_prefix:
                for m in targets[k]:
                    tau[m] += 1

            E1 = sum(tau.values())
            E2 = sum(v * v for v in tau.values())
            max_tau = max(tau.values()) if tau else 0
            cs_actual = E1 * E1 / (tp * E2) if E2 > 0 else 0

            # Ford ratio
            ford_ratio = log(max(max_tau, 1)) / sqrt(log(max(n, 2)))

            # CS with capped tau
            cs_capped = {}
            for cap in cap_values:
                E2_cap = sum(min(v, cap) ** 2 for v in tau.values())
                cs_cap = E1 * E1 / (tp * E2_cap) if E2_cap > 0 else 0
                cs_capped[cap] = cs_cap

            # T_needed: the cap that would make CS = 1
            # CS@cap = E1^2 / (tp * sum min(tau,cap)^2) = 1
            # => sum min(tau,cap)^2 = E1^2 / tp
            target_E2 = E1 * E1 / tp if tp > 0 else 0

            # Binary search for smallest cap that gives CS >= 1
            cap_needed = max_tau
            for test_cap in range(1, max_tau + 1):
                E2_test = sum(min(v, test_cap) ** 2 for v in tau.values())
                if E1 * E1 / (tp * E2_test) >= 1.0 if E2_test > 0 else False:
                    cap_needed = test_cap
                    break

            # Tau distribution summary
            tau_dist = Counter(tau.values())
            top_taus = sorted(tau_dist.items(), reverse=True)[:5]

            # Average tau = E1 / |NH|
            nhp = len(NH_prefix)
            avg_tau = E1 / nhp if nhp > 0 else 0

            print(f"\n  frac={frac:.2f}: |T'|={tp}, |NH|={nhp}")
            print(f"    max_τ={max_tau}, avg_τ={avg_tau:.2f}, "
                  f"Ford ratio={ford_ratio:.4f}")
            print(f"    CS_actual={cs_actual:.4f}")
            for cap in cap_values:
                print(f"    CS@cap{cap}={cs_capped[cap]:.4f}", end="")
            print()
            print(f"    Cap needed for CS≥1: {cap_needed}")
            print(f"    Top τ values: {top_taus}")

            rec = {
                'n': n, 'frac': frac, 'tp': tp, 'nhp': nhp,
                'max_tau': max_tau, 'avg_tau': avg_tau,
                'ford_ratio': ford_ratio,
                'cs_actual': cs_actual,
                'cap_needed': cap_needed,
            }
            for cap in cap_values:
                rec[f'cs_cap{cap}'] = cs_capped[cap]
            results.append(rec)

    # Summary
    print(f"\n{'='*78}")
    print(f"  Z20e SUMMARY")
    print(f"{'='*78}")
    print(f"\n  {'n':>6} {'frac':>5} {'|T|':>5} "
          f"{'max_τ':>5} {'Ford':>6} {'CS':>6} "
          f"{'@2':>6} {'@3':>6} {'@5':>6} {'@10':>6} "
          f"{'cap▸1':>5}")
    for r in results:
        print(f"  {r['n']:>6} {r['frac']:>5.2f} {r['tp']:>5} "
              f"{r['max_tau']:>5} {r['ford_ratio']:>6.4f} "
              f"{r['cs_actual']:>6.4f} "
              f"{r.get('cs_cap2',0):>6.4f} "
              f"{r.get('cs_cap3',0):>6.4f} "
              f"{r.get('cs_cap5',0):>6.4f} "
              f"{r.get('cs_cap10',0):>6.4f} "
              f"{r['cap_needed']:>5}")

    # Check: CS@cap3 >= 1.0 for all configs
    if results:
        all_cap3_pass = all(r.get('cs_cap3', 0) >= 1.0 for r in results)
        min_cap3 = min(r.get('cs_cap3', 0) for r in results)
        print(f"\n  CS@cap3 ≥ 1.0 for ALL: {'YES' if all_cap3_pass else 'NO'}")
        print(f"  Min CS@cap3: {min_cap3:.4f}")

        # Ford ratio trend
        ford_by_n = defaultdict(list)
        for r in results:
            if abs(r['frac'] - 1.0) < 0.01:
                ford_by_n[r['n']].append(r['ford_ratio'])
        print(f"\n  Ford ratio trend (full pool):")
        for nv in sorted(ford_by_n):
            avg_ford = sum(ford_by_n[nv]) / len(ford_by_n[nv])
            print(f"    n={nv}: Ford ratio = {avg_ford:.4f}")

        # Cap needed trend
        cap_by_n = defaultdict(list)
        for r in results:
            if 0.45 <= r['frac'] <= 0.55:
                cap_by_n[r['n']].append(r['cap_needed'])
        print(f"\n  Cap needed trend (50% prefix):")
        for nv in sorted(cap_by_n):
            avg_cap = sum(cap_by_n[nv]) / len(cap_by_n[nv])
            print(f"    n={nv}: cap_needed = {avg_cap:.1f}")

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Erdős 710 — Experiments Z20a-Z20e: Literature Technique Tests')
    parser.add_argument('--n_values', type=str, default='500,1000,2000,5000,10000',
                        help='Comma-separated n values')
    parser.add_argument('--experiments', type=str, default='Z20a,Z20b,Z20c,Z20d,Z20e',
                        help='Which experiments to run')
    parser.add_argument('--sf', type=float, default=0.5,
                        help='s/N fraction (default 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    n_values = [int(x) for x in args.n_values.split(',')]
    exps = [x.strip().upper() for x in args.experiments.split(',')]

    random.seed(args.seed)

    print("ERDŐS 710 — EXPERIMENTS Z20a-Z20e: LITERATURE TECHNIQUE TESTS")
    print(f"n values: {n_values}")
    print(f"sf: {args.sf}")
    print(f"Experiments: {exps}")
    print(f"Random seed: {args.seed}")
    print(f"numpy available: {HAS_NUMPY}")
    print(f"{'=' * 78}")
    t0 = time.time()

    all_results = {}

    if 'Z20A' in exps:
        all_results['Z20a'] = experiment_Z20a(n_values, args.sf)
    if 'Z20B' in exps:
        all_results['Z20b'] = experiment_Z20b(n_values, args.sf)
    if 'Z20C' in exps:
        all_results['Z20c'] = experiment_Z20c(n_values, args.sf)
    if 'Z20D' in exps:
        all_results['Z20d'] = experiment_Z20d(n_values, args.sf)
    if 'Z20E' in exps:
        all_results['Z20e'] = experiment_Z20e(n_values, args.sf)

    elapsed = time.time() - t0
    print(f"\n{'='*78}")
    print(f"  TOTAL TIME: {elapsed:.1f}s")
    print(f"{'='*78}")

    # Final verdict
    print(f"\n  FINAL VERDICTS:")
    for exp_name, res in all_results.items():
        if not res:
            print(f"    {exp_name}: SKIPPED (no data)")
            continue
        if exp_name == 'Z20a':
            if any(r.get('hit', False) for r in res):
                print(f"    {exp_name} (Spectral): VIABLE — spectral gap sufficient")
            else:
                print(f"    {exp_name} (Spectral): DEAD — spectral gap too small")
        elif exp_name == 'Z20b':
            hardest = [r for r in res if 0.45 <= r['frac'] <= 0.95]
            if hardest and max(r.get('C_3', 0) for r in hardest) < 1.0:
                print(f"    {exp_name} (Green-Walker): VIABLE — C_3 < 1.0 at hardest prefix")
            else:
                print(f"    {exp_name} (Green-Walker): CLOSE/DEAD — C_3 ≥ 1.0")
        elif exp_name == 'Z20c':
            if all(r['ratio'] >= 1.0 for r in res):
                print(f"    {exp_name} (Mult. Table): CONFIRMED — |NH|/t ≥ 1.0 always")
            else:
                min_r = min(r['ratio'] for r in res)
                print(f"    {exp_name} (Mult. Table): VIOLATION — min ratio {min_r:.4f}")
        elif exp_name == 'Z20d':
            gaps = [r['gap'] for r in res if 0.45 <= r['frac'] <= 0.95]
            if gaps:
                max_gap = max(gaps)
                if max_gap <= 0.15:
                    print(f"    {exp_name} (Janson): VIABLE — gap bounded at {max_gap:.4f}")
                else:
                    print(f"    {exp_name} (Janson): CLOSE — gap={max_gap:.4f}")
            else:
                print(f"    {exp_name} (Janson): NO DATA")
        elif exp_name == 'Z20e':
            if all(r.get('cs_cap3', 0) >= 1.0 for r in res):
                print(f"    {exp_name} (Ford cap): VIABLE — CS@cap3 ≥ 1.0 always")
            else:
                min_c3 = min(r.get('cs_cap3', 0) for r in res)
                print(f"    {exp_name} (Ford cap): DEAD — min CS@cap3 = {min_c3:.4f}")
