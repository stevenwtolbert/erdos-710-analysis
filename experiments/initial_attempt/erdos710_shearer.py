#!/usr/bin/env python3
"""
Erdos 710 -- Shearer/Entropy Computational Experiments
Tests the Shearer inequality approach for Case B of Hall's condition.

Key bound: |N_H(S)| >= (prod |N_k|)^{1/Delta}
  where Delta = max_{m in N_H(S)} tau_S(m) = max right-degree.

Key inequality to validate: s * log(2M/n) >= Delta * log(s)

Usage: python erdos710_shearer.py [--full] [--quick]
  Default: n in [200, 500, 1000, 2000]
  --full:  extend to [3000, 5000]
  --quick: fast mode (n in [200, 500])
"""

import sys, time, math, random
from math import gcd, log, sqrt, exp, floor, ceil
from collections import defaultdict, Counter
from itertools import combinations

from erdos710_toolkit import (
    targets, target_L, C_TARGET, get_params,
    hopcroft_karp, largest_prime_factor,
    factorize, divisors_up_to, is_smooth,
)

random.seed(42)

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def build_adj_V_H(n, L):
    """Build adjacency dict for V = {1,...,n//2} -> H = (2n, n+L]."""
    V = list(range(1, n // 2 + 1))
    adj = {}
    for k in V:
        j0 = (2 * n) // k + 1
        j1 = (n + L) // k
        adj[k] = set(k * j for j in range(j0, j1 + 1))
    return V, adj


def compute_right_degrees(S, adj):
    """
    For subset S, compute:
      NH_S: neighborhood N_H(S) = union of adj[k] for k in S
      tau: dict mapping each m in NH_S to tau_S(m) = |{k in S : k | m}|
      Delta: max tau_S(m) over NH_S
    """
    tau = defaultdict(int)
    for k in S:
        for m in adj.get(k, set()):
            tau[m] += 1
    NH_S = set(tau.keys())
    Delta = max(tau.values()) if tau else 0
    return NH_S, dict(tau), Delta


def case_b_filter(S, M):
    """Check if S is a Case B subset: min(S) > M/(|S|+1)."""
    s = len(S)
    if s == 0:
        return False
    return min(S) > M / (s + 1)


def shearer_log_bound(S, adj, Delta):
    """Compute log of Shearer bound: (1/Delta) * sum log|N_k|.
    Returns None if Delta=0 or any |N_k|=0."""
    if Delta == 0:
        return None
    log_sum = 0.0
    for k in S:
        deg = len(adj.get(k, set()))
        if deg == 0:
            return None
        log_sum += log(deg)
    return log_sum / Delta


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 1: MAX RIGHT-DEGREE
# ═══════════════════════════════════════════════════════════════

def test_max_right_degree(n, V, adj, L):
    """Compute max_{m in H} tau_V(m) -- global max right-degree."""
    M = L - n

    tau = defaultdict(int)
    for k in V:
        for m in adj[k]:
            tau[m] += 1

    if not tau:
        return {'n': n, 'M': M, 'Delta_V': 0, 'avg_tau': 0, 'median_tau': 0,
                'worst_m': [], 'worst_factorization': []}

    Delta_V = max(tau.values())
    worst_m = sorted([m for m, t in tau.items() if t == Delta_V])[:3]
    tau_vals = sorted(tau.values())
    avg_tau = sum(tau_vals) / len(tau_vals)
    median_tau = tau_vals[len(tau_vals) // 2]

    tau_dist = Counter(tau.values())
    top5 = sorted(tau_dist.items(), key=lambda x: -x[0])[:5]

    return {
        'n': n, 'M': M, '|V|': len(V),
        'Delta_V': Delta_V,
        'avg_tau': avg_tau,
        'median_tau': median_tau,
        'worst_m': worst_m,
        'worst_factorization': [factorize(m) for m in worst_m],
        'tau_dist_top5': top5,
        'NH_size': len(tau),
    }


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 2: CASE B RIGHT-DEGREE
# ═══════════════════════════════════════════════════════════════

def test_case_b_right_degree(n, V, adj, L, num_samples=500):
    """For Case B subsets of various sizes, find max Delta_S."""
    M = L - n
    V_set = set(V)

    results = []
    sizes = [s for s in [2, 3, 5, 10, 20, 50, min(100, len(V) // 2)] if s <= len(V)]

    for s in sizes:
        best_Delta = 0
        best_S = None
        best_NH_size = 0

        # Strategy 1: Consecutive near n/2
        for a in range(max(1, n // 2 - s - 20), n // 2 - s + 2):
            S = [k for k in range(a, a + s) if k in V_set]
            if len(S) != s:
                continue
            if not case_b_filter(S, M):
                continue
            NH_S, tau, Delta = compute_right_degrees(S, adj)
            if Delta > best_Delta:
                best_Delta = Delta
                best_S = S
                best_NH_size = len(NH_S)

        # Strategy 2: Random subsets of top elements
        top_V = V[-min(60, len(V)):]
        for _ in range(num_samples):
            S = sorted(random.sample(top_V, min(s, len(top_V))))
            if not case_b_filter(S, M):
                continue
            NH_S, tau, Delta = compute_right_degrees(S, adj)
            if Delta > best_Delta:
                best_Delta = Delta
                best_S = S
                best_NH_size = len(NH_S)

        # Strategy 3: Evens near n/2
        evens = [k for k in V if k % 2 == 0 and k > n // 4]
        if len(evens) >= s:
            S = sorted(evens[-s:])
            if case_b_filter(S, M):
                NH_S, tau, Delta = compute_right_degrees(S, adj)
                if Delta > best_Delta:
                    best_Delta = Delta
                    best_S = S
                    best_NH_size = len(NH_S)

        results.append({
            's': s, 'Delta_S': best_Delta,
            'NH_size': best_NH_size,
            'S_head': best_S[:5] if best_S else None,
            'is_case_b': best_S is not None,
        })

    return {'n': n, 'M': M, 'results': results}


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 3: SHEARER VERIFICATION (CORE TEST)
# ═══════════════════════════════════════════════════════════════

def test_shearer_verification(n, V, adj, L, num_samples=1000):
    """Verify (prod |N_k|)^{1/Delta} >= s for all Case B subsets."""
    M = L - n
    V_set = set(V)

    results = []
    any_failure = False
    sizes = [s for s in [1, 2, 3, 5, 10, 20, 50, min(100, len(V) // 3)] if s <= len(V)]

    for s in sizes:
        worst_gap = float('inf')
        worst_info = None

        candidates = []

        # Consecutive near n/2
        for a in range(max(1, n // 2 - s - 30), n // 2 - s + 2):
            S = [k for k in range(a, a + s) if k in V_set]
            if len(S) == s and case_b_filter(S, M):
                candidates.append(S)

        # Random subsets of top elements
        top_V = [k for k in V if k > n // 4]
        for _ in range(num_samples):
            if len(top_V) < s:
                break
            S = sorted(random.sample(top_V, s))
            if case_b_filter(S, M):
                candidates.append(S)

        for S in candidates:
            NH_S, tau, Delta = compute_right_degrees(S, adj)
            if Delta == 0:
                continue

            shearer_log = shearer_log_bound(S, adj, Delta)
            if shearer_log is None:
                continue

            log_s = log(s) if s > 1 else 0
            gap = shearer_log - log_s  # positive = success

            # Key inequality form: s*log(2M/n) vs Delta*log(s)
            if 2 * M > n and s > 1:
                log_2M_n = log(2 * M / n)
                key_lhs = s * log_2M_n
                key_rhs = Delta * log_s
                key_ratio = key_lhs / key_rhs if key_rhs > 0 else float('inf')
            else:
                key_lhs = key_rhs = 0
                key_ratio = float('inf')

            actual_N = len(NH_S)

            if gap < worst_gap:
                worst_gap = gap
                worst_info = {
                    's': s, 'Delta': Delta, 'actual_N': actual_N,
                    'shearer_bound': exp(shearer_log),
                    'log_gap': gap,
                    'key_ratio': key_ratio,
                    'S_head': S[:5],
                }

        if worst_info:
            if worst_info['log_gap'] < 0:
                any_failure = True
            results.append(worst_info)
        elif s == 1:
            # s=1 always works
            results.append({
                's': 1, 'Delta': 1, 'actual_N': 0,
                'shearer_bound': 0, 'log_gap': float('inf'),
                'key_ratio': float('inf'), 'S_head': [],
            })

    return {
        'n': n, 'M': M,
        'by_size': results,
        'any_failure': any_failure,
    }


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 4: TIGHTEST SUBSETS
# ═══════════════════════════════════════════════════════════════

def test_tightest_subsets(n, V, adj, L, num_samples=2000):
    """Find subsets with smallest |N_H(S)|/|S| and analyze Shearer."""
    M = L - n
    V_set = set(V)

    results = []
    sizes = [s for s in [1, 2, 3, 5, 10, 20, 50] if s <= len(V)]

    for s in sizes:
        worst_ratio = float('inf')
        worst_S = None
        worst_Delta = 0
        worst_shearer_log = None

        # Random sampling of top elements
        top_V = V[-min(80, len(V)):]
        for _ in range(num_samples):
            if len(top_V) < s:
                break
            S = sorted(random.sample(top_V, s))
            NH_S, tau, Delta = compute_right_degrees(S, adj)
            r = len(NH_S) / s if s > 0 else float('inf')
            if r < worst_ratio:
                worst_ratio = r
                worst_S = S
                worst_Delta = Delta
                worst_shearer_log = shearer_log_bound(S, adj, Delta)

        # Consecutive near n/2
        for a in range(max(1, n // 2 - s - 30), n // 2 - s + 2):
            S = [k for k in range(a, a + s) if k in V_set]
            if len(S) != s:
                continue
            NH_S, tau, Delta = compute_right_degrees(S, adj)
            r = len(NH_S) / s
            if r < worst_ratio:
                worst_ratio = r
                worst_S = S
                worst_Delta = Delta
                worst_shearer_log = shearer_log_bound(S, adj, Delta)

        if worst_S is None:
            continue

        is_consec = (worst_S == list(range(worst_S[0], worst_S[0] + s)))
        gcd_all = worst_S[0]
        for k in worst_S[1:]:
            gcd_all = gcd(gcd_all, k)

        shearer_bound = exp(worst_shearer_log) if worst_shearer_log is not None else None
        shearer_ok = shearer_bound >= s if shearer_bound is not None else None

        results.append({
            's': s,
            'worst_ratio': worst_ratio,
            'actual_N': int(worst_ratio * s),
            'Delta': worst_Delta,
            'shearer_bound': shearer_bound,
            'shearer_ok': shearer_ok,
            'is_consecutive': is_consec,
            'gcd_all': gcd_all,
            'S_head': worst_S[:5],
        })

    return {'n': n, 'M': M, 'results': results}


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 5: SHEARER VS ACTUAL
# ═══════════════════════════════════════════════════════════════

def test_shearer_vs_actual(n, V, adj, L, num_samples=1000):
    """Compare Shearer bound to actual |N_H(S)|."""
    M = L - n

    results = []
    sizes = [s for s in [2, 5, 10, 20, 50] if s <= len(V)]

    for s in sizes:
        tightness_ratios = []

        top_V = V[-min(60, len(V)):]
        candidates = []

        # Consecutive near n/2
        for a in range(max(1, n // 2 - s - 20), n // 2 - s + 2):
            S = [k for k in range(a, a + s) if 1 <= k <= n // 2]
            if len(S) == s:
                candidates.append(S)

        # Random
        for _ in range(num_samples):
            if len(top_V) < s:
                break
            S = sorted(random.sample(top_V, s))
            candidates.append(S)

        for S in candidates:
            NH_S, tau, Delta = compute_right_degrees(S, adj)
            if Delta == 0 or len(NH_S) == 0:
                continue
            sl = shearer_log_bound(S, adj, Delta)
            if sl is None:
                continue
            shearer_bound = exp(sl)
            actual_N = len(NH_S)
            tightness_ratios.append(shearer_bound / actual_N)

        if tightness_ratios:
            results.append({
                's': s,
                'avg_tightness': sum(tightness_ratios) / len(tightness_ratios),
                'min_tightness': min(tightness_ratios),
                'max_tightness': max(tightness_ratios),
                'count': len(tightness_ratios),
            })

    return {'n': n, 'M': M, 'results': results}


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 6: KEY INEQUALITY SCALING (CRITICAL)
# ═══════════════════════════════════════════════════════════════

def test_key_inequality_scaling(ns, adj_cache, num_samples=500):
    """Test s*log(2M/n) >= Delta*log(s) across different n."""
    all_results = []

    for n in ns:
        V, adj, L = adj_cache[n]
        M = L - n
        V_set = set(V)

        if 2 * M <= n:
            continue
        log_2M_n = log(2 * M / n)

        sizes = [s for s in [2, 5, 10, 20, 50, min(100, len(V) // 3)]
                 if s <= len(V) and s > 1]

        for s in sizes:
            worst_Delta = 0

            # Consecutive near n/2
            for a in range(max(1, n // 2 - s - 20), n // 2 - s + 2):
                S = [k for k in range(a, a + s) if k in V_set]
                if len(S) != s:
                    continue
                if not case_b_filter(S, M):
                    continue
                _, _, Delta = compute_right_degrees(S, adj)
                worst_Delta = max(worst_Delta, Delta)

            # Random from top
            top_V = V[-min(60, len(V)):]
            for _ in range(num_samples):
                if len(top_V) < s:
                    break
                S = sorted(random.sample(top_V, s))
                if not case_b_filter(S, M):
                    continue
                _, _, Delta = compute_right_degrees(S, adj)
                worst_Delta = max(worst_Delta, Delta)

            if worst_Delta == 0:
                continue

            lhs = s * log_2M_n
            rhs = worst_Delta * log(s)
            ratio = lhs / rhs if rhs > 0 else float('inf')

            all_results.append({
                'n': n, 's': s, 'Delta': worst_Delta,
                '2M/n': 2 * M / n,
                'LHS': lhs, 'RHS': rhs,
                'ratio': ratio,
                'passes': ratio >= 1.0,
            })

    return all_results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 7: NEAR-VIOLATOR STRUCTURE
# ═══════════════════════════════════════════════════════════════

def test_near_violator_structure(n, V, adj, L, num_samples=2000):
    """Analyze structure of tightest subsets."""
    M = L - n
    V_set = set(V)

    analysis = []
    sizes = [s for s in [2, 3, 5, 10, 20] if s <= len(V)]

    for s in sizes:
        worst_ratio = float('inf')
        worst_S = None

        top_V = V[-min(80, len(V)):]
        for _ in range(num_samples):
            if len(top_V) < s:
                break
            S = sorted(random.sample(top_V, s))
            NH_S, _, _ = compute_right_degrees(S, adj)
            r = len(NH_S) / s
            if r < worst_ratio:
                worst_ratio = r
                worst_S = S

        # Consecutive
        for a in range(max(1, n // 2 - s - 30), n // 2 - s + 2):
            S = [k for k in range(a, a + s) if k in V_set]
            if len(S) != s:
                continue
            NH_S, _, _ = compute_right_degrees(S, adj)
            r = len(NH_S) / s
            if r < worst_ratio:
                worst_ratio = r
                worst_S = S

        if worst_S is None:
            continue

        # Structure analysis
        is_consec = (worst_S == list(range(worst_S[0], worst_S[0] + s)))
        gcd_all = worst_S[0]
        for k in worst_S[1:]:
            gcd_all = gcd(gcd_all, k)

        pairwise_gcds = []
        for i in range(len(worst_S)):
            for j in range(i + 1, len(worst_S)):
                pairwise_gcds.append(gcd(worst_S[i], worst_S[j]))
        avg_gcd = sum(pairwise_gcds) / len(pairwise_gcds) if pairwise_gcds else 0
        frac_coprime = sum(1 for g in pairwise_gcds if g == 1) / max(1, len(pairwise_gcds))

        share_2 = sum(1 for k in worst_S if k % 2 == 0) / s
        share_3 = sum(1 for k in worst_S if k % 3 == 0) / s

        NH_S, tau, Delta = compute_right_degrees(worst_S, adj)
        sl = shearer_log_bound(worst_S, adj, Delta)
        shearer_bound = exp(sl) if sl is not None else None

        tau_dist = Counter(tau.values())

        analysis.append({
            's': s, 'worst_ratio': worst_ratio,
            'is_consecutive': is_consec,
            'gcd_all': gcd_all,
            'avg_pairwise_gcd': avg_gcd,
            'frac_coprime': frac_coprime,
            'share_2': share_2, 'share_3': share_3,
            'Delta': Delta,
            'shearer_bound': shearer_bound,
            'actual_N': len(NH_S),
            'tau_dist': dict(sorted(tau_dist.items())),
            'S': worst_S,
        })

    return {'n': n, 'M': M, 'analysis': analysis}


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 8: FORD DIVISOR DISTRIBUTION
# ═══════════════════════════════════════════════════════════════

def test_ford_divisor_distribution(n, V, adj, L):
    """Count m in H with divisors in various size ranges."""
    M = L - n
    H_range = range(2 * n + 1, n + L + 1)

    boundaries = [0, int(n ** 0.25), int(n ** 0.5), n // 4, n // 2]
    range_names = []
    for i in range(len(boundaries) - 1):
        range_names.append(f"({boundaries[i]},{boundaries[i+1]}]")

    counts = [0] * len(range_names)
    total_div_counts = [0] * len(range_names)

    for m in H_range:
        divs = divisors_up_to(m, n // 2)
        for i in range(len(range_names)):
            lo, hi = boundaries[i], boundaries[i + 1]
            in_range = [d for d in divs if lo < d <= hi]
            if in_range:
                counts[i] += 1
            total_div_counts[i] += len(in_range)

    results = []
    for i, name in enumerate(range_names):
        results.append({
            'range': name,
            'count': counts[i],
            'fraction': counts[i] / max(1, M),
            'avg_divs': total_div_counts[i] / max(1, M),
        })

    return {'n': n, 'M': M, 'ranges': results}


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 9: ITERATIVE BOOTSTRAP
# ═══════════════════════════════════════════════════════════════

def test_iterative_bootstrap(n, V, adj, L):
    """Check if near-violating sets can structurally exist."""
    M = L - n

    if 2 * M <= n:
        return {'n': n, 'M': M, 'results': []}

    log_2M_n = log(2 * M / n)

    # Precompute tau_V for all m in H
    tau_v = defaultdict(int)
    for k in V:
        for m in adj[k]:
            tau_v[m] += 1

    results = []
    for s in [5, 10, 20, 50]:
        if s > len(V):
            break

        log_s = log(s)
        Delta_required = s * log_2M_n / log_s

        # How many m have tau_V(m) >= Delta_required?
        m_above = sum(1 for t in tau_v.values() if t >= Delta_required)

        # Which k can participate? (k divides some high-tau m)
        high_tau_m = {m for m, t in tau_v.items() if t >= Delta_required}
        k_in_support = set()
        for k in V:
            if adj[k] & high_tau_m:
                k_in_support.add(k)

        # Degree stats for k near n/2
        k_near_half = [k for k in V if k > n // 3]
        degs_near_half = [len(adj[k]) for k in k_near_half]
        min_deg = min(degs_near_half) if degs_near_half else 0
        max_deg = max(degs_near_half) if degs_near_half else 0

        results.append({
            's': s,
            'Delta_req': Delta_required,
            'm_above': m_above,
            'k_support': len(k_in_support),
            'support_ok': len(k_in_support) >= s,
            'min_deg_n/2': min_deg,
            'max_deg_n/2': max_deg,
            'feasible': len(k_in_support) >= s and m_above > 0,
        })

    return {'n': n, 'M': M, 'results': results}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    full_mode = '--full' in sys.argv
    quick_mode = '--quick' in sys.argv
    t0 = time.time()

    eps = 0.05

    if quick_mode:
        ns = [200, 500]
    else:
        ns = [200, 500, 1000, 2000]
    if full_mode:
        ns += [3000, 5000]

    print("=" * 78)
    print("  ERDOS 710 -- SHEARER/ENTROPY COMPUTATIONAL EXPERIMENTS")
    print("=" * 78)
    print(f"  n values: {ns}")
    print(f"  eps = {eps}")

    # Build adjacency cache
    adj_cache = {}
    for n in ns:
        L = target_L(n, eps)
        V, adj = build_adj_V_H(n, L)
        adj_cache[n] = (V, adj, L)
        print(f"  n={n}: |V|={len(V)}, L={L}, M={L-n}, 2M/n={2*(L-n)/n:.4f}")

    # ─── EXP 1: Max Right-Degree ───
    print("\n" + "=" * 78)
    print("  EXP 1: Max Right-Degree tau_V(m) over H")
    print("=" * 78)
    print(f"  {'n':>6} {'Delta_V':>8} {'avg_tau':>8} {'med_tau':>8} "
          f"{'|NH|':>6} {'worst_m':>10} {'factorization':>30}")

    for n in ns:
        V, adj, L = adj_cache[n]
        r = test_max_right_degree(n, V, adj, L)
        wm = r['worst_m'][0] if r['worst_m'] else '-'
        wf = r['worst_factorization'][0] if r['worst_factorization'] else '-'
        wf_str = '*'.join(f"{p}^{e}" if e > 1 else str(p) for p, e in wf) if isinstance(wf, list) else str(wf)
        print(f"  {n:>6} {r['Delta_V']:>8} {r['avg_tau']:>8.2f} {r['median_tau']:>8} "
              f"{r['NH_size']:>6} {wm:>10} {wf_str:>30}")

    # ─── EXP 2: Case B Right-Degree ───
    print("\n" + "=" * 78)
    print("  EXP 2: Case B Right-Degree (max Delta_S for Case B subsets)")
    print("=" * 78)

    for n in ns:
        V, adj, L = adj_cache[n]
        r = test_case_b_right_degree(n, V, adj, L)
        M = r['M']
        print(f"\n  n={n}, M={M}:")
        print(f"    {'s':>4} {'Delta_S':>8} {'|NH(S)|':>8} {'case_B':>7} {'S_head':>30}")
        for row in r['results']:
            sh = str(row['S_head']) if row['S_head'] else '-'
            cb = 'yes' if row['is_case_b'] else 'NO'
            print(f"    {row['s']:>4} {row['Delta_S']:>8} {row['NH_size']:>8} {cb:>7} {sh:>30}")

    # ─── EXP 3: Shearer Verification ───
    print("\n" + "=" * 78)
    print("  EXP 3: Shearer Inequality Verification (CORE TEST)")
    print("  Need: (prod |N_k|)^{1/Delta} >= s for all Case B subsets")
    print("=" * 78)

    global_failure = False
    for n in ns:
        V, adj, L = adj_cache[n]
        r = test_shearer_verification(n, V, adj, L)
        if r['any_failure']:
            global_failure = True
        M = r['M']
        print(f"\n  n={n}, M={M}:")
        print(f"    {'s':>4} {'Delta':>6} {'actual_N':>9} {'shearer':>9} "
              f"{'log_gap':>9} {'key_ratio':>10} {'status':>8}")
        for row in r['by_size']:
            status = "FAIL" if row['log_gap'] < 0 else "ok"
            kr = f"{row['key_ratio']:.4f}" if row['key_ratio'] < 1000 else "inf"
            sb = f"{row['shearer_bound']:.2f}" if row['shearer_bound'] < 1e6 else "large"
            print(f"    {row['s']:>4} {row['Delta']:>6} {row['actual_N']:>9} {sb:>9} "
                  f"{row['log_gap']:>9.4f} {kr:>10} {status:>8}")

    # ─── EXP 4: Tightest Subsets ───
    print("\n" + "=" * 78)
    print("  EXP 4: Tightest Subsets (min |N_H(S)|/|S|) + Shearer")
    print("=" * 78)

    for n in ns:
        V, adj, L = adj_cache[n]
        r = test_tightest_subsets(n, V, adj, L)
        M = r['M']
        print(f"\n  n={n}, M={M}:")
        print(f"    {'s':>4} {'|N|/s':>8} {'|N|':>6} {'Delta':>6} "
              f"{'shearer':>9} {'sh>=s':>6} {'consec':>7} {'gcd':>5}")
        for row in r['results']:
            sb = f"{row['shearer_bound']:.2f}" if row['shearer_bound'] is not None else '-'
            sok = 'YES' if row['shearer_ok'] else ('NO' if row['shearer_ok'] is not None else '-')
            con = 'yes' if row['is_consecutive'] else 'no'
            print(f"    {row['s']:>4} {row['worst_ratio']:>8.3f} {row['actual_N']:>6} "
                  f"{row['Delta']:>6} {sb:>9} {sok:>6} {con:>7} {row['gcd_all']:>5}")

    # ─── EXP 5: Shearer vs Actual ───
    print("\n" + "=" * 78)
    print("  EXP 5: Shearer Bound Tightness (shearer/actual)")
    print("  Values near 1.0 = tight bound; near 0 = very loose")
    print("=" * 78)

    for n in ns:
        V, adj, L = adj_cache[n]
        r = test_shearer_vs_actual(n, V, adj, L)
        M = r['M']
        print(f"\n  n={n}, M={M}:")
        print(f"    {'s':>4} {'avg':>8} {'min':>8} {'max':>8} {'count':>6}")
        for row in r['results']:
            print(f"    {row['s']:>4} {row['avg_tightness']:>8.4f} "
                  f"{row['min_tightness']:>8.4f} {row['max_tightness']:>8.4f} "
                  f"{row['count']:>6}")

    # ─── EXP 6: Key Inequality Scaling ───
    print("\n" + "=" * 78)
    print("  EXP 6: Key Inequality  s*log(2M/n) vs Delta*log(s)")
    print("  CRITICAL: ratio must be >= 1.0 for Shearer to work")
    print("=" * 78)

    scaling = test_key_inequality_scaling(ns, adj_cache)

    # Pivot table
    s_vals = sorted(set(r['s'] for r in scaling))
    n_vals = sorted(set(r['n'] for r in scaling))
    lookup = {(r['n'], r['s']): r for r in scaling}

    header = f"    {'s':>4} " + " ".join(f"{'n='+str(n):>12}" for n in n_vals)
    print(header)
    any_scaling_fail = False
    for s in s_vals:
        row_parts = []
        for n in n_vals:
            r = lookup.get((n, s))
            if r:
                mark = " ***" if r['ratio'] < 1.0 else ""
                if r['ratio'] < 1.0:
                    any_scaling_fail = True
                row_parts.append(f"{r['ratio']:>8.4f}{mark}")
            else:
                row_parts.append(f"{'--':>12}")
        print(f"    {s:>4} " + " ".join(row_parts))

    # ─── EXP 7: Near-Violator Structure ───
    print("\n" + "=" * 78)
    print("  EXP 7: Structure of Near-Violators")
    print("=" * 78)

    for n in ns[:3]:  # skip largest for speed
        V, adj, L = adj_cache[n]
        r = test_near_violator_structure(n, V, adj, L)
        M = r['M']
        print(f"\n  n={n}, M={M}:")
        for a in r['analysis']:
            s = a['s']
            print(f"    s={s}: ratio={a['worst_ratio']:.3f}, "
                  f"consec={a['is_consecutive']}, gcd={a['gcd_all']}, "
                  f"coprime%={a['frac_coprime']*100:.0f}%, "
                  f"share2={a['share_2']*100:.0f}%, share3={a['share_3']*100:.0f}%")
            print(f"      Delta={a['Delta']}, shearer={a['shearer_bound']:.2f}" if a['shearer_bound'] else
                  f"      Delta={a['Delta']}, shearer=N/A")
            print(f"      S={a['S']}")
            # Compact tau distribution
            td = a['tau_dist']
            td_str = ", ".join(f"{t}:{c}" for t, c in sorted(td.items())[:6])
            print(f"      tau_dist: {{{td_str}}}")

    # ─── EXP 8: Ford Divisor Distribution ───
    print("\n" + "=" * 78)
    print("  EXP 8: Ford Divisor Distribution in H")
    print("=" * 78)

    for n in ns[:3]:  # skip largest for speed
        V, adj, L = adj_cache[n]
        r = test_ford_divisor_distribution(n, V, adj, L)
        M = r['M']
        print(f"\n  n={n}, M={M}:")
        print(f"    {'range':>15} {'count':>6} {'frac':>8} {'avg_divs':>10}")
        for row in r['ranges']:
            print(f"    {row['range']:>15} {row['count']:>6} "
                  f"{row['fraction']:>8.4f} {row['avg_divs']:>10.4f}")

    # ─── EXP 9: Iterative Bootstrap ───
    print("\n" + "=" * 78)
    print("  EXP 9: Iterative Bootstrap Feasibility")
    print("  Delta_req = s*log(2M/n)/log(s) for near-violators to exist")
    print("=" * 78)

    for n in ns:
        V, adj, L = adj_cache[n]
        r = test_iterative_bootstrap(n, V, adj, L)
        M = r['M']
        print(f"\n  n={n}, M={M}:")
        print(f"    {'s':>4} {'Delta_req':>10} {'m_above':>8} "
              f"{'k_support':>10} {'min_deg':>8} {'feasible':>9}")
        for row in r['results']:
            feas = 'YES' if row['feasible'] else 'no'
            print(f"    {row['s']:>4} {row['Delta_req']:>10.2f} {row['m_above']:>8} "
                  f"{row['k_support']:>10} {row['min_deg_n/2']:>8} {feas:>9}")

    # ═══ SUMMARY ═══
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)

    if global_failure:
        print("  *** SHEARER VERIFICATION: FAILURES DETECTED ***")
        print("  Some Case B subsets have Shearer bound < |S|.")
        print("  The raw Shearer approach may need refinement.")
    else:
        print("  Shearer verification: PASSED for all tested Case B subsets")

    if any_scaling_fail:
        print("  *** KEY INEQUALITY: FAILURES DETECTED ***")
        print("  s*log(2M/n) < Delta*log(s) for some (n, s) pairs.")
    else:
        print("  Key inequality: PASSED for all tested (n, s) pairs")

    print(f"\n  Total time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
