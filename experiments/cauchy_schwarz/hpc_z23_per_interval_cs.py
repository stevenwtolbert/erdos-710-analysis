#!/usr/bin/env python3
"""
ERDŐS 710 — EXPERIMENTS Z23a–Z23c: PER-INTERVAL CAUCHY-SCHWARZ

The breakthrough from Z22a: dyadic decomposition of S₊ into intervals
[2^j, 2^{j+1}) gives PERFECT matching per-interval at ALL n tested.

The key question: can we prove Hall per-interval using Cauchy-Schwarz?
CS fails on ALL of S₊ (dips to 0.83), but per-interval the structure
is much better (expansion ratios 2-50+).

Z23a: Per-Interval Adversarial CS
  - Within each dyadic interval, build the GREEDY MINIMIZER
  - Compute CS for adversarial prefixes within the interval
  - Does CS ≥ 1 per-interval, even for the hardest subset?

Z23b: Per-Interval Minimum Expansion Ratio
  - For each interval, find the adversarial subset that minimizes |NH|/|T'|
  - Track how this minimum ratio scales with n
  - Key: does min_ratio per-interval stay bounded above 1?

Z23c: Per-Interval Degree Analysis for Analytic Proof
  - For each interval: min_deg, avg_deg, |I_j|, |smooth_in_interval|
  - Compute: min_deg vs |I_j| (is degree enough to prove Hall?)
  - Compute: forced_gcd_min, max elements per GCD group
  - Determine which proof technique works for each interval
"""

import sys
import time
import argparse
import random
from math import gcd, log, sqrt, exp, floor, ceil, log2
from collections import defaultdict, Counter

C_TARGET = 2 / sqrt(exp(1))
EPS = 0.05


def target_L(n, eps=EPS):
    if n < 3:
        return 3 * n
    return int((C_TARGET + eps) * n * sqrt(log(n) / log(log(n))))


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


def compute_targets(k, n, L):
    j0 = (2 * n) // k + 1
    j1 = (n + L) // k
    return set(k * j for j in range(j0, j1 + 1))


def hopcroft_karp(graph, U, V_set):
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


def build_smooth_data(n, sf=0.5):
    """Build smooth pool data."""
    L = target_L(n)
    if L <= n:
        return None
    M = L - n
    N = n // 2
    if N < 10:
        return None
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

    pool_smooth = [k for k in pool if lpf_cache[k] <= sqrt_nL]

    return {
        'n': n, 'L': L, 'M': M, 'N': N, 'delta': delta,
        'nL': nL, 'sqrt_nL': sqrt_nL, 'smooth_bound': smooth_bound,
        's': s, 'pool_smooth': pool_smooth,
        'targets': targets, 'target_to_pool': target_to_pool,
        'lpf_cache': lpf_cache,
    }


def get_dyadic_intervals(sqrt_nL, N):
    """Get dyadic intervals covering (√(n+L), N]."""
    lo = int(sqrt_nL) + 1
    hi = N
    intervals = []
    j = int(log2(lo)) if lo > 0 else 0
    while (1 << j) <= hi:
        ivl_lo = max(lo, 1 << j)
        ivl_hi = min(hi, (1 << (j + 1)) - 1)
        if ivl_lo <= ivl_hi:
            intervals.append((j, ivl_lo, ivl_hi))
        j += 1
    return intervals


def build_greedy_minimizer_for_interval(elements, targets, s=None):
    """Build greedy |NH|-minimizer for a set of elements."""
    if s is None:
        s = len(elements)
    s = min(s, len(elements))

    # Build target_to_pool for this interval
    t2p = {}
    for k in elements:
        for h in targets.get(k, set()):
            if h not in t2p:
                t2p[h] = []
            t2p[h].append(k)

    T = []
    NH = set()
    rem = set(elements)
    new_count = {k: len(targets.get(k, set())) for k in elements}

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
        newly_covered = targets.get(best_k, set()) - NH
        NH |= newly_covered
        for h in newly_covered:
            for k2 in t2p.get(h, []):
                if k2 in rem:
                    new_count[k2] -= 1

    return T, NH


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT Z23a: PER-INTERVAL ADVERSARIAL CAUCHY-SCHWARZ
# ═══════════════════════════════════════════════════════════════

def experiment_Z23a(n_values, sf=0.5):
    """
    Within each dyadic interval, build greedy minimizer and test CS.
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z23a: PER-INTERVAL ADVERSARIAL CAUCHY-SCHWARZ")
    print("=" * 78)
    print("  Key question: does CS ≥ 1 per-interval for adversarial subsets?")

    prefix_fracs = [0.25, 0.50, 0.75, 0.90, 1.00]
    results = []

    for n in n_values:
        data = build_smooth_data(n, sf)
        if data is None:
            continue

        N = data['N']
        nL = data['nL']
        sqrt_nL = data['sqrt_nL']
        delta = data['delta']
        targets = data['targets']
        pool_smooth = data['pool_smooth']

        s = data['s']
        pool_smooth_set = set(pool_smooth)

        intervals = get_dyadic_intervals(sqrt_nL, N)

        print(f"\n{'='*70}")
        print(f"  n={n}, N={N}, √(n+L)={sqrt_nL:.1f}, δ={delta:.3f}")

        for jj, ivl_lo, ivl_hi in intervals:
            # Elements of S₊ in this interval (smooth, in pool)
            I_j = sorted([k for k in pool_smooth if ivl_lo <= k <= ivl_hi
                          and k > int(sqrt_nL)])
            if len(I_j) < 3:
                continue

            # Restrict targets to those in the original target range
            ivl_targets = {}
            for k in I_j:
                ivl_targets[k] = targets.get(k, set())

            # Build greedy minimizer WITHIN this interval
            T_greedy, NH_greedy = build_greedy_minimizer_for_interval(
                I_j, ivl_targets)

            # Now test CS for adversarial prefixes within this interval
            t_total = len(T_greedy)

            print(f"\n  Interval [{ivl_lo},{ivl_hi}]: |I_j|={t_total}, "
                  f"|NH_full|={len(NH_greedy)}, "
                  f"ratio_full={len(NH_greedy)/t_total:.2f}")

            min_cs = float('inf')
            min_ratio = float('inf')

            for frac in prefix_fracs:
                size = max(1, int(frac * t_total))
                if size > t_total:
                    size = t_total
                if size < 2:
                    continue

                T_prefix = T_greedy[:size]
                NH_prefix = set()
                for k in T_prefix:
                    NH_prefix |= ivl_targets.get(k, set())

                tp = len(T_prefix)
                nhp = len(NH_prefix)
                ratio = nhp / tp

                # Compute CS
                tau_prefix = Counter()
                for k in T_prefix:
                    for m in ivl_targets.get(k, set()):
                        tau_prefix[m] += 1
                E1 = sum(tau_prefix.values())
                E2 = sum(v * v for v in tau_prefix.values())
                cs = E1 * E1 / (tp * E2) if E2 > 0 else 0

                min_cs = min(min_cs, cs)
                min_ratio = min(min_ratio, ratio)

                flag = "OK" if cs >= 1.0 else "FAIL"
                print(f"    frac={frac:.2f}: t={tp:>5}, |NH|={nhp:>6}, "
                      f"ratio={ratio:.3f}, CS={cs:.4f} {flag}")

                results.append({
                    'n': n, 'j': jj, 'lo': ivl_lo, 'hi': ivl_hi,
                    'frac': frac, 'tp': tp, 'nhp': nhp,
                    'ratio': ratio, 'cs': cs,
                })

            cs_ok = "ALL CS ≥ 1" if min_cs >= 1.0 else f"MIN CS = {min_cs:.4f}"
            print(f"    => {cs_ok}, min ratio = {min_ratio:.3f}")

    # Global summary
    print(f"\n{'='*78}")
    print(f"  Z23a SUMMARY: Per-interval CS results")
    print(f"{'='*78}")

    for n in n_values:
        nr = [r for r in results if r['n'] == n]
        if not nr:
            continue

        # Group by interval
        intervals_seen = sorted(set((r['lo'], r['hi']) for r in nr))
        print(f"\n  n={n}:")
        print(f"  {'Interval':>16} {'min_CS':>8} {'min_ratio':>10} "
              f"{'CS≥1?':>6} {'worst_frac':>10}")

        all_pass = True
        for lo, hi in intervals_seen:
            ir = [r for r in nr if r['lo'] == lo and r['hi'] == hi]
            min_cs = min(r['cs'] for r in ir)
            min_ratio = min(r['ratio'] for r in ir)
            worst = min(ir, key=lambda r: r['cs'])
            ok = min_cs >= 1.0
            if not ok:
                all_pass = False
            print(f"  [{lo:>5},{hi:>5}] {min_cs:>8.4f} {min_ratio:>10.3f} "
                  f"{'YES' if ok else 'NO':>6} {worst['frac']:>10.2f}")

        if all_pass:
            print(f"  => ALL INTERVALS PASS CS! Analytic proof via CS per-interval!")
        else:
            failing = [(lo, hi) for lo, hi in intervals_seen
                       if min(r['cs'] for r in nr if r['lo'] == lo and r['hi'] == hi) < 1.0]
            print(f"  => {len(failing)} intervals fail CS")

    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT Z23b: PER-INTERVAL MINIMUM EXPANSION SCALING
# ═══════════════════════════════════════════════════════════════

def experiment_Z23b(n_values, sf=0.5):
    """
    For each interval, find the minimum |NH|/|T'| over greedy prefixes.
    Track how the per-interval minimum ratio scales with n.
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z23b: PER-INTERVAL MINIMUM EXPANSION SCALING")
    print("=" * 78)

    fine_fracs = [i / 20.0 for i in range(1, 21)]  # 0.05, 0.10, ..., 1.00
    results = []

    for n in n_values:
        data = build_smooth_data(n, sf)
        if data is None:
            continue

        N = data['N']
        nL = data['nL']
        sqrt_nL = data['sqrt_nL']
        delta = data['delta']
        targets = data['targets']
        pool_smooth = data['pool_smooth']

        intervals = get_dyadic_intervals(sqrt_nL, N)

        print(f"\n{'='*70}")
        print(f"  n={n}, N={N}, √(n+L)={sqrt_nL:.1f}, δ={delta:.3f}")
        print(f"\n  {'Interval':>16} {'|I_j|':>6} {'min_ratio':>10} "
              f"{'at_frac':>8} {'min_CS':>8} {'at_frac':>8}")

        for jj, ivl_lo, ivl_hi in intervals:
            I_j = sorted([k for k in pool_smooth if ivl_lo <= k <= ivl_hi
                          and k > int(sqrt_nL)])
            if len(I_j) < 3:
                continue

            ivl_targets = {k: targets.get(k, set()) for k in I_j}
            T_greedy, NH_greedy = build_greedy_minimizer_for_interval(I_j, ivl_targets)
            t_total = len(T_greedy)

            min_ratio = float('inf')
            min_ratio_frac = 0
            min_cs = float('inf')
            min_cs_frac = 0

            for frac in fine_fracs:
                size = max(2, int(frac * t_total))
                if size > t_total:
                    size = t_total

                T_prefix = T_greedy[:size]
                NH_prefix = set()
                for k in T_prefix:
                    NH_prefix |= ivl_targets.get(k, set())

                tp = len(T_prefix)
                nhp = len(NH_prefix)
                ratio = nhp / tp

                tau_prefix = Counter()
                for k in T_prefix:
                    for m in ivl_targets.get(k, set()):
                        tau_prefix[m] += 1
                E1 = sum(tau_prefix.values())
                E2 = sum(v * v for v in tau_prefix.values())
                cs = E1 * E1 / (tp * E2) if E2 > 0 else 0

                if ratio < min_ratio:
                    min_ratio = ratio
                    min_ratio_frac = frac
                if cs < min_cs:
                    min_cs = cs
                    min_cs_frac = frac

            print(f"  [{ivl_lo:>5},{ivl_hi:>5}] {t_total:>6} {min_ratio:>10.4f} "
                  f"{min_ratio_frac:>8.2f} {min_cs:>8.4f} {min_cs_frac:>8.2f}")

            results.append({
                'n': n, 'j': jj, 'lo': ivl_lo, 'hi': ivl_hi,
                'count': t_total, 'min_ratio': min_ratio,
                'min_ratio_frac': min_ratio_frac,
                'min_cs': min_cs, 'min_cs_frac': min_cs_frac,
            })

    # Summary: track worst interval per n
    print(f"\n{'='*78}")
    print(f"  Z23b SUMMARY: Per-interval worst-case scaling")
    print(f"{'='*78}")
    print(f"\n  {'n':>6} {'worst_ratio':>12} {'worst_CS':>10} "
          f"{'worst_interval':>20} {'|I|':>6}")
    for n in n_values:
        nr = [r for r in results if r['n'] == n]
        if not nr:
            continue
        worst_r = min(nr, key=lambda r: r['min_ratio'])
        worst_c = min(nr, key=lambda r: r['min_cs'])
        print(f"  {n:>6} {worst_r['min_ratio']:>12.4f} {worst_c['min_cs']:>10.4f} "
              f"[{worst_r['lo']},{worst_r['hi']}]{' ':>5} {worst_r['count']:>6}")

    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT Z23c: PER-INTERVAL PROOF CHARACTERIZATION
# ═══════════════════════════════════════════════════════════════

def experiment_Z23c(n_values, sf=0.5):
    """
    For each interval, determine which proof technique works:
    - Degree surplus: min_deg ≥ |I_j| (trivial Hall)
    - CS ≥ 1 at all densities
    - GCD group bound: max group size < average degree
    - None: needs further analysis
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z23c: PER-INTERVAL PROOF CHARACTERIZATION")
    print("=" * 78)

    results = []

    for n in n_values:
        data = build_smooth_data(n, sf)
        if data is None:
            continue

        N = data['N']
        nL = data['nL']
        sqrt_nL = data['sqrt_nL']
        delta = data['delta']
        targets = data['targets']
        pool_smooth = data['pool_smooth']

        intervals = get_dyadic_intervals(sqrt_nL, N)

        print(f"\n{'='*70}")
        print(f"  n={n}, N={N}, √(n+L)={sqrt_nL:.1f}, δ={delta:.3f}")
        print(f"\n  {'Interval':>16} {'|I|':>5} {'min_d':>6} {'avg_d':>6} "
              f"{'deg≥|I|':>7} {'fgcd':>6} {'maxgrp':>6} {'CS_min':>8} "
              f"{'technique':>12}")

        for jj, ivl_lo, ivl_hi in intervals:
            I_j = sorted([k for k in pool_smooth if ivl_lo <= k <= ivl_hi
                          and k > int(sqrt_nL)])
            if len(I_j) < 2:
                continue

            ivl_targets = {k: targets.get(k, set()) for k in I_j}

            # Degree stats
            degs = [len(ivl_targets.get(k, set())) for k in I_j]
            min_d = min(degs)
            avg_d = sum(degs) / len(degs)
            t_j = len(I_j)

            # Degree surplus test
            deg_ok = min_d >= t_j

            # Forced GCD
            forced_gcd = ivl_lo * ivl_lo / nL

            # Max group size: elements sharing a large factor
            # For elements in [X, 2X], sharing target requires gcd ≥ X²/(n+L)
            # Number of multiples of gcd in [X, 2X] ≈ X/gcd ≈ (n+L)/X
            if forced_gcd >= 2:
                max_group = min(t_j, int(ceil((ivl_hi - ivl_lo + 1) / forced_gcd)) + 1)
            else:
                max_group = t_j

            # CS test: build greedy minimizer and check CS at all densities
            T_greedy, NH_greedy = build_greedy_minimizer_for_interval(I_j, ivl_targets)

            min_cs = float('inf')
            for frac_i in range(1, 21):
                frac = frac_i / 20.0
                size = max(2, int(frac * t_j))
                if size > t_j:
                    size = t_j

                T_prefix = T_greedy[:size]
                NH_prefix = set()
                for k in T_prefix:
                    NH_prefix |= ivl_targets.get(k, set())
                tp = len(T_prefix)

                tau_prefix = Counter()
                for k in T_prefix:
                    for m in ivl_targets.get(k, set()):
                        tau_prefix[m] += 1
                E1 = sum(tau_prefix.values())
                E2 = sum(v * v for v in tau_prefix.values())
                cs = E1 * E1 / (tp * E2) if E2 > 0 else 0
                min_cs = min(min_cs, cs)

            cs_ok = min_cs >= 1.0

            # Determine proof technique
            if deg_ok:
                technique = "DEGREE"
            elif cs_ok:
                technique = "CS"
            elif max_group < avg_d * 0.5:
                technique = "GCD_GROUP"
            else:
                technique = "OPEN"

            print(f"  [{ivl_lo:>5},{ivl_hi:>5}] {t_j:>5} {min_d:>6} "
                  f"{avg_d:>6.1f} {'YES' if deg_ok else 'no':>7} "
                  f"{forced_gcd:>6.1f} {max_group:>6} {min_cs:>8.4f} "
                  f"{technique:>12}")

            results.append({
                'n': n, 'j': jj, 'lo': ivl_lo, 'hi': ivl_hi,
                'count': t_j, 'min_d': min_d, 'avg_d': avg_d,
                'deg_ok': deg_ok, 'forced_gcd': forced_gcd,
                'max_group': max_group, 'min_cs': min_cs,
                'cs_ok': cs_ok, 'technique': technique,
            })

    # Summary
    print(f"\n{'='*78}")
    print(f"  Z23c SUMMARY: Proof technique per interval")
    print(f"{'='*78}")
    for n in n_values:
        nr = [r for r in results if r['n'] == n]
        if not nr:
            continue
        by_tech = Counter(r['technique'] for r in nr)
        open_list = [r for r in nr if r['technique'] == 'OPEN']
        print(f"\n  n={n}: {dict(by_tech)}")
        if open_list:
            print(f"    OPEN intervals:")
            for r in open_list:
                print(f"      [{r['lo']},{r['hi']}]: |I|={r['count']}, "
                      f"min_d={r['min_d']}, avg_d={r['avg_d']:.1f}, "
                      f"CS_min={r['min_cs']:.4f}, fgcd={r['forced_gcd']:.1f}")
        else:
            print(f"    ALL INTERVALS HAVE PROOF TECHNIQUE!")

    return results


def main():
    parser = argparse.ArgumentParser(description="Z23: Per-interval CS")
    parser.add_argument('--n_values', type=str,
                        default='500,1000,2000,5000,10000,20000,50000')
    parser.add_argument('--experiments', type=str, default='Z23a,Z23b,Z23c')
    parser.add_argument('--sf', type=float, default=0.5)
    args = parser.parse_args()

    n_values = [int(x) for x in args.n_values.split(',')]
    experiments = [x.strip().upper() for x in args.experiments.split(',')]

    print(f"ERDŐS 710 — Z23a-Z23c: PER-INTERVAL CAUCHY-SCHWARZ")
    print(f"n values: {n_values}")
    print(f"sf: {args.sf}")
    print(f"Experiments: {experiments}")
    print("=" * 78)

    t0 = time.time()

    if 'Z23A' in experiments:
        experiment_Z23a(n_values, args.sf)
    if 'Z23B' in experiments:
        experiment_Z23b(n_values, args.sf)
    if 'Z23C' in experiments:
        experiment_Z23c(n_values, args.sf)

    elapsed = time.time() - t0
    print(f"\n{'='*78}")
    print(f"  TOTAL TIME: {elapsed:.1f}s")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
