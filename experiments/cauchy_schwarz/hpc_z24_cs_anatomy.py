#!/usr/bin/env python3
"""
ERDŐS 710 — EXPERIMENTS Z24a–Z24e: CLOSING THE ANALYTIC PROOF

Z23 showed per-interval CS ≥ 1 at all n tested. To complete the ANALYTIC proof,
we need to understand WHY CS works per-interval and identify the right bound.

The CS bound: |NH(T')| ≥ E₁²/E₂  where
  E₁ = Σ deg(k),  E₂ = Σ_m τ_j(m)²

E₂ = E₁ + 2·Δ  where  Δ = Σ_{k<k'} codeg(k,k')

CS ≥ 1  ⟺  E₁²/(t·E₂) ≥ 1  ⟺  d̄² ≥ d̄ + 2Δ/t  ⟺  d̄(d̄-1) ≥ 2Δ/t

KEY INSIGHT: Within [2^j, 2^{j+1}), no element divides another (ratio < 2).
So lcm(k,k') > max(k,k') for all pairs. This constrains codegrees.

Z24a: CS Anatomy — exact E₁, E₂, Δ, τ_max, d̄ per interval (full pool + adversarial)
Z24b: τ_j Distribution — per-interval target multiplicity; test if τ_max < d̄
Z24c: Codegree Bound — what determines Δ/t? The GCD structure within each interval
Z24d: Asymptotic Scaling — many n values, fit worst CS as function of n
Z24e: Analytic Bound Candidates — test specific bounds that could close the proof
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
    if s is None:
        s = len(elements)
    s = min(s, len(elements))

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


def factorize(m):
    """Return prime factorization as dict {p: e}."""
    factors = {}
    d = 2
    while d * d <= m:
        while m % d == 0:
            factors[d] = factors.get(d, 0) + 1
            m //= d
        d += 1
    if m > 1:
        factors[m] = factors.get(m, 0) + 1
    return factors


def num_divisors(m):
    """Count total number of divisors of m."""
    f = factorize(m)
    result = 1
    for e in f.values():
        result *= (e + 1)
    return result


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT Z24a: CS ANATOMY PER INTERVAL
# ═══════════════════════════════════════════════════════════════

def experiment_Z24a(n_values, sf=0.5):
    """
    Exact CS decomposition: E₁, E₂, Δ, τ_max_j, d̄ per interval.
    Both for full interval and adversarial worst-case prefix.
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z24a: CS ANATOMY PER INTERVAL")
    print("=" * 78)
    print("  E₂ = E₁ + 2Δ where Δ = Σ codeg(k,k')")
    print("  CS = d̄²/(d̄ + 2Δ/t)")
    print("  Key: what makes per-interval CS work?")

    results = []

    for n in n_values:
        data = build_smooth_data(n, sf)
        if data is None:
            continue

        N = data['N']
        nL = data['nL']
        M = data['M']
        sqrt_nL = data['sqrt_nL']
        delta = data['delta']
        targets = data['targets']
        pool_smooth = data['pool_smooth']

        intervals = get_dyadic_intervals(sqrt_nL, N)

        print(f"\n{'='*70}")
        print(f"  n={n}, N={N}, M={M}, √(n+L)={sqrt_nL:.1f}, δ={delta:.3f}")
        print(f"  {'Interval':>16} {'|I|':>5} {'d_min':>5} {'d_avg':>6} {'d_max':>5} "
              f"{'τ_max':>5} {'E₂/E₁':>6} {'2Δ/t':>7} {'CS':>6} "
              f"{'d̄/τmax':>7}")

        for jj, ivl_lo, ivl_hi in intervals:
            I_j = sorted([k for k in pool_smooth if ivl_lo <= k <= ivl_hi
                          and k > int(sqrt_nL)])
            if len(I_j) < 3:
                continue

            ivl_targets = {}
            for k in I_j:
                ivl_targets[k] = targets.get(k, set())

            t = len(I_j)
            degs = [len(ivl_targets[k]) for k in I_j]
            d_min = min(degs)
            d_max = max(degs)
            d_avg = sum(degs) / t

            # Compute τ_j(m) for each target
            tau_j = Counter()
            for k in I_j:
                for m in ivl_targets[k]:
                    tau_j[m] += 1

            E1 = sum(tau_j.values())
            E2 = sum(v * v for v in tau_j.values())
            tau_max = max(tau_j.values()) if tau_j else 0
            Delta = (E2 - E1) // 2  # integer since E2 - E1 is always even
            two_Delta_over_t = 2 * Delta / t if t > 0 else 0
            E2_over_E1 = E2 / E1 if E1 > 0 else 0
            CS = E1 * E1 / (t * E2) if E2 > 0 else 0
            d_bar_over_tau_max = d_avg / tau_max if tau_max > 0 else float('inf')

            print(f"  [{ivl_lo:>5},{ivl_hi:>5}] {t:>5} {d_min:>5} {d_avg:>6.1f} {d_max:>5} "
                  f"{tau_max:>5} {E2_over_E1:>6.2f} {two_Delta_over_t:>7.2f} {CS:>6.2f} "
                  f"{d_bar_over_tau_max:>7.2f}")

            results.append({
                'n': n, 'j': jj, 'lo': ivl_lo, 'hi': ivl_hi,
                'type': 'full', 't': t,
                'd_min': d_min, 'd_avg': d_avg, 'd_max': d_max,
                'tau_max': tau_max, 'E1': E1, 'E2': E2, 'Delta': Delta,
                'E2_over_E1': E2_over_E1, 'two_Delta_t': two_Delta_over_t,
                'CS': CS, 'd_over_tau': d_bar_over_tau_max,
            })

        # Now worst adversarial prefix per interval
        print(f"\n  ADVERSARIAL worst prefix (greedy minimizer):")
        print(f"  {'Interval':>16} {'frac':>5} {'t':>5} {'d_avg':>6} "
              f"{'τ_max':>5} {'E₂/E₁':>6} {'2Δ/t':>7} {'CS':>6}")

        for jj, ivl_lo, ivl_hi in intervals:
            I_j = sorted([k for k in pool_smooth if ivl_lo <= k <= ivl_hi
                          and k > int(sqrt_nL)])
            if len(I_j) < 3:
                continue

            ivl_targets = {}
            for k in I_j:
                ivl_targets[k] = targets.get(k, set())

            T_greedy, _ = build_greedy_minimizer_for_interval(I_j, ivl_targets)

            # Find worst CS prefix
            worst_cs = float('inf')
            worst_info = None

            for frac in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]:
                size = max(2, int(frac * len(T_greedy)))
                if size > len(T_greedy):
                    size = len(T_greedy)
                T_prefix = T_greedy[:size]
                tp = len(T_prefix)

                tau_p = Counter()
                for k in T_prefix:
                    for m in ivl_targets.get(k, set()):
                        tau_p[m] += 1

                E1 = sum(tau_p.values())
                E2 = sum(v * v for v in tau_p.values())
                tau_max = max(tau_p.values()) if tau_p else 0
                CS = E1 * E1 / (tp * E2) if E2 > 0 else 0
                d_avg = E1 / tp if tp > 0 else 0
                Delta = (E2 - E1) // 2
                two_dt = 2 * Delta / tp if tp > 0 else 0
                E2E1 = E2 / E1 if E1 > 0 else 0

                if CS < worst_cs:
                    worst_cs = CS
                    worst_info = {
                        'frac': frac, 'tp': tp, 'd_avg': d_avg,
                        'tau_max': tau_max, 'E2_over_E1': E2E1,
                        'two_Delta_t': two_dt, 'CS': CS,
                    }

            if worst_info:
                w = worst_info
                print(f"  [{ivl_lo:>5},{ivl_hi:>5}] {w['frac']:>5.2f} {w['tp']:>5} "
                      f"{w['d_avg']:>6.1f} {w['tau_max']:>5} {w['E2_over_E1']:>6.2f} "
                      f"{w['two_Delta_t']:>7.2f} {w['CS']:>6.2f}")

                results.append({
                    'n': n, 'j': jj, 'lo': ivl_lo, 'hi': ivl_hi,
                    'type': 'adversarial', **w,
                })

    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT Z24b: τ_j DISTRIBUTION PER INTERVAL
# ═══════════════════════════════════════════════════════════════

def experiment_Z24b(n_values, sf=0.5):
    """
    Per-interval target multiplicity distribution.
    Key question: is τ_max_j < d̄? If so, CS follows trivially.
    Also: what targets achieve high τ_j? What determines τ_j?
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z24b: PER-INTERVAL τ_j DISTRIBUTION")
    print("=" * 78)
    print("  τ_j(m) = |{k ∈ I_j : k|m}| = smooth divisors of m in [2^j, 2^{j+1})")
    print("  Key: is τ_max < d̄? What determines τ_max?")

    results = []

    for n in n_values:
        data = build_smooth_data(n, sf)
        if data is None:
            continue

        N = data['N']
        nL = data['nL']
        M = data['M']
        sqrt_nL = data['sqrt_nL']
        delta = data['delta']
        targets = data['targets']
        pool_smooth = data['pool_smooth']

        intervals = get_dyadic_intervals(sqrt_nL, N)

        print(f"\n{'='*70}")
        print(f"  n={n}, N={N}, M={M}, δ={delta:.3f}")

        for jj, ivl_lo, ivl_hi in intervals:
            I_j = sorted([k for k in pool_smooth if ivl_lo <= k <= ivl_hi
                          and k > int(sqrt_nL)])
            if len(I_j) < 3:
                continue

            ivl_targets = {}
            for k in I_j:
                ivl_targets[k] = targets.get(k, set())

            t = len(I_j)
            degs = [len(ivl_targets[k]) for k in I_j]
            d_avg = sum(degs) / t
            d_min = min(degs)

            # Compute τ_j(m) for each target
            tau_j = Counter()
            for k in I_j:
                for m in ivl_targets[k]:
                    tau_j[m] += 1

            tau_max = max(tau_j.values()) if tau_j else 0
            tau_vals = sorted(tau_j.values(), reverse=True)
            tau_dist = Counter(tau_j.values())

            # Find the worst targets
            worst_targets = [(m, tau_j[m]) for m in tau_j if tau_j[m] == tau_max]
            worst_targets.sort(key=lambda x: x[0])

            # For top-3 targets, analyze structure
            top3 = sorted(tau_j.items(), key=lambda x: -x[1])[:3]

            print(f"\n  Interval [{ivl_lo},{ivl_hi}]: |I|={t}, d̄={d_avg:.1f}, d_min={d_min}")
            print(f"    τ_max={tau_max}, d̄/τ_max={d_avg/tau_max:.2f}, d_min/τ_max={d_min/tau_max:.2f}")

            # τ distribution
            for tv in sorted(tau_dist.keys()):
                pct = 100 * tau_dist[tv] / len(tau_j)
                bar = '#' * min(50, int(pct))
                print(f"    τ={tv}: {tau_dist[tv]:>6} targets ({pct:>5.1f}%) {bar}")

            # Top targets
            for m, tv in top3:
                # How many divisors of m are in this interval?
                divs_in_interval = [k for k in I_j if m % k == 0]
                facts = factorize(m)
                fact_str = '·'.join(f'{p}^{e}' if e > 1 else str(p)
                                    for p, e in sorted(facts.items()))
                print(f"    Top target m={m} ({fact_str}): τ_j={tv}, "
                      f"divisors in interval: {divs_in_interval[:8]}{'...' if len(divs_in_interval)>8 else ''}")

            # Key test: does d̄ > τ_max? (Simple CS bound)
            simple_cs = d_avg > tau_max
            strong_cs = d_min > tau_max

            results.append({
                'n': n, 'j': jj, 'lo': ivl_lo, 'hi': ivl_hi,
                't': t, 'd_avg': d_avg, 'd_min': d_min,
                'tau_max': tau_max, 'd_over_tau': d_avg / tau_max,
                'simple_cs': simple_cs, 'strong_cs': strong_cs,
                'tau_dist': dict(tau_dist),
            })

    # Summary
    print(f"\n{'='*78}")
    print(f"  Z24b SUMMARY: d̄/τ_max per interval")
    print(f"{'='*78}")

    for n in n_values:
        nr = [r for r in results if r['n'] == n]
        if not nr:
            continue
        print(f"\n  n={n}:")
        print(f"  {'Interval':>16} {'|I|':>5} {'d̄':>6} {'d_min':>5} {'τ_max':>5} "
              f"{'d̄/τmax':>7} {'d_min/τmax':>10} {'simple_CS':>10}")
        for r in nr:
            print(f"  [{r['lo']:>5},{r['hi']:>5}] {r['t']:>5} {r['d_avg']:>6.1f} "
                  f"{r['d_min']:>5} {r['tau_max']:>5} "
                  f"{r['d_over_tau']:>7.2f} {r['d_min']/r['tau_max']:>10.2f} "
                  f"{'YES' if r['simple_cs'] else 'NO':>10}")

    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT Z24c: CODEGREE STRUCTURE WITHIN INTERVAL
# ═══════════════════════════════════════════════════════════════

def experiment_Z24c(n_values, sf=0.5):
    """
    Codegree analysis within each interval.
    codeg(k,k') ≈ M/lcm(k,k'). Since k,k' ∈ [2^j, 2^{j+1}), neither divides the other.
    Key question: what determines the codeg sum Δ?
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z24c: CODEGREE STRUCTURE WITHIN INTERVAL")
    print("=" * 78)
    print("  Key: within [2^j, 2^{j+1}), no element divides another.")
    print("  So lcm(k,k') > max(k,k'). What's the codegree distribution?")

    results = []

    for n in n_values:
        data = build_smooth_data(n, sf)
        if data is None:
            continue

        N = data['N']
        nL = data['nL']
        M = data['M']
        sqrt_nL = data['sqrt_nL']
        delta = data['delta']
        targets = data['targets']
        pool_smooth = data['pool_smooth']

        intervals = get_dyadic_intervals(sqrt_nL, N)

        print(f"\n{'='*70}")
        print(f"  n={n}, N={N}, M={M}, δ={delta:.3f}")

        for jj, ivl_lo, ivl_hi in intervals:
            I_j = sorted([k for k in pool_smooth if ivl_lo <= k <= ivl_hi
                          and k > int(sqrt_nL)])
            if len(I_j) < 3:
                continue

            ivl_targets = {}
            for k in I_j:
                ivl_targets[k] = targets.get(k, set())

            t = len(I_j)
            degs = [len(ivl_targets[k]) for k in I_j]
            d_avg = sum(degs) / t

            # Sample pairs for codegree analysis (cap at 5000 pairs)
            max_pairs = 5000
            if t * (t - 1) // 2 <= max_pairs:
                pairs = [(I_j[i], I_j[j]) for i in range(t) for j in range(i + 1, t)]
            else:
                pairs = []
                for _ in range(max_pairs):
                    i, j = random.sample(range(t), 2)
                    pairs.append((I_j[min(i, j)], I_j[max(i, j)]))

            # Compute codegrees
            codegs = []
            gcds = []
            lcms = []
            for k1, k2 in pairs:
                g = gcd(k1, k2)
                l = k1 * k2 // g
                # codeg = number of common multiples in (2n, n+L]
                if l > nL:
                    cd = 0
                else:
                    cd = (nL) // l - (2 * n) // l
                codegs.append(cd)
                gcds.append(g)
                lcms.append(l)

            codeg_0_frac = sum(1 for c in codegs if c == 0) / len(codegs) if codegs else 0
            codeg_pos = [c for c in codegs if c > 0]
            max_codeg = max(codegs) if codegs else 0
            avg_codeg = sum(codegs) / len(codegs) if codegs else 0
            avg_gcd = sum(gcds) / len(gcds) if gcds else 0
            avg_lcm = sum(lcms) / len(lcms) if lcms else 0

            # The key ratio: avg_codeg vs d̄
            codeg_over_d = avg_codeg / d_avg if d_avg > 0 else 0

            # gcd/k ratio (what fraction of k is shared)
            avg_gcd_over_k = avg_gcd / (ivl_lo + ivl_hi) * 2 if (ivl_lo + ivl_hi) > 0 else 0

            print(f"\n  Interval [{ivl_lo},{ivl_hi}]: |I|={t}, d̄={d_avg:.1f}")
            print(f"    codeg=0: {codeg_0_frac*100:.1f}%, max_codeg={max_codeg}, "
                  f"avg_codeg={avg_codeg:.3f}")
            print(f"    avg_gcd={avg_gcd:.1f}, avg_gcd/k̄={avg_gcd_over_k:.4f}")
            print(f"    avg_lcm={avg_lcm:.0f}, avg_codeg/d̄={codeg_over_d:.4f}")
            if codeg_pos:
                print(f"    codeg>0: count={len(codeg_pos)}, avg={sum(codeg_pos)/len(codeg_pos):.2f}, "
                      f"max={max(codeg_pos)}")

            # Distribution of codeg values
            codeg_dist = Counter(codegs)
            for cv in sorted(codeg_dist.keys())[:10]:
                pct = 100 * codeg_dist[cv] / len(codegs)
                print(f"    codeg={cv}: {codeg_dist[cv]:>5} pairs ({pct:.1f}%)")

            results.append({
                'n': n, 'j': jj, 'lo': ivl_lo, 'hi': ivl_hi,
                't': t, 'd_avg': d_avg,
                'codeg_0_frac': codeg_0_frac, 'max_codeg': max_codeg,
                'avg_codeg': avg_codeg, 'avg_gcd': avg_gcd,
                'codeg_over_d': codeg_over_d,
            })

    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT Z24d: ASYMPTOTIC SCALING
# ═══════════════════════════════════════════════════════════════

def experiment_Z24d(n_values_extended, sf=0.5):
    """
    Track worst per-interval CS across many n values.
    Fit asymptotic scaling law.
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z24d: ASYMPTOTIC SCALING OF WORST PER-INTERVAL CS")
    print("=" * 78)
    print("  Goal: show worst CS → ∞ as n → ∞")

    results = []

    for n in n_values_extended:
        data = build_smooth_data(n, sf)
        if data is None:
            continue

        N = data['N']
        nL = data['nL']
        M = data['M']
        sqrt_nL = data['sqrt_nL']
        delta = data['delta']
        targets = data['targets']
        pool_smooth = data['pool_smooth']

        intervals = get_dyadic_intervals(sqrt_nL, N)

        worst_cs = float('inf')
        worst_info = None

        for jj, ivl_lo, ivl_hi in intervals:
            I_j = sorted([k for k in pool_smooth if ivl_lo <= k <= ivl_hi
                          and k > int(sqrt_nL)])
            if len(I_j) < 3:
                continue

            ivl_targets = {}
            for k in I_j:
                ivl_targets[k] = targets.get(k, set())

            T_greedy, _ = build_greedy_minimizer_for_interval(I_j, ivl_targets)

            for frac in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]:
                size = max(2, int(frac * len(T_greedy)))
                if size > len(T_greedy):
                    size = len(T_greedy)
                T_prefix = T_greedy[:size]
                tp = len(T_prefix)

                tau_p = Counter()
                for k in T_prefix:
                    for m in ivl_targets.get(k, set()):
                        tau_p[m] += 1

                E1 = sum(tau_p.values())
                E2 = sum(v * v for v in tau_p.values())
                tau_max = max(tau_p.values()) if tau_p else 0
                CS = E1 * E1 / (tp * E2) if E2 > 0 else 0
                d_avg = E1 / tp if tp > 0 else 0

                if CS < worst_cs:
                    worst_cs = CS
                    worst_info = {
                        'n': n, 'j': jj, 'lo': ivl_lo, 'hi': ivl_hi,
                        'frac': frac, 'tp': tp, 'd_avg': d_avg,
                        'tau_max': tau_max, 'CS': CS, 'delta': delta,
                    }

        if worst_info:
            results.append(worst_info)
            w = worst_info
            print(f"  n={n:>6}: worst_CS={w['CS']:.4f} at [{w['lo']},{w['hi']}] "
                  f"frac={w['frac']:.2f} d̄={w['d_avg']:.1f} τ_max={w['tau_max']} δ={delta:.3f}")

    # Fit scaling
    print(f"\n  SCALING ANALYSIS:")
    if len(results) >= 3:
        # Try CS ~ a * delta^b
        # log(CS) ~ log(a) + b * log(delta)
        import math
        log_cs = [math.log(r['CS']) for r in results if r['CS'] > 0]
        log_delta = [math.log(r['delta']) for r in results if r['CS'] > 0]

        if len(log_cs) >= 3:
            n_pts = len(log_cs)
            sx = sum(log_delta)
            sy = sum(log_cs)
            sxx = sum(x * x for x in log_delta)
            sxy = sum(x * y for x, y in zip(log_delta, log_cs))
            denom = n_pts * sxx - sx * sx
            if abs(denom) > 1e-12:
                b = (n_pts * sxy - sx * sy) / denom
                a_log = (sy - b * sx) / n_pts
                a = math.exp(a_log)
                print(f"  Fit: CS ≈ {a:.3f} · δ^{b:.3f}")

                # Residuals
                max_resid = max(abs(log_cs[i] - a_log - b * log_delta[i])
                                for i in range(n_pts))
                print(f"  Max log-residual: {max_resid:.4f}")

                # Prediction
                for n_pred in [100000, 200000, 500000]:
                    L_pred = target_L(n_pred)
                    M_pred = L_pred - n_pred
                    d_pred = 2 * M_pred / n_pred - 1
                    cs_pred = a * d_pred ** b
                    print(f"  Predicted CS at n={n_pred}: {cs_pred:.3f} (δ={d_pred:.3f})")

    # Also try CS ~ a * log(n)^b
    print(f"\n  Alternative fit: CS ~ a · (log n)^b")
    if len(results) >= 3:
        import math
        log_cs = [math.log(r['CS']) for r in results if r['CS'] > 0]
        log_logn = [math.log(math.log(r['n'])) for r in results if r['CS'] > 0]

        n_pts = len(log_cs)
        sx = sum(log_logn)
        sy = sum(log_cs)
        sxx = sum(x * x for x in log_logn)
        sxy = sum(x * y for x, y in zip(log_logn, log_cs))
        denom = n_pts * sxx - sx * sx
        if abs(denom) > 1e-12:
            b = (n_pts * sxy - sx * sy) / denom
            a_log = (sy - b * sx) / n_pts
            a = math.exp(a_log)
            print(f"  Fit: CS ≈ {a:.3f} · (log n)^{b:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT Z24e: ANALYTIC BOUND CANDIDATES
# ═══════════════════════════════════════════════════════════════

def experiment_Z24e(n_values, sf=0.5):
    """
    Test specific analytic bounds that could close the proof.

    Bound 1: τ_j(m) ≤ (n+L)/2^{j+1} (trivial — complementary divisor count)
    Bound 2: τ_j(m) ≤ d(m) restricted to [2^j, 2^{j+1}) ≤ d(m)/j (pigeonhole over j dyadic intervals)
    Bound 3: τ_j(m) ≤ M/(2^j · (something)) — using the smooth constraint
    Bound 4: Σ τ_j² ≤ C · t · d̄ for some universal C (implies CS ≥ d̄/C)

    Also: for the WORST interval, compute (n+L)/(2^{j+1} · M) = τ_bound / d̄.
    This ratio must be < 1 for the trivial bound to work.
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z24e: ANALYTIC BOUND CANDIDATES")
    print("=" * 78)

    results = []

    for n in n_values:
        data = build_smooth_data(n, sf)
        if data is None:
            continue

        N = data['N']
        nL = data['nL']
        M = data['M']
        sqrt_nL = data['sqrt_nL']
        delta = data['delta']
        targets = data['targets']
        pool_smooth = data['pool_smooth']

        intervals = get_dyadic_intervals(sqrt_nL, N)

        print(f"\n{'='*70}")
        print(f"  n={n}, N={N}, M={M}, (n+L)={nL}, δ={delta:.3f}")
        print(f"  {'Interval':>16} {'|I|':>5} {'d̄':>6} {'τmax':>5} {'τ_triv':>7} "
              f"{'nL/M':>6} {'E₂/tE₁':>7} {'CS':>6} {'C_eff':>6}")

        for jj, ivl_lo, ivl_hi in intervals:
            I_j = sorted([k for k in pool_smooth if ivl_lo <= k <= ivl_hi
                          and k > int(sqrt_nL)])
            if len(I_j) < 3:
                continue

            ivl_targets = {}
            for k in I_j:
                ivl_targets[k] = targets.get(k, set())

            t = len(I_j)
            degs = [len(ivl_targets[k]) for k in I_j]
            d_avg = sum(degs) / t

            tau_j = Counter()
            for k in I_j:
                for m in ivl_targets[k]:
                    tau_j[m] += 1

            E1 = sum(tau_j.values())
            E2 = sum(v * v for v in tau_j.values())
            tau_max = max(tau_j.values()) if tau_j else 0
            CS = E1 * E1 / (t * E2) if E2 > 0 else 0
            E2_over_tE1 = E2 / (t * d_avg) if d_avg > 0 and t > 0 else 0

            # Trivial bound: τ ≤ (n+L)/2^{j+1}
            tau_trivial = nL // (1 << (jj + 1))
            nL_over_M = nL / M

            # Effective constant C: E₂ = C · t · d̄
            C_eff = E2 / (t * d_avg) if t > 0 and d_avg > 0 else 0
            # Then CS = d̄/C_eff

            # Number of dyadic intervals that m's divisors span
            # For smooth m ≤ n+L with P(m) ≤ √(n+L):
            # d(m) = product of (e_p + 1) over primes p | m
            # Divisors of m in [2^j, 2^{j+1}) = divisors in a factor-2 range
            # Average d(m) for targets hit by this interval:
            targets_hit = list(tau_j.keys())
            if targets_hit:
                sample_targets = targets_hit[:min(100, len(targets_hit))]
                avg_d_m = sum(num_divisors(m) for m in sample_targets) / len(sample_targets)
                num_dyadic = int(log2(nL)) - int(log2(max(1, int(sqrt_nL)))) + 1
                pigeonhole_bound = avg_d_m / num_dyadic if num_dyadic > 0 else 0
            else:
                avg_d_m = 0
                pigeonhole_bound = 0

            print(f"  [{ivl_lo:>5},{ivl_hi:>5}] {t:>5} {d_avg:>6.1f} {tau_max:>5} "
                  f"{tau_trivial:>7} {nL_over_M:>6.2f} {E2_over_tE1:>7.2f} "
                  f"{CS:>6.2f} {C_eff:>6.2f}")

            results.append({
                'n': n, 'j': jj, 'lo': ivl_lo, 'hi': ivl_hi,
                't': t, 'd_avg': d_avg, 'tau_max': tau_max,
                'tau_trivial': tau_trivial, 'nL_over_M': nL_over_M,
                'CS': CS, 'C_eff': C_eff, 'avg_d_m': avg_d_m,
                'pigeonhole': pigeonhole_bound,
            })

    # Key analysis: what's the effective C and does it stay bounded?
    print(f"\n{'='*78}")
    print(f"  Z24e SUMMARY: Effective constant C where E₂ = C·t·d̄")
    print(f"  (CS = d̄/C, so need C < d̄)")
    print(f"{'='*78}")

    for n in n_values:
        nr = [r for r in results if r['n'] == n]
        if not nr:
            continue
        worst = max(nr, key=lambda r: r['C_eff'])
        print(f"\n  n={n}: worst C_eff = {worst['C_eff']:.3f} at [{worst['lo']},{worst['hi']}], "
              f"d̄={worst['d_avg']:.1f}, CS={worst['CS']:.2f}")
        print(f"    τ_max={worst['tau_max']}, τ_trivial={worst['tau_trivial']}, "
              f"(n+L)/M={worst['nL_over_M']:.2f}")
        print(f"    avg d(m) for targets={worst['avg_d_m']:.1f}")

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_values', default='500,1000,2000,5000,10000,20000,50000')
    parser.add_argument('--n_extended', default='100,200,300,500,700,1000,1500,2000,3000,5000,7000,10000,15000,20000,30000,50000,70000,100000')
    parser.add_argument('--experiments', default='Z24a,Z24b,Z24c,Z24d,Z24e')
    parser.add_argument('--sf', type=float, default=0.5)
    args = parser.parse_args()

    n_values = [int(x) for x in args.n_values.split(',')]
    n_extended = [int(x) for x in args.n_extended.split(',')]
    experiments = [e.strip().upper() for e in args.experiments.split(',')]
    sf = args.sf

    print(f"ERDŐS 710 — Z24a-Z24e: CLOSING THE ANALYTIC PROOF")
    print(f"n values: {n_values}")
    print(f"n extended: {n_extended}")
    print(f"sf: {sf}")
    print(f"Experiments: {experiments}")
    print("=" * 78)

    t0 = time.time()

    if 'Z24A' in experiments:
        experiment_Z24a(n_values, sf)

    if 'Z24B' in experiments:
        experiment_Z24b(n_values, sf)

    if 'Z24C' in experiments:
        experiment_Z24c(n_values, sf)

    if 'Z24D' in experiments:
        experiment_Z24d(n_extended, sf)

    if 'Z24E' in experiments:
        experiment_Z24e(n_values, sf)

    elapsed = time.time() - t0
    print(f"\n{'='*78}")
    print(f"  TOTAL TIME: {elapsed:.1f}s")
    print(f"{'='*78}")
