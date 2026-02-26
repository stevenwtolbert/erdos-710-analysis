#!/usr/bin/env python3
"""
ERDOS 710 -- EXPERIMENT Z40: QUASI-INDEPENDENCE OF TARGET OVERLAP

Key hypothesis: For dyadic intervals I_j, I_{j'} of smooth sources, the
target overlap satisfies:
  |NH(I_j) ∩ NH(I_{j'})| ≈ |NH(I_j)| · |NH(I_{j'})| / |H_smooth|

If targets are hit "independently" by different intervals, then:
  SURPLUS >= (alpha_min - 1)|T| >= 0.389|T|
  EXCESS = sum overlaps approx sum_{j<j'} |NH(T_j)|*|NH(T_{j'})|/|H|
  and if EXCESS < SURPLUS, Hall holds.

Parts:
  A: Pairwise overlap vs product formula (R ratios)
  B: Overlap for adversarial T spanning 2 intervals
  C: Inclusion-exclusion with quasi-independence
  D: Per-target correlation structure
  E: The key analytic ratio (EXCESS_qi / SURPLUS)
  F: Scaling of key quantities with n
"""

import time
import sys
from math import gcd, log, sqrt, exp, floor, ceil, log2
from collections import Counter, defaultdict

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


def get_dyadic_intervals(B, N):
    lo = B + 1
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


def setup_n(n):
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
    B = int(sqrt_nL)

    # S_+ = B-smooth elements in (B, N]
    pool_smooth = sorted([k for k in range(B + 1, N + 1) if largest_prime_factor(k) <= B])

    # Targets for each k
    targets = {}
    for k in pool_smooth:
        targets[k] = compute_targets(k, n, L)

    # H_smooth = union of all targets
    H_smooth = set()
    for k in pool_smooth:
        H_smooth |= targets[k]

    # Dyadic intervals
    intervals = get_dyadic_intervals(B, N)

    # Partition pool_smooth into intervals
    interval_members = {}
    for jj, ivl_lo, ivl_hi in intervals:
        I_j = sorted([k for k in pool_smooth if ivl_lo <= k <= ivl_hi])
        if I_j:
            interval_members[jj] = I_j

    # NH per interval
    interval_NH = {}
    for jj, members in interval_members.items():
        nh = set()
        for k in members:
            nh |= targets.get(k, set())
        interval_NH[jj] = nh

    return {
        'n': n, 'L': L, 'M': M, 'N': N, 'B': B, 'delta': delta, 'nL': nL,
        'sqrt_nL': sqrt_nL, 'pool_smooth': pool_smooth, 'targets': targets,
        'H_smooth': H_smooth, 'intervals': intervals,
        'interval_members': interval_members, 'interval_NH': interval_NH
    }


def compute_NH(elements, targets):
    nh = set()
    for k in elements:
        nh |= targets.get(k, set())
    return nh


def greedy_minimize_ratio(pool, targets, max_size=None):
    """
    Greedy: build T by adding element that adds fewest NEW targets.
    Track the worst (minimum) ratio |NH(T)|/|T| seen at any step.
    """
    if max_size is None:
        max_size = len(pool)
    max_size = min(max_size, len(pool))

    t2p = defaultdict(list)
    for k in pool:
        for h in targets.get(k, set()):
            t2p[h].append(k)

    T = []
    NH = set()
    rem = set(pool)
    new_count = {k: len(targets.get(k, set())) for k in pool}

    best_ratio = float('inf')
    best_T = []
    best_NH = set()

    for step in range(max_size):
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

        ratio = len(NH) / len(T)
        if ratio < best_ratio:
            best_ratio = ratio
            best_T = list(T)
            best_NH = set(NH)

    return best_T, best_NH, best_ratio


# ================================================================
#  PART A: PAIRWISE OVERLAP vs PRODUCT FORMULA
# ================================================================

def part_a(ctx):
    print("\n" + "=" * 72)
    print("  PART A: PAIRWISE OVERLAP vs PRODUCT FORMULA")
    print("=" * 72)

    im = ctx['interval_members']
    inh = ctx['interval_NH']
    H_smooth = ctx['H_smooth']
    H_size = len(H_smooth)

    sorted_jjs = sorted(im.keys())
    n_intervals = len(sorted_jjs)

    print(f"\n  |H_smooth| = {H_size}")
    print(f"  Intervals: {sorted_jjs}")
    for jj in sorted_jjs:
        print(f"    I_{jj}: |I_j|={len(im[jj])}, |NH(I_j)|={len(inh[jj])}, "
              f"hit_rate={len(inh[jj])/H_size:.4f}")

    # Pairwise overlap analysis
    pair_data = []
    print(f"\n  Pairwise overlaps:")
    print(f"  {'pair':>12s}  {'|OV|':>8s}  {'PROD':>10s}  {'R=OV/PROD':>10s}  "
          f"|NH_j|  |NH_j'|  hit_j  hit_j'")
    print(f"  {'─'*12}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*6}")

    for i in range(n_intervals):
        j1 = sorted_jjs[i]
        for ii in range(i + 1, n_intervals):
            j2 = sorted_jjs[ii]
            ov = len(inh[j1] & inh[j2])
            prod = len(inh[j1]) * len(inh[j2]) / H_size if H_size > 0 else 0
            R = ov / prod if prod > 0 else float('inf')
            hit1 = len(inh[j1]) / H_size
            hit2 = len(inh[j2]) / H_size

            pair_data.append({
                'j1': j1, 'j2': j2, 'ov': ov, 'prod': prod, 'R': R,
                'nh1': len(inh[j1]), 'nh2': len(inh[j2]),
                'hit1': hit1, 'hit2': hit2
            })

            print(f"  ({j1:>3},{j2:>3})   {ov:>8d}  {prod:>10.1f}  {R:>10.4f}  "
                  f"{len(inh[j1]):>6d}  {len(inh[j2]):>7d}  {hit1:>.4f}  {hit2:>.4f}")

    if pair_data:
        R_vals = [p['R'] for p in pair_data if p['R'] != float('inf')]
        if R_vals:
            print(f"\n  R statistics: min={min(R_vals):.4f}, max={max(R_vals):.4f}, "
                  f"mean={sum(R_vals)/len(R_vals):.4f}")
            if max(R_vals) > 1:
                print(f"  POSITIVE CORRELATION detected (R > 1) — overlap worse than independent")
            elif min(R_vals) < 1:
                print(f"  NEGATIVE CORRELATION detected (R < 1) — overlap better than independent")
        else:
            print(f"\n  No valid R values computed.")
    else:
        print(f"\n  No pairs to analyze.")

    return pair_data


# ================================================================
#  PART B: OVERLAP FOR ADVERSARIAL T SPANNING 2 INTERVALS
# ================================================================

def part_b(ctx):
    print("\n" + "=" * 72)
    print("  PART B: ADVERSARIAL T SPANNING 2 INTERVALS")
    print("=" * 72)

    im = ctx['interval_members']
    targets = ctx['targets']
    H_smooth = ctx['H_smooth']
    H_size = len(H_smooth)
    sorted_jjs = sorted(im.keys(), reverse=True)

    results_b = []

    # Test all adjacent pairs
    for idx in range(len(sorted_jjs) - 1):
        j1 = sorted_jjs[idx]
        j2 = sorted_jjs[idx + 1]
        if abs(j1 - j2) != 1:
            continue

        pool_2 = im[j1] + im[j2]
        max_sz = min(len(pool_2), 200)  # cap for speed

        # Build targets dict for this 2-interval pool
        tgt_2 = {k: targets.get(k, set()) for k in pool_2}

        # Greedy minimizer
        T_adv, NH_adv, min_ratio = greedy_minimize_ratio(pool_2, tgt_2, max_size=max_sz)

        # Partition T_adv into its two intervals
        T_j1 = [k for k in T_adv if k in set(im[j1])]
        T_j2 = [k for k in T_adv if k in set(im[j2])]

        NH_j1 = compute_NH(T_j1, targets)
        NH_j2 = compute_NH(T_j2, targets)

        ov = len(NH_j1 & NH_j2)
        prod = len(NH_j1) * len(NH_j2) / H_size if H_size > 0 else 0
        R_T = ov / prod if prod > 0 else float('inf')

        surplus_j1 = len(NH_j1) - len(T_j1) if T_j1 else 0
        surplus_j2 = len(NH_j2) - len(T_j2) if T_j2 else 0
        surplus = surplus_j1 + surplus_j2
        actual_expansion = len(NH_adv) - len(T_adv)

        # per-interval min alpha
        alpha_j1 = len(NH_j1) / len(T_j1) if T_j1 else float('inf')
        alpha_j2 = len(NH_j2) / len(T_j2) if T_j2 else float('inf')
        alpha_min = min(alpha_j1, alpha_j2)

        print(f"\n  Pair (I_{j1}, I_{j2}):")
        print(f"    |T|={len(T_adv)}, |T_j1|={len(T_j1)}, |T_j2|={len(T_j2)}")
        print(f"    |NH_j1|={len(NH_j1)}, |NH_j2|={len(NH_j2)}, |NH(T)|={len(NH_adv)}")
        print(f"    alpha_j1={alpha_j1:.4f}, alpha_j2={alpha_j2:.4f}, alpha_min={alpha_min:.4f}")
        print(f"    OV={ov}, PROD={prod:.1f}, R_T=OV/PROD={R_T:.4f}")
        print(f"    SURPLUS={surplus}, EXCESS(=OV)={ov}")
        print(f"    OV vs 0.389*|T| = {0.389*len(T_adv):.1f}:  OV {'<' if ov < 0.389*len(T_adv) else '>='} 0.389|T|")
        print(f"    |NH(T)|/|T| = {min_ratio:.4f}")
        print(f"    SURPLUS > EXCESS? {'YES' if surplus > ov else 'NO'}")

        results_b.append({
            'j1': j1, 'j2': j2, 'T_size': len(T_adv),
            'T_j1_size': len(T_j1), 'T_j2_size': len(T_j2),
            'nh_j1': len(NH_j1), 'nh_j2': len(NH_j2), 'nh_T': len(NH_adv),
            'ov': ov, 'prod': prod, 'R_T': R_T,
            'surplus': surplus, 'min_ratio': min_ratio,
            'alpha_min': alpha_min
        })

    # Also test top-2 intervals (may not be adjacent)
    if len(sorted_jjs) >= 2:
        j1, j2 = sorted_jjs[0], sorted_jjs[1]
        pool_2 = im[j1] + im[j2]
        max_sz = min(len(pool_2), 200)
        tgt_2 = {k: targets.get(k, set()) for k in pool_2}
        T_adv, NH_adv, min_ratio = greedy_minimize_ratio(pool_2, tgt_2, max_size=max_sz)

        T_j1 = [k for k in T_adv if k in set(im[j1])]
        T_j2 = [k for k in T_adv if k in set(im[j2])]
        NH_j1 = compute_NH(T_j1, targets)
        NH_j2 = compute_NH(T_j2, targets)
        ov = len(NH_j1 & NH_j2)
        prod = len(NH_j1) * len(NH_j2) / H_size if H_size > 0 else 0
        R_T = ov / prod if prod > 0 else float('inf')
        surplus = (len(NH_j1) - len(T_j1)) + (len(NH_j2) - len(T_j2))

        print(f"\n  Top-2 intervals (I_{j1}, I_{j2}):")
        print(f"    |T|={len(T_adv)}, OV={ov}, PROD={prod:.1f}, R_T={R_T:.4f}")
        print(f"    SURPLUS={surplus}, |NH(T)|/|T|={min_ratio:.4f}")
        print(f"    SURPLUS > EXCESS? {'YES' if surplus > ov else 'NO'}")

        results_b.append({
            'j1': j1, 'j2': j2, 'T_size': len(T_adv),
            'T_j1_size': len(T_j1), 'T_j2_size': len(T_j2),
            'nh_j1': len(NH_j1), 'nh_j2': len(NH_j2), 'nh_T': len(NH_adv),
            'ov': ov, 'prod': prod, 'R_T': R_T,
            'surplus': surplus, 'min_ratio': min_ratio,
            'alpha_min': min(len(NH_j1)/max(len(T_j1),1), len(NH_j2)/max(len(T_j2),1))
        })

    return results_b


# ================================================================
#  PART C: INCLUSION-EXCLUSION WITH QUASI-INDEPENDENCE
# ================================================================

def part_c(ctx):
    print("\n" + "=" * 72)
    print("  PART C: INCLUSION-EXCLUSION WITH QUASI-INDEPENDENCE")
    print("=" * 72)

    im = ctx['interval_members']
    inh = ctx['interval_NH']
    targets = ctx['targets']
    pool = ctx['pool_smooth']
    H_smooth = ctx['H_smooth']
    H_size = len(H_smooth)

    sorted_jjs = sorted(im.keys())

    # T = S_+ (all sources)
    # SURPLUS = sum_j (|NH(I_j)| - |I_j|)
    surplus = 0
    for jj in sorted_jjs:
        s_j = len(inh[jj]) - len(im[jj])
        surplus += s_j

    # EXCESS_actual = sum_{h in H_smooth} (mu(h) - 1) where mu(h) = |{j : exists k in I_j dividing h}|
    mu = Counter()  # mu(h) = number of intervals contributing to h
    for jj in sorted_jjs:
        for h in inh[jj]:
            mu[h] += 1
    excess_actual = sum(max(0, v - 1) for v in mu.values())

    # EXCESS_qi = sum_{j<j'} |NH(I_j)| * |NH(I_{j'})| / |H_smooth|
    excess_qi = 0.0
    for i in range(len(sorted_jjs)):
        j1 = sorted_jjs[i]
        for ii in range(i + 1, len(sorted_jjs)):
            j2 = sorted_jjs[ii]
            excess_qi += len(inh[j1]) * len(inh[j2]) / H_size

    # EXCESS_qi_pairwise (using actual pairwise overlaps as ground truth)
    excess_actual_pairwise = 0
    for i in range(len(sorted_jjs)):
        j1 = sorted_jjs[i]
        for ii in range(i + 1, len(sorted_jjs)):
            j2 = sorted_jjs[ii]
            excess_actual_pairwise += len(inh[j1] & inh[j2])

    # Triple overlaps for Bonferroni correction
    triple_overlap_total = 0
    n_triples = 0
    if len(sorted_jjs) >= 3:
        # Sample triples for large numbers of intervals
        triple_count = 0
        for i in range(len(sorted_jjs)):
            j1 = sorted_jjs[i]
            for ii in range(i + 1, len(sorted_jjs)):
                j2 = sorted_jjs[ii]
                for iii in range(ii + 1, len(sorted_jjs)):
                    j3 = sorted_jjs[iii]
                    triple = len(inh[j1] & inh[j2] & inh[j3])
                    triple_overlap_total += triple
                    triple_count += 1
        n_triples = triple_count

    # The sum of |NH(I_j)|
    sum_NH = sum(len(inh[jj]) for jj in sorted_jjs)

    # Quasi-independence upper bound: (sum |NH(I_j)|)^2 / (2 * |H_smooth|)
    excess_qi_upper = sum_NH ** 2 / (2 * H_size) if H_size > 0 else 0

    # alpha_bar = sum |NH(I_j)| / sum |I_j|
    sum_T = sum(len(im[jj]) for jj in sorted_jjs)
    alpha_bar = sum_NH / sum_T if sum_T > 0 else 0

    # NH(S_+) actual
    NH_all = compute_NH(pool, targets)

    print(f"\n  T = S_+ (full pool), |T| = {len(pool)}")
    print(f"  |NH(T)| = {len(NH_all)}")
    print(f"  |H_smooth| = {H_size}")
    print(f"\n  sum_j |NH(I_j)| = {sum_NH}")
    print(f"  alpha_bar = {alpha_bar:.4f}")
    print(f"  sum_j |I_j| = {sum_T}")
    print(f"\n  SURPLUS = {surplus} (= {surplus/len(pool):.4f} per source)")
    print(f"  EXCESS_actual = {excess_actual} (= {excess_actual/len(pool):.4f} per source)")
    print(f"  EXCESS_pairwise = {excess_actual_pairwise}")
    print(f"  EXCESS_qi (product formula) = {excess_qi:.1f}")
    print(f"  EXCESS_qi_upper ((sum NH)^2/2H) = {excess_qi_upper:.1f}")
    if n_triples > 0:
        print(f"  Triple overlap total = {triple_overlap_total} ({n_triples} triples)")
        excess_qi_bonferroni = excess_qi - triple_overlap_total
        print(f"  EXCESS_qi - triples (Bonferroni) = {excess_qi_bonferroni:.1f}")

    print(f"\n  COMPARISON:")
    print(f"  EXCESS_actual / EXCESS_qi = {excess_actual / excess_qi:.4f}" if excess_qi > 0 else "  EXCESS_qi = 0")
    print(f"  EXCESS_actual / SURPLUS = {excess_actual / surplus:.4f}" if surplus > 0 else "  SURPLUS = 0")
    print(f"  EXCESS_qi / SURPLUS = {excess_qi / surplus:.4f}" if surplus > 0 else "  SURPLUS = 0")
    print(f"  EXCESS_qi_upper / SURPLUS = {excess_qi_upper / surplus:.4f}" if surplus > 0 else "  SURPLUS = 0")

    print(f"\n  SURPLUS > EXCESS_actual? {'YES' if surplus > excess_actual else 'NO'}")
    print(f"  SURPLUS > EXCESS_qi?     {'YES' if surplus > excess_qi else 'NO'}")
    print(f"  SURPLUS > EXCESS_qi_upper? {'YES' if surplus > excess_qi_upper else 'NO'}")

    # Per-interval breakdown
    print(f"\n  Per-interval breakdown:")
    for jj in sorted_jjs:
        s_j = len(inh[jj]) - len(im[jj])
        alpha_j = len(inh[jj]) / len(im[jj]) if im[jj] else float('inf')
        print(f"    I_{jj}: |I_j|={len(im[jj]):5d}, |NH|={len(inh[jj]):6d}, "
              f"surplus={s_j:6d}, alpha={alpha_j:.4f}")

    return {
        'surplus': surplus, 'excess_actual': excess_actual,
        'excess_qi': excess_qi, 'excess_qi_upper': excess_qi_upper,
        'triple_overlap': triple_overlap_total,
        'sum_NH': sum_NH, 'alpha_bar': alpha_bar,
        'surplus_per_T': surplus / len(pool) if pool else 0,
        'excess_per_T': excess_actual / len(pool) if pool else 0,
        'qi_ratio': excess_actual / excess_qi if excess_qi > 0 else float('inf'),
        'excess_over_surplus': excess_actual / surplus if surplus > 0 else float('inf'),
        'qi_over_surplus': excess_qi / surplus if surplus > 0 else float('inf')
    }


# ================================================================
#  PART D: PER-TARGET CORRELATION STRUCTURE
# ================================================================

def part_d(ctx):
    print("\n" + "=" * 72)
    print("  PART D: PER-TARGET CORRELATION STRUCTURE")
    print("=" * 72)

    im = ctx['interval_members']
    inh = ctx['interval_NH']
    targets = ctx['targets']
    H_smooth = ctx['H_smooth']
    H_size = len(H_smooth)

    sorted_jjs = sorted(im.keys())

    # For each h in H_smooth, find which intervals contribute
    h_to_intervals = defaultdict(set)
    for jj in sorted_jjs:
        for h in inh[jj]:
            h_to_intervals[h].add(jj)

    # Multi-interval targets
    multi = {h: jset for h, jset in h_to_intervals.items() if len(jset) >= 2}
    n_multi = len(multi)
    frac_multi = n_multi / H_size if H_size > 0 else 0

    print(f"\n  Total targets: {H_size}")
    print(f"  Multi-interval targets (mu >= 2): {n_multi} ({100*frac_multi:.1f}%)")

    if not multi:
        print(f"  No multi-interval targets. Quasi-independence is trivially exact.")
        return {'n_multi': 0, 'frac_multi': 0}

    # Multiplicity distribution
    mu_dist = Counter(len(jset) for jset in h_to_intervals.values())
    print(f"\n  Multiplicity distribution:")
    for m in sorted(mu_dist.keys()):
        print(f"    mu={m}: {mu_dist[m]} targets ({100*mu_dist[m]/H_size:.1f}%)")

    # For mu>=2 targets: interval spread
    spread_dist = Counter()
    adjacent_count = 0
    for h, jset in multi.items():
        sorted_j = sorted(jset)
        spread = sorted_j[-1] - sorted_j[0]
        spread_dist[spread] += 1
        if len(jset) == 2 and spread == 1:
            adjacent_count += 1

    print(f"\n  Among multi-interval targets:")
    print(f"    Adjacent-only (mu=2, spread=1): {adjacent_count} ({100*adjacent_count/n_multi:.1f}%)")
    print(f"\n  Spread distribution:")
    for s in sorted(spread_dist.keys()):
        print(f"    spread={s}: {spread_dist[s]} ({100*spread_dist[s]/n_multi:.1f}%)")

    # Per-pair correlation factor:
    # For each pair (j, j'), compute:
    #   P_j = |NH(I_j)| / |H_smooth| (hit rate)
    #   Expected_shared = P_j * P_{j'} * |H_smooth| = |NH(I_j)| * |NH(I_{j'})| / |H_smooth|
    #   Actual_shared = |NH(I_j) ∩ NH(I_{j'})|
    #   CF(j,j') = Actual / Expected
    print(f"\n  Per-pair correlation factors (CF = actual_shared / expected_shared):")
    print(f"  {'pair':>12s}  {'actual':>8s}  {'expected':>10s}  {'CF':>8s}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*10}  {'─'*8}")

    pair_CFs = []
    for i in range(len(sorted_jjs)):
        j1 = sorted_jjs[i]
        for ii in range(i + 1, len(sorted_jjs)):
            j2 = sorted_jjs[ii]
            actual = len(inh[j1] & inh[j2])
            expected = len(inh[j1]) * len(inh[j2]) / H_size if H_size > 0 else 0
            CF = actual / expected if expected > 0 else float('inf')
            print(f"  ({j1:>3},{j2:>3})   {actual:>8d}  {expected:>10.1f}  {CF:>8.4f}")
            pair_CFs.append({'j1': j1, 'j2': j2, 'actual': actual, 'expected': expected, 'CF': CF})

    # Correlation by target "type": highly composite vs typical
    # Define: count number of divisors of h that are in S_+
    # High-div targets have more sharing
    print(f"\n  Correlation by target type (divisor count in S_+):")
    pool_set = set(ctx['pool_smooth'])
    h_div_count = {}
    for h in H_smooth:
        cnt = 0
        for jj, members in im.items():
            for k in members:
                if h in targets.get(k, set()):
                    cnt += 1
                    break  # count intervals, not individual divisors
        h_div_count[h] = cnt

    # Bin by div_count
    div_bins = defaultdict(list)
    for h, cnt in h_div_count.items():
        div_bins[cnt].append(h)

    for cnt in sorted(div_bins.keys()):
        bin_targets = div_bins[cnt]
        n_bin = len(bin_targets)
        # What fraction of these targets are in each interval?
        if cnt >= 2:
            # These are the shared targets; count contributing interval pairs
            pair_counts = Counter()
            for h in bin_targets:
                jset = h_to_intervals.get(h, set())
                sorted_j = sorted(jset)
                for a in range(len(sorted_j)):
                    for b in range(a + 1, len(sorted_j)):
                        pair_counts[(sorted_j[a], sorted_j[b])] += 1
            top_pair = pair_counts.most_common(1)
            top_str = f"top pair: {top_pair[0]}" if top_pair else ""
            print(f"    mu={cnt}: {n_bin} targets. {top_str}")
        else:
            print(f"    mu={cnt}: {n_bin} targets (unique to one interval)")

    return {
        'n_multi': n_multi, 'frac_multi': frac_multi,
        'mu_dist': dict(mu_dist), 'spread_dist': dict(spread_dist),
        'adjacent_frac': adjacent_count / n_multi if n_multi > 0 else 0,
        'pair_CFs': pair_CFs
    }


# ================================================================
#  PART E: THE KEY ANALYTIC RATIO
# ================================================================

def part_e(ctx):
    print("\n" + "=" * 72)
    print("  PART E: THE KEY ANALYTIC RATIO (EXCESS_qi / SURPLUS)")
    print("=" * 72)

    im = ctx['interval_members']
    inh = ctx['interval_NH']
    targets = ctx['targets']
    pool = ctx['pool_smooth']
    H_smooth = ctx['H_smooth']
    H_size = len(H_smooth)

    sorted_jjs = sorted(im.keys())

    results_e = []

    # ---- Test 1: T = S_+ (full pool) ----
    surplus_full = sum(len(inh[jj]) - len(im[jj]) for jj in sorted_jjs)
    excess_qi_full = 0.0
    for i in range(len(sorted_jjs)):
        j1 = sorted_jjs[i]
        for ii in range(i + 1, len(sorted_jjs)):
            j2 = sorted_jjs[ii]
            excess_qi_full += len(inh[j1]) * len(inh[j2]) / H_size

    ratio_full = excess_qi_full / surplus_full if surplus_full > 0 else float('inf')
    print(f"\n  1. T = S_+ (full pool):")
    print(f"     SURPLUS = {surplus_full}, EXCESS_qi = {excess_qi_full:.1f}")
    print(f"     EXCESS_qi / SURPLUS = {ratio_full:.6f}")
    print(f"     Margin: SURPLUS - EXCESS_qi = {surplus_full - excess_qi_full:.1f}")
    print(f"     Hall via QI? {'YES' if ratio_full < 1 else 'NO'}")
    results_e.append({'name': 'Full S_+', 'surplus': surplus_full,
                       'excess_qi': excess_qi_full, 'ratio': ratio_full})

    # ---- Test 2: Top-2 intervals ----
    if len(sorted_jjs) >= 2:
        top2 = sorted(sorted_jjs, reverse=True)[:2]
        surplus_t2 = sum(len(inh[jj]) - len(im[jj]) for jj in top2)
        excess_qi_t2 = len(inh[top2[0]]) * len(inh[top2[1]]) / H_size
        ratio_t2 = excess_qi_t2 / surplus_t2 if surplus_t2 > 0 else float('inf')
        print(f"\n  2. Top-2 intervals (I_{top2[0]}, I_{top2[1]}):")
        print(f"     |T|={sum(len(im[jj]) for jj in top2)}")
        print(f"     SURPLUS = {surplus_t2}, EXCESS_qi = {excess_qi_t2:.1f}")
        print(f"     EXCESS_qi / SURPLUS = {ratio_t2:.6f}")
        print(f"     Hall via QI? {'YES' if ratio_t2 < 1 else 'NO'}")
        results_e.append({'name': 'Top-2', 'surplus': surplus_t2,
                           'excess_qi': excess_qi_t2, 'ratio': ratio_t2})

    # ---- Test 3: Top-3 intervals ----
    if len(sorted_jjs) >= 3:
        top3 = sorted(sorted_jjs, reverse=True)[:3]
        surplus_t3 = sum(len(inh[jj]) - len(im[jj]) for jj in top3)
        excess_qi_t3 = 0.0
        for i in range(len(top3)):
            for ii in range(i + 1, len(top3)):
                excess_qi_t3 += len(inh[top3[i]]) * len(inh[top3[ii]]) / H_size
        ratio_t3 = excess_qi_t3 / surplus_t3 if surplus_t3 > 0 else float('inf')
        print(f"\n  3. Top-3 intervals ({top3}):")
        print(f"     SURPLUS = {surplus_t3}, EXCESS_qi = {excess_qi_t3:.1f}")
        print(f"     EXCESS_qi / SURPLUS = {ratio_t3:.6f}")
        print(f"     Hall via QI? {'YES' if ratio_t3 < 1 else 'NO'}")
        results_e.append({'name': 'Top-3', 'surplus': surplus_t3,
                           'excess_qi': excess_qi_t3, 'ratio': ratio_t3})

    # ---- Test 4: Adversarial T via greedy on full pool ----
    max_greedy = min(len(pool), 500)
    T_adv, NH_adv, min_rat = greedy_minimize_ratio(pool, targets, max_size=max_greedy)

    # Partition T_adv into intervals
    T_by_jj = defaultdict(list)
    for k in T_adv:
        for jj, members in im.items():
            if k in set(members):
                T_by_jj[jj].append(k)
                break

    surplus_adv = 0
    NH_by_jj = {}
    for jj, T_j in T_by_jj.items():
        nh_j = compute_NH(T_j, targets)
        NH_by_jj[jj] = nh_j
        surplus_adv += len(nh_j) - len(T_j)

    excess_qi_adv = 0.0
    jjs_adv = sorted(T_by_jj.keys())
    for i in range(len(jjs_adv)):
        j1 = jjs_adv[i]
        for ii in range(i + 1, len(jjs_adv)):
            j2 = jjs_adv[ii]
            excess_qi_adv += len(NH_by_jj[j1]) * len(NH_by_jj[j2]) / H_size

    ratio_adv = excess_qi_adv / surplus_adv if surplus_adv > 0 else float('inf')
    print(f"\n  4. Adversarial T (greedy, |T|={len(T_adv)}):")
    print(f"     Min |NH(T)|/|T| = {min_rat:.4f}")
    print(f"     Intervals used: {jjs_adv}")
    for jj in jjs_adv:
        print(f"       I_{jj}: |T_j|={len(T_by_jj[jj])}, |NH_j|={len(NH_by_jj[jj])}")
    print(f"     SURPLUS = {surplus_adv}, EXCESS_qi = {excess_qi_adv:.1f}")
    print(f"     EXCESS_qi / SURPLUS = {ratio_adv:.6f}")
    print(f"     Hall via QI? {'YES' if ratio_adv < 1 else 'NO'}")
    results_e.append({'name': 'Greedy adversarial', 'surplus': surplus_adv,
                       'excess_qi': excess_qi_adv, 'ratio': ratio_adv,
                       'min_ratio': min_rat})

    # ---- Test 5: Each adjacent pair (greedy within pair) ----
    print(f"\n  5. Adjacent-pair adversarial:")
    worst_pair_ratio = 0
    for idx in range(len(sorted_jjs) - 1):
        j1 = sorted_jjs[idx]
        j2 = sorted_jjs[idx + 1]
        if abs(j1 - j2) != 1:
            continue
        pool_2 = im[j1] + im[j2]
        if len(pool_2) < 3:
            continue
        tgt_2 = {k: targets.get(k, set()) for k in pool_2}
        max_sz = min(len(pool_2), 200)
        T_2, NH_2, mr_2 = greedy_minimize_ratio(pool_2, tgt_2, max_size=max_sz)

        T_j1 = [k for k in T_2 if k in set(im[j1])]
        T_j2 = [k for k in T_2 if k in set(im[j2])]
        NH_j1 = compute_NH(T_j1, targets)
        NH_j2 = compute_NH(T_j2, targets)
        s_pair = (len(NH_j1) - len(T_j1)) + (len(NH_j2) - len(T_j2))
        e_qi_pair = len(NH_j1) * len(NH_j2) / H_size if H_size > 0 else 0
        r_pair = e_qi_pair / s_pair if s_pair > 0 else float('inf')

        if r_pair > worst_pair_ratio and r_pair != float('inf'):
            worst_pair_ratio = r_pair

        print(f"     (I_{j1},I_{j2}): |T|={len(T_2)}, SUR={s_pair}, EXC_qi={e_qi_pair:.1f}, "
              f"ratio={r_pair:.4f}, OK={'YES' if r_pair < 1 else 'NO'}")

    print(f"\n  Worst adjacent-pair ratio: {worst_pair_ratio:.6f}")

    # Summary
    all_ratios = [r['ratio'] for r in results_e if r['ratio'] != float('inf')]
    worst = max(all_ratios) if all_ratios else 0
    all_ok = all(r < 1 for r in all_ratios)
    print(f"\n  SUMMARY:")
    print(f"    Worst EXCESS_qi / SURPLUS = {worst:.6f}")
    print(f"    All < 1? {'YES — QI gives Hall' if all_ok else 'NO — QI insufficient'}")
    print(f"    Worst adjacent-pair ratio = {worst_pair_ratio:.6f}")

    return {
        'results': results_e,
        'worst_ratio': worst,
        'worst_pair_ratio': worst_pair_ratio,
        'all_ok': all_ok
    }


# ================================================================
#  PART F: SCALING OF KEY QUANTITIES
# ================================================================

def part_f(all_n_results):
    print("\n\n" + "=" * 72)
    print("  PART F: SCALING OF KEY QUANTITIES")
    print("=" * 72)

    ns = sorted(all_n_results.keys())

    # Table 1: Basic scaling
    print(f"\n  {'n':>8s}  {'delta':>7s}  {'|S+|':>7s}  {'|H|':>8s}  "
          f"{'max_R':>8s}  {'EQ/SUR(S+)':>11s}  {'EQ/SUR(adv)':>12s}  "
          f"{'min|NH|/|T|':>12s}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*11}  {'─'*12}  {'─'*12}")

    for n in ns:
        r = all_n_results[n]
        pa = r['part_a']
        pe = r['part_e']

        # max pairwise R
        R_vals = [p['R'] for p in pa if p['R'] != float('inf')]
        max_R = max(R_vals) if R_vals else 0

        # EXCESS_qi/SURPLUS for full pool
        eq_sur_full = pe['results'][0]['ratio'] if pe['results'] else 0

        # Adversarial ratio
        adv_results = [x for x in pe['results'] if x['name'] == 'Greedy adversarial']
        eq_sur_adv = adv_results[0]['ratio'] if adv_results else 0

        # Min |NH|/|T|
        min_nh_t = adv_results[0].get('min_ratio', 0) if adv_results else 0

        print(f"  {n:>8d}  {r['delta']:>7.3f}  {r['S_plus']:>7d}  {r['H_size']:>8d}  "
              f"{max_R:>8.4f}  {eq_sur_full:>11.6f}  {eq_sur_adv:>12.6f}  "
              f"{min_nh_t:>12.4f}")

    # Table 2: Correlation structure
    print(f"\n  {'n':>8s}  {'frac_mu2':>9s}  {'adj_frac':>9s}  {'mean_CF':>9s}  "
          f"{'max_CF':>8s}  {'SUR/|T|':>8s}  {'EXC/|T|':>8s}  {'net/|T|':>8s}")
    print(f"  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for n in ns:
        r = all_n_results[n]
        pd = r['part_d']
        pc = r['part_c']
        mu_dist = pd.get('mu_dist', {})
        total_h = sum(mu_dist.values())
        frac_mu2 = mu_dist.get(2, 0) / total_h if total_h > 0 else 0

        CFs = [p['CF'] for p in pd.get('pair_CFs', []) if p['CF'] != float('inf')]
        mean_CF = sum(CFs) / len(CFs) if CFs else 0
        max_CF = max(CFs) if CFs else 0

        sur_per_T = pc['surplus_per_T']
        exc_per_T = pc['excess_per_T']
        net_per_T = sur_per_T - exc_per_T

        adj_frac = pd.get('adjacent_frac', 0)

        print(f"  {n:>8d}  {frac_mu2:>9.4f}  {adj_frac:>9.4f}  {mean_CF:>9.4f}  "
              f"{max_CF:>8.4f}  {sur_per_T:>8.4f}  {exc_per_T:>8.4f}  {net_per_T:>8.4f}")

    # Table 3: Quasi-independence validity
    print(f"\n  {'n':>8s}  {'actual_EXC':>11s}  {'qi_EXC':>11s}  {'actual/qi':>10s}  "
          f"{'SUR>EXC':>8s}  {'SUR>qi':>7s}")
    print(f"  {'─'*8}  {'─'*11}  {'─'*11}  {'─'*10}  {'─'*8}  {'─'*7}")

    for n in ns:
        r = all_n_results[n]
        pc = r['part_c']
        print(f"  {n:>8d}  {pc['excess_actual']:>11d}  {pc['excess_qi']:>11.1f}  "
              f"{pc['qi_ratio']:>10.4f}  "
              f"{'YES' if pc['excess_over_surplus'] < 1 else 'NO':>8s}  "
              f"{'YES' if pc['qi_over_surplus'] < 1 else 'NO':>7s}")

    # Key findings
    print(f"\n  KEY FINDINGS:")

    # 1. Is R approx 1?
    all_max_R = []
    for n in ns:
        R_vals = [p['R'] for p in all_n_results[n]['part_a'] if p['R'] != float('inf')]
        if R_vals:
            all_max_R.append(max(R_vals))
    if all_max_R:
        print(f"    max pairwise R across all n: {max(all_max_R):.4f}")
        if max(all_max_R) < 2.0:
            print(f"    R is moderately close to 1 — quasi-independence roughly holds")
        else:
            print(f"    R significantly > 1 — POSITIVE CORRELATION present")

    # 2. Does QI give Hall?
    all_qi_ok = all(all_n_results[n]['part_e']['all_ok'] for n in ns)
    print(f"    QI gives Hall for all T at all n? {'YES' if all_qi_ok else 'NO'}")

    # 3. Scaling trend
    if len(ns) >= 2:
        first_ratio = all_n_results[ns[0]]['part_e']['worst_ratio']
        last_ratio = all_n_results[ns[-1]]['part_e']['worst_ratio']
        if last_ratio < first_ratio:
            print(f"    Worst EXCESS_qi/SURPLUS: {first_ratio:.4f} (n={ns[0]}) -> {last_ratio:.4f} (n={ns[-1]}) IMPROVING")
        else:
            print(f"    Worst EXCESS_qi/SURPLUS: {first_ratio:.4f} (n={ns[0]}) -> {last_ratio:.4f} (n={ns[-1]}) WORSENING")


# ================================================================
#  MAIN
# ================================================================

def run_all(n):
    print("\n" + "#" * 72)
    print(f"# Z40: QUASI-INDEPENDENCE OF TARGET OVERLAP — n = {n}")
    print("#" * 72)

    t0 = time.time()
    ctx = setup_n(n)
    if ctx is None:
        print(f"  Setup failed for n={n}")
        return None

    print(f"\n  Setup: n={n}, L={ctx['L']}, B={ctx['B']}, delta={ctx['delta']:.4f}")
    print(f"  |S_+| = {len(ctx['pool_smooth'])}, |H_smooth| = {len(ctx['H_smooth'])}")
    print(f"  Intervals: {sorted(ctx['interval_members'].keys())}")
    for jj in sorted(ctx['interval_members'].keys()):
        print(f"    I_{jj}: [{2**jj}, {2**(jj+1)}), |I_j| = {len(ctx['interval_members'][jj])}")
    print(f"  Setup time: {time.time()-t0:.2f}s")

    t1 = time.time()
    res_a = part_a(ctx)
    print(f"  Part A time: {time.time()-t1:.2f}s")

    t2 = time.time()
    res_b = part_b(ctx)
    print(f"  Part B time: {time.time()-t2:.2f}s")

    t3 = time.time()
    res_c = part_c(ctx)
    print(f"  Part C time: {time.time()-t3:.2f}s")

    t4 = time.time()
    res_d = part_d(ctx)
    print(f"  Part D time: {time.time()-t4:.2f}s")

    t5 = time.time()
    res_e = part_e(ctx)
    print(f"  Part E time: {time.time()-t5:.2f}s")

    total = time.time() - t0
    print(f"\n  Total time for n={n}: {total:.2f}s")

    return {
        'n': n, 'delta': ctx['delta'], 'B': ctx['B'],
        'S_plus': len(ctx['pool_smooth']), 'H_size': len(ctx['H_smooth']),
        'part_a': res_a, 'part_b': res_b, 'part_c': res_c,
        'part_d': res_d, 'part_e': res_e,
        'time': total
    }


def save_state(all_results):
    state_path = "/home/ashbringer/projects/e710_new_H/states/state_52_z40_quasi_independence.md"
    with open(state_path, 'w') as f:
        f.write("# Z40: QUASI-INDEPENDENCE OF TARGET OVERLAP\n\n")
        f.write("**Date:** 2026-02-24\n\n")
        f.write("## Hypothesis\n\n")
        f.write("For dyadic intervals I_j, I_{j'}, the target overlap satisfies:\n")
        f.write("  |NH(I_j) cap NH(I_{j'})| approx |NH(I_j)| * |NH(I_{j'})| / |H_smooth|\n\n")
        f.write("If this holds, EXCESS_qi = sum_{j<j'} |NH(I_j)|*|NH(I_{j'})|/|H| is an estimate of EXCESS.\n")
        f.write("Hall holds if SURPLUS > EXCESS for all T.\n\n")

        ns = sorted(all_results.keys())

        # Table 1: Pairwise R
        f.write("## Part A: Pairwise Correlation Ratio R = Overlap / Product\n\n")
        f.write("R approx 1 means quasi-independence. R > 1 means positive correlation.\n\n")

        f.write("| n | max R | min R | mean R |\n")
        f.write("|---|-------|-------|--------|\n")
        for n in ns:
            pa = all_results[n]['part_a']
            R_vals = [p['R'] for p in pa if p['R'] != float('inf')]
            if R_vals:
                f.write(f"| {n} | {max(R_vals):.4f} | {min(R_vals):.4f} | {sum(R_vals)/len(R_vals):.4f} |\n")
            else:
                f.write(f"| {n} | N/A | N/A | N/A |\n")

        # Table 2: Key analytic ratio
        f.write("\n## Part E: Key Analytic Ratio (EXCESS_qi / SURPLUS)\n\n")
        f.write("| n | delta | Full S+ | Top-2 | Adversarial | Worst adj-pair | All OK? |\n")
        f.write("|---|-------|---------|-------|-------------|----------------|--------|\n")
        for n in ns:
            pe = all_results[n]['part_e']
            full_r = pe['results'][0]['ratio'] if pe['results'] else 0
            top2_r = pe['results'][1]['ratio'] if len(pe['results']) > 1 else 0
            adv_r = [x['ratio'] for x in pe['results'] if x['name'] == 'Greedy adversarial']
            adv_ratio = adv_r[0] if adv_r else 0
            f.write(f"| {n} | {all_results[n]['delta']:.3f} | {full_r:.4f} | {top2_r:.4f} "
                    f"| {adv_ratio:.4f} | {pe['worst_pair_ratio']:.4f} "
                    f"| {'YES' if pe['all_ok'] else 'NO'} |\n")

        # Table 3: Quasi-independence validity
        f.write("\n## Part C: QI Estimate vs Actual EXCESS\n\n")
        f.write("| n | SURPLUS | EXCESS_actual | EXCESS_qi | actual/qi | SUR>EXC | SUR>qi |\n")
        f.write("|---|---------|---------------|-----------|-----------|---------|--------|\n")
        for n in ns:
            pc = all_results[n]['part_c']
            f.write(f"| {n} | {pc['surplus']} | {pc['excess_actual']} | {pc['excess_qi']:.1f} "
                    f"| {pc['qi_ratio']:.4f} "
                    f"| {'YES' if pc['excess_over_surplus'] < 1 else 'NO'} "
                    f"| {'YES' if pc['qi_over_surplus'] < 1 else 'NO'} |\n")

        # Table 4: Correlation structure
        f.write("\n## Part D: Correlation Structure\n\n")
        f.write("| n | frac mu>=2 | adj_frac | mean_CF | max_CF |\n")
        f.write("|---|------------|----------|---------|--------|\n")
        for n in ns:
            pd = all_results[n]['part_d']
            CFs = [p['CF'] for p in pd.get('pair_CFs', []) if p['CF'] != float('inf')]
            mean_CF = sum(CFs) / len(CFs) if CFs else 0
            max_CF = max(CFs) if CFs else 0
            f.write(f"| {n} | {pd['frac_multi']:.4f} | {pd.get('adjacent_frac',0):.4f} "
                    f"| {mean_CF:.4f} | {max_CF:.4f} |\n")

        # Scaling summary
        f.write("\n## Part F: Scaling Summary\n\n")
        f.write("| n | delta | max_R | EQ/SUR(S+) | EQ/SUR(adv) | SUR/|T| | EXC/|T| | net/|T| |\n")
        f.write("|---|-------|-------|------------|-------------|---------|---------|--------|\n")
        for n in ns:
            r = all_results[n]
            pa = r['part_a']
            pe = r['part_e']
            pc = r['part_c']
            R_vals = [p['R'] for p in pa if p['R'] != float('inf')]
            max_R = max(R_vals) if R_vals else 0
            full_r = pe['results'][0]['ratio'] if pe['results'] else 0
            adv_r = [x['ratio'] for x in pe['results'] if x['name'] == 'Greedy adversarial']
            adv_ratio = adv_r[0] if adv_r else 0
            f.write(f"| {n} | {r['delta']:.3f} | {max_R:.4f} | {full_r:.4f} | {adv_ratio:.4f} "
                    f"| {pc['surplus_per_T']:.4f} | {pc['excess_per_T']:.4f} "
                    f"| {pc['surplus_per_T'] - pc['excess_per_T']:.4f} |\n")

        # Key findings
        f.write("\n## Key Findings\n\n")

        all_qi_ok = all(all_results[n]['part_e']['all_ok'] for n in ns)
        f.write(f"- **QI gives Hall for all T at all n?** {'YES' if all_qi_ok else 'NO'}\n")

        all_max_R = []
        for n in ns:
            R_vals = [p['R'] for p in all_results[n]['part_a'] if p['R'] != float('inf')]
            if R_vals:
                all_max_R.append(max(R_vals))
        if all_max_R:
            f.write(f"- **Max pairwise R:** {max(all_max_R):.4f}\n")
            if max(all_max_R) < 2.0:
                f.write(f"  R < 2 at all n: quasi-independence roughly holds\n")
            else:
                f.write(f"  R >= 2 at some n: significant positive correlation\n")

        worst_ratios = [all_results[n]['part_e']['worst_ratio'] for n in ns]
        f.write(f"- **Worst EXCESS_qi/SURPLUS:** {max(worst_ratios):.4f}\n")
        if max(worst_ratios) < 1:
            f.write(f"  All < 1: QI bound suffices for Hall\n")
        else:
            f.write(f"  Some >= 1: QI bound alone insufficient\n")

        if len(ns) >= 2:
            first = worst_ratios[0]
            last = worst_ratios[-1]
            if last < first:
                f.write(f"- **Trend:** Improving ({first:.4f} -> {last:.4f})\n")
            else:
                f.write(f"- **Trend:** Worsening ({first:.4f} -> {last:.4f})\n")

        # Interpretation for proof
        f.write("\n## Proof Implications\n\n")
        if all_qi_ok:
            f.write("The quasi-independence estimate of cross-interval overlap is always smaller\n")
            f.write("than the per-interval surplus. This means:\n")
            f.write("1. SURPLUS - EXCESS_qi > 0 for all T tested\n")
            f.write("2. If actual EXCESS <= EXCESS_qi (positive correlation absent), Hall holds\n")
            f.write("3. Need to verify R <= 1 or bound the excess correlation analytically\n")
        else:
            f.write("The quasi-independence bound does NOT always suffice.\n")
            f.write("Need either:\n")
            f.write("1. Tighter overlap bound (exploiting negative correlation)\n")
            f.write("2. Different proof strategy for cross-interval contribution\n")
            f.write("3. Capacity-splitting or weighted Hall approach\n")

    print(f"\n  State saved to {state_path}")


def main():
    print("=" * 72)
    print("ERDOS 710 -- Z40: QUASI-INDEPENDENCE OF TARGET OVERLAP")
    print("=" * 72)
    print(f"Date: 2026-02-24")
    print(f"C_TARGET = {C_TARGET:.6f}, EPS = {EPS}")

    n_values = [1000, 2000, 5000, 10000, 20000, 50000, 100000]

    all_results = {}
    for n in n_values:
        try:
            res = run_all(n)
            if res is not None:
                all_results[n] = res
        except Exception as e:
            print(f"\n  ERROR for n={n}: {e}")
            import traceback
            traceback.print_exc()

    # Part F: Scaling across all n
    if all_results:
        part_f(all_results)

    # Save state
    if all_results:
        save_state(all_results)

    print("\n" + "=" * 72)
    print("  Z40 COMPLETE")
    print("=" * 72)


if __name__ == '__main__':
    main()
