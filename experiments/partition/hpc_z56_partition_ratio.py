#!/usr/bin/env python3
"""
ERDŐS 710 — Z56: NON-DYADIC PARTITION RATIOS

KEY INSIGHT: With dyadic partition (ratio 2), Σ 1/CS can exceed 1 at J-transitions.
But the geometric series Σ 1/α_j converges as ~ 1/(d_min_bottom · (1 - 1/r)).
Larger r → fewer intervals → smaller sum, even though CS per interval is slightly worse.

Test partition ratios r = 2, 2.5, 3, 4, 6 and compute:
  - Per-interval α (greedy) and CS
  - Σ 1/α and Σ 1/CS for each ratio
  - Find optimal ratio that minimizes max Σ 1/CS across all n

If ratio r=3 or r=4 gives Σ 1/CS < 1 at ALL n (including J-transitions),
then we have a viable analytic proof path:
  - C_eff ≤ 2r (from universal bound C_eff ≤ 2·d_max/d_min)
  - d̄ → ∞ at each interval
  - CS = d̄/C_eff ≥ d̄/(2r) → ∞
  - Σ 1/CS = Σ C_eff/d̄ = geometric series with ratio < 1
"""

import math
import time
import sys
from collections import defaultdict

C_TARGET = 2 / math.e**0.5
EPS = 0.05


def compute_params(n):
    ln_n = math.log(n)
    ln_ln_n = math.log(ln_n) if ln_n > 1 else 0.1
    L = (C_TARGET + EPS) * n * math.sqrt(ln_n / ln_ln_n)
    M = n + L
    B = int(math.sqrt(M))
    return L, M, B


def sieve_primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            for mult in range(p*p, limit + 1, p):
                sieve[mult] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]


def get_smooth_numbers_fast(B, lo, hi):
    if hi <= lo:
        return []
    size = hi - lo
    remaining = list(range(lo + 1, hi + 1))
    primes = sieve_primes(B)
    for p in primes:
        start = lo + 1
        first = start + (-start % p)
        for idx in range(first - lo - 1, size, p):
            while remaining[idx] % p == 0:
                remaining[idx] //= p
    return [lo + 1 + i for i in range(size) if remaining[i] == 1]


def greedy_alpha_with_cs(sources, adj):
    """Run α-greedy, track both α and CS at each prefix."""
    if not sources:
        return float('inf'), float('inf'), float('inf')

    tgt_to_srcs = defaultdict(set)
    for k in sources:
        for h in adj.get(k, set()):
            tgt_to_srcs[h].add(k)

    new_count = {k: len(adj.get(k, set())) for k in sources}
    NH = set()
    rem = set(sources)
    tau = defaultdict(int)
    E1 = 0
    E2 = 0
    T_size = 0
    min_alpha = float('inf')
    cs_at_min_alpha = 0
    min_cs = float('inf')

    for step in range(len(sources)):
        best_k = None
        best_new = float('inf')
        for k in rem:
            nc = new_count[k]
            if nc < best_new or (nc == best_new and (best_k is None or k > best_k)):
                best_new = nc
                best_k = k
        if best_k is None:
            break

        T_size += 1
        rem.discard(best_k)
        dk = len(adj.get(best_k, set()))
        codeg_sum = sum(tau[h] for h in adj.get(best_k, set()))
        E1 += dk
        E2 += 2 * codeg_sum + dk
        for h in adj.get(best_k, set()):
            tau[h] += 1
        newly = adj.get(best_k, set()) - NH
        NH |= newly

        alpha = len(NH) / T_size
        cs = E1 * E1 / (T_size * E2) if E2 > 0 else float('inf')

        if alpha < min_alpha:
            min_alpha = alpha
            cs_at_min_alpha = cs
        if cs < min_cs:
            min_cs = cs

        for h in newly:
            for k2 in tgt_to_srcs[h]:
                if k2 in rem:
                    new_count[k2] -= 1

    return min_alpha, cs_at_min_alpha, min_cs


def compute_cs_full(sources, adj):
    """Compute CS = d̄/C_eff = E₁²/(|I|·E₂) for the full interval."""
    t = len(sources)
    if t == 0:
        return 0, 0, 0

    degs = [len(adj.get(k, set())) for k in sources]
    E1 = sum(degs)
    d_mean = E1 / t
    d_min = min(degs)

    target_count = defaultdict(int)
    for k in sources:
        for h in adj.get(k, set()):
            target_count[h] += 1
    E2 = sum(c**2 for c in target_count.values())

    CS = E1 * E1 / (t * E2) if E2 > 0 else float('inf')
    C_eff = E2 / E1 if E1 > 0 else 1

    return CS, d_min, C_eff


def analyze_n_with_ratio(n, ratio):
    """Analyze n using partition with given ratio."""
    L, M, B = compute_params(n)
    n_half = n // 2
    nL = int(n + L)

    if B < 2 or n_half <= B:
        return None

    S_plus = get_smooth_numbers_fast(B, B, n_half)
    H_smooth = get_smooth_numbers_fast(B, n, nL)
    if not S_plus or not H_smooth:
        return None

    H_set = set(H_smooth)
    adj = {}
    for k in S_plus:
        targets = set()
        lo_mult = n // k + 1
        hi_mult = nL // k
        for m in range(lo_mult, hi_mult + 1):
            h = k * m
            if h in H_set:
                targets.add(h)
        adj[k] = targets

    # Partition by ratio r: I_j = S₊ ∩ [r^j, r^{j+1})
    log_r = math.log(ratio)
    intervals = defaultdict(list)
    for k in S_plus:
        j = int(math.log(k) / log_r)
        intervals[j].append(k)

    js = sorted(intervals.keys())
    J = len(js)

    sum_1_alpha = 0
    sum_1_cs_at_alpha = 0
    sum_1_cs_min = 0
    sum_1_cs_full = 0
    interval_data = []

    for j in js:
        I_j = sorted(intervals[j])
        t = len(I_j)

        alpha, cs_at_alpha, cs_min = greedy_alpha_with_cs(I_j, adj)
        cs_full, d_min, c_eff = compute_cs_full(I_j, adj)

        if alpha > 0 and alpha != float('inf'):
            sum_1_alpha += 1.0 / alpha
        if cs_at_alpha > 0 and cs_at_alpha != float('inf'):
            sum_1_cs_at_alpha += 1.0 / cs_at_alpha
        if cs_min > 0 and cs_min != float('inf'):
            sum_1_cs_min += 1.0 / cs_min
        if cs_full > 0 and cs_full != float('inf'):
            sum_1_cs_full += 1.0 / cs_full

        interval_data.append({
            'j': j, 'size': t, 'd_min': d_min,
            'alpha': round(alpha, 3) if alpha != float('inf') else 'inf',
            'cs_min': round(cs_min, 3) if cs_min != float('inf') else 'inf',
            'cs_full': round(cs_full, 3) if cs_full != float('inf') else 'inf',
            'c_eff': round(c_eff, 3),
        })

    return {
        'n': n, 'ratio': ratio, 'J': J,
        'sum_1_alpha': sum_1_alpha,
        'sum_1_cs_at_alpha': sum_1_cs_at_alpha,
        'sum_1_cs_min': sum_1_cs_min,
        'sum_1_cs_full': sum_1_cs_full,
        'intervals': interval_data,
    }


def main():
    print("ERDŐS 710 — Z56: NON-DYADIC PARTITION RATIOS")
    print("=" * 90)
    print()
    print("  Testing partition ratios r = 2, 2.5, 3, 4, 6")
    print("  Goal: Find r where Σ 1/CS < 1 at ALL n (including J-transitions)")
    print()

    ratios = [2.0, 2.5, 3.0, 4.0, 6.0]

    # Dense scan focused on J-transition points for each ratio
    # Start with a moderate scan, then zoom into trouble spots
    base_ns = list(range(100, 2001, 100))
    base_ns += list(range(2000, 10001, 200))
    base_ns += list(range(10000, 50001, 1000))
    base_ns += list(range(50000, 200001, 2000))
    base_ns = sorted(set(base_ns))

    # Track best/worst for each ratio
    ratio_results = {}
    for r in ratios:
        ratio_results[r] = {
            'max_sum_alpha': 0, 'max_sum_alpha_n': 0,
            'max_sum_cs_min': 0, 'max_sum_cs_min_n': 0,
            'max_sum_cs_full': 0, 'max_sum_cs_full_n': 0,
            'all_results': [],
        }

    t0 = time.time()

    for i, n in enumerate(base_ns):
        for r in ratios:
            res = analyze_n_with_ratio(n, r)
            if res is None:
                continue

            rr = ratio_results[r]
            rr['all_results'].append(res)

            if res['sum_1_alpha'] > rr['max_sum_alpha']:
                rr['max_sum_alpha'] = res['sum_1_alpha']
                rr['max_sum_alpha_n'] = n
            if res['sum_1_cs_min'] > rr['max_sum_cs_min']:
                rr['max_sum_cs_min'] = res['sum_1_cs_min']
                rr['max_sum_cs_min_n'] = n
            if res['sum_1_cs_full'] > rr['max_sum_cs_full']:
                rr['max_sum_cs_full'] = res['sum_1_cs_full']
                rr['max_sum_cs_full_n'] = n

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            pct = 100.0 * (i + 1) / len(base_ns)
            # Print current best for each ratio
            print(f"  [{pct:.0f}% {elapsed:.0f}s] n={n}", end="")
            for r in ratios:
                rr = ratio_results[r]
                print(f"  r={r}: Σ1/α={rr['max_sum_alpha']:.3f} Σ1/CS={rr['max_sum_cs_min']:.3f}", end="")
            print()
            sys.stdout.flush()

    total_time = time.time() - t0

    # Summary
    print("\n" + "=" * 90)
    print("  SUMMARY: PARTITION RATIO COMPARISON")
    print("=" * 90)
    print()
    print(f"  Tested {len(base_ns)} n-values, {len(ratios)} ratios, total time: {total_time:.0f}s")
    print()

    print(f"  {'Ratio':>6} {'J_range':>8} {'max Σ1/α':>10} {'@n':>8} {'max Σ1/CSm':>11} {'@n':>8} {'max Σ1/CSf':>11} {'@n':>8} {'CSm<1?':>6} {'CSf<1?':>6}")
    print("  " + "-" * 85)

    for r in ratios:
        rr = ratio_results[r]
        if not rr['all_results']:
            continue

        j_min = min(res['J'] for res in rr['all_results'])
        j_max = max(res['J'] for res in rr['all_results'])
        j_range = f"{j_min}-{j_max}"

        csm_pass = "YES" if rr['max_sum_cs_min'] < 1.0 else "NO"
        csf_pass = "YES" if rr['max_sum_cs_full'] < 1.0 else "NO"

        print(f"  {r:>6.1f} {j_range:>8} {rr['max_sum_alpha']:>10.4f} {rr['max_sum_alpha_n']:>8} "
              f"{rr['max_sum_cs_min']:>11.4f} {rr['max_sum_cs_min_n']:>8} "
              f"{rr['max_sum_cs_full']:>11.4f} {rr['max_sum_cs_full_n']:>8} "
              f"{csm_pass:>6} {csf_pass:>6}")

    # For the best ratio(s), show per-n detail at critical points
    print("\n" + "=" * 90)
    print("  DETAIL AT CRITICAL n VALUES")
    print("=" * 90)

    for r in ratios:
        rr = ratio_results[r]
        if not rr['all_results']:
            continue

        # Show all n where Σ1/CS_min > 0.90
        high_cs = [res for res in rr['all_results'] if res['sum_1_cs_min'] > 0.90]
        if high_cs:
            print(f"\n  Ratio {r}: n-values with Σ 1/CS_min > 0.90:")
            for res in high_cs[:20]:
                print(f"    n={res['n']:>7} J={res['J']} Σ1/α={res['sum_1_alpha']:.4f} "
                      f"Σ1/CSm={res['sum_1_cs_min']:.4f} Σ1/CSf={res['sum_1_cs_full']:.4f}")

    # For the best ratio, show interval detail at the worst n
    best_ratio = min(ratios, key=lambda r: ratio_results[r]['max_sum_cs_min'] if ratio_results[r]['all_results'] else float('inf'))
    print(f"\n  BEST RATIO: {best_ratio}")
    print(f"  Max Σ 1/CS_min = {ratio_results[best_ratio]['max_sum_cs_min']:.6f}")

    # Show interval detail at worst n for best ratio
    worst_n = ratio_results[best_ratio]['max_sum_cs_min_n']
    worst_res = None
    for res in ratio_results[best_ratio]['all_results']:
        if res['n'] == worst_n:
            worst_res = res
            break

    if worst_res:
        print(f"\n  Interval detail at n={worst_n} (worst case for ratio {best_ratio}):")
        print(f"  {'j':>4} {'|I|':>6} {'d_min':>6} {'α':>8} {'CS_min':>8} {'CS_full':>8} {'C_eff':>6}")
        print("  " + "-" * 55)
        for iv in worst_res['intervals']:
            print(f"  {iv['j']:>4} {iv['size']:>6} {iv['d_min']:>6} "
                  f"{iv['alpha']:>8} {iv['cs_min']:>8} {iv['cs_full']:>8} {iv['c_eff']:>6}")

    # Also show for ratio 2 (baseline) at its worst
    if 2.0 in ratio_results:
        worst_n_2 = ratio_results[2.0]['max_sum_cs_min_n']
        worst_res_2 = None
        for res in ratio_results[2.0]['all_results']:
            if res['n'] == worst_n_2:
                worst_res_2 = res
                break

        if worst_res_2:
            print(f"\n  Baseline (ratio 2) interval detail at n={worst_n_2}:")
            print(f"  {'j':>4} {'|I|':>6} {'d_min':>6} {'α':>8} {'CS_min':>8} {'CS_full':>8} {'C_eff':>6}")
            print("  " + "-" * 55)
            for iv in worst_res_2['intervals']:
                print(f"  {iv['j']:>4} {iv['size']:>6} {iv['d_min']:>6} "
                      f"{iv['alpha']:>8} {iv['cs_min']:>8} {iv['cs_full']:>8} {iv['c_eff']:>6}")

    # Dense zoom around J-transition for best ratio
    if best_ratio != 2.0 and ratio_results[best_ratio]['max_sum_cs_min'] < 1.05:
        print(f"\n\n  DENSE ZOOM for ratio {best_ratio} around worst n...")
        worst_n = ratio_results[best_ratio]['max_sum_cs_min_n']
        zoom_ns = list(range(max(100, worst_n - 5000), worst_n + 5001, 100))
        zoom_max_csm = 0
        zoom_max_n = 0
        zoom_max_alpha = 0
        zoom_max_alpha_n = 0
        for n in zoom_ns:
            res = analyze_n_with_ratio(n, best_ratio)
            if res is None:
                continue
            if res['sum_1_cs_min'] > zoom_max_csm:
                zoom_max_csm = res['sum_1_cs_min']
                zoom_max_n = n
            if res['sum_1_alpha'] > zoom_max_alpha:
                zoom_max_alpha = res['sum_1_alpha']
                zoom_max_alpha_n = n
        print(f"  Dense zoom ({len(zoom_ns)} values around n={worst_n}):")
        print(f"    max Σ 1/CS_min = {zoom_max_csm:.6f} at n = {zoom_max_n}")
        print(f"    max Σ 1/α = {zoom_max_alpha:.6f} at n = {zoom_max_alpha_n}")

    # Write state file
    state_path = "/home/ashbringer/projects/e710_new_H/states/state_67_z56_partition_ratio.md"
    with open(state_path, 'w') as f:
        f.write("# State 67: Z56 Partition Ratio Optimization\n\n")
        f.write(f"Tested {len(base_ns)} n-values with ratios {ratios}\n\n")
        f.write("## Summary\n\n")
        f.write("| Ratio | J range | max Σ1/α | @n | max Σ1/CSm | @n | CSm<1? |\n")
        f.write("|-------|---------|----------|-----|-----------|-----|--------|\n")
        for r in ratios:
            rr = ratio_results[r]
            if not rr['all_results']:
                continue
            j_min = min(res['J'] for res in rr['all_results'])
            j_max = max(res['J'] for res in rr['all_results'])
            csm_pass = "YES" if rr['max_sum_cs_min'] < 1.0 else "NO"
            f.write(f"| {r} | {j_min}-{j_max} | {rr['max_sum_alpha']:.4f} | {rr['max_sum_alpha_n']} | "
                    f"{rr['max_sum_cs_min']:.4f} | {rr['max_sum_cs_min_n']} | {csm_pass} |\n")

        f.write(f"\n## Best Ratio: {best_ratio}\n")
        f.write(f"Max Σ 1/CS_min = {ratio_results[best_ratio]['max_sum_cs_min']:.6f}\n")

    print(f"\n  State file: {state_path}")


if __name__ == '__main__':
    main()
