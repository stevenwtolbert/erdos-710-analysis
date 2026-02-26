#!/usr/bin/env python3
"""
ERDŐS 710 — Z54: CS AT WORST SUBSET + GLOBAL ALPHA

Critical questions for the analytic proof:

1. At the α-minimizing subset T*, what is CS(T*) = E₁²/(|T*|·E₂)?
   If CS(T*) ≈ α: the CS bound is tight, and CS_adv ≈ α.
   Then proving Σ 1/CS_adv < 1 is equivalent to proving Σ 1/α < 1.

2. Global greedy α: min_{T ⊆ S₊} |NH(T)|/|T| over ALL of S₊.
   If global α ≥ 1: Hall holds directly (no FMC needed).
   Key question: does the worst T span multiple intervals?

3. d_min ratio: d_min(I_{j-1}) / d_min(I_j) for consecutive intervals.
   If ≈ 2: geometric doubling gives convergent Σ 1/d_min.

4. α/d_min ratio: is there a constant c > 0 with α_j ≥ c·d_min(I_j)?
   If yes with c·Σ 1/d_min < 1: analytic proof via FMC.

5. CS-greedy: run a SEPARATE greedy that minimizes CS(T) at each step.
   Compare CS_adv (from CS-greedy) with α (from α-greedy).
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


def alpha_greedy_with_cs(sources, adj):
    """
    Run α-greedy (add element with fewest new targets at each step).
    At each step, also compute CS(T) = E₁²/(|T|·E₂).

    Returns dict with:
      alpha: minimum |NH(T)|/|T| over all prefixes
      alpha_T: size of worst prefix for alpha
      cs_at_alpha: CS at the worst-alpha prefix
      cs_min: minimum CS over all prefixes
      cs_min_T: size of worst prefix for CS
      details: list of per-step data
    """
    if not sources:
        return {'alpha': float('inf'), 'cs_at_alpha': float('inf'),
                'cs_min': float('inf')}

    # Build reverse index: target -> set of sources
    tgt_to_srcs = defaultdict(set)
    for k in sources:
        for h in adj.get(k, set()):
            tgt_to_srcs[h].add(k)

    new_count = {k: len(adj.get(k, set())) for k in sources}
    NH = set()
    rem = set(sources)

    # CS tracking
    tau = defaultdict(int)  # τ(h) for current T
    E1 = 0
    E2 = 0
    T_size = 0

    min_alpha = float('inf')
    min_alpha_T = 0
    cs_at_min_alpha = 0
    ceff_at_min_alpha = 0

    min_cs = float('inf')
    min_cs_T = 0

    details = []

    for step in range(len(sources)):
        # Find element with fewest new targets (ties: largest k)
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

        # Update E1, E2 incrementally
        # ΔE₂ = Σ_{h ∈ adj[k]} (2·τ_old(h) + 1) = 2·codeg_sum + dk
        codeg_sum = sum(tau[h] for h in adj.get(best_k, set()))
        E1 += dk
        E2 += 2 * codeg_sum + dk

        for h in adj.get(best_k, set()):
            tau[h] += 1

        # Update NH
        newly = adj.get(best_k, set()) - NH
        NH |= newly

        # Compute ratios
        alpha = len(NH) / T_size
        cs = E1 * E1 / (T_size * E2) if E2 > 0 else float('inf')
        c_eff = E2 / E1 if E1 > 0 else 1

        if alpha < min_alpha:
            min_alpha = alpha
            min_alpha_T = T_size
            cs_at_min_alpha = cs
            ceff_at_min_alpha = c_eff

        if cs < min_cs:
            min_cs = cs
            min_cs_T = T_size

        # Update new_count
        for h in newly:
            for k2 in tgt_to_srcs[h]:
                if k2 in rem:
                    new_count[k2] -= 1

        # Record detail for small prefixes and the minimum points
        if T_size <= 30 or T_size == len(sources):
            details.append({
                'T': T_size, 'NH': len(NH), 'alpha': round(alpha, 4),
                'E1': E1, 'E2': E2, 'CS': round(cs, 4),
                'C_eff': round(c_eff, 4), 'new': best_new,
            })

    return {
        'alpha': min_alpha,
        'alpha_T': min_alpha_T,
        'cs_at_alpha': cs_at_min_alpha,
        'ceff_at_alpha': ceff_at_min_alpha,
        'cs_min': min_cs,
        'cs_min_T': min_cs_T,
        'details': details,
    }


def cs_greedy(sources, adj):
    """
    CS-greedy: at each step, add the element that minimizes CS(T ∪ {k}).
    CS(T ∪ {k}) = (E₁ + dk)² / ((|T|+1) · (E₂ + 2·codeg_T(k) + dk))

    Returns the minimum CS over all prefixes.
    """
    if not sources:
        return float('inf'), float('inf')

    tau = defaultdict(int)
    E1 = 0
    E2 = 0
    T_size = 0
    rem = set(sources)

    min_cs = float('inf')
    min_alpha = float('inf')
    NH = set()

    for step in range(len(sources)):
        best_k = None
        best_cs = float('inf')

        for k in rem:
            dk = len(adj.get(k, set()))
            codeg_k = sum(tau[h] for h in adj.get(k, set()))
            new_E1 = E1 + dk
            new_E2 = E2 + 2 * codeg_k + dk
            new_T = T_size + 1
            cs_val = new_E1 * new_E1 / (new_T * new_E2) if new_E2 > 0 else float('inf')

            if cs_val < best_cs or (cs_val == best_cs and (best_k is None or k > best_k)):
                best_cs = cs_val
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

        cs = E1 * E1 / (T_size * E2) if E2 > 0 else float('inf')
        alpha = len(NH) / T_size

        if cs < min_cs:
            min_cs = cs
        if alpha < min_alpha:
            min_alpha = alpha

    return min_cs, min_alpha


def global_greedy_alpha(sources_by_interval, adj):
    """
    Global greedy: across ALL intervals, greedily add element with fewest
    new targets. Track |NH(T)|/|T| and which intervals contribute.
    """
    all_sources = []
    src_interval = {}
    for j, srcs in sources_by_interval.items():
        for k in srcs:
            all_sources.append(k)
            src_interval[k] = j

    if not all_sources:
        return {'alpha': float('inf')}

    tgt_to_srcs = defaultdict(set)
    for k in all_sources:
        for h in adj.get(k, set()):
            tgt_to_srcs[h].add(k)

    new_count = {k: len(adj.get(k, set())) for k in all_sources}
    NH = set()
    rem = set(all_sources)

    min_alpha = float('inf')
    min_alpha_T = 0
    T_size = 0
    interval_count = defaultdict(int)
    min_alpha_intervals = {}

    for step in range(len(all_sources)):
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
        interval_count[src_interval[best_k]] += 1

        newly = adj.get(best_k, set()) - NH
        NH |= newly

        alpha = len(NH) / T_size
        if alpha < min_alpha:
            min_alpha = alpha
            min_alpha_T = T_size
            min_alpha_intervals = dict(interval_count)

        for h in newly:
            for k2 in tgt_to_srcs[h]:
                if k2 in rem:
                    new_count[k2] -= 1

    return {
        'alpha': min_alpha,
        'T_size': min_alpha_T,
        'interval_dist': min_alpha_intervals,
    }


def analyze_n(n, verbose=True):
    L, M, B = compute_params(n)
    n_half = n // 2
    nL = int(n + L)
    delta = 2 * M / n - 1

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

    # Dyadic partition
    intervals = defaultdict(list)
    for k in S_plus:
        j = int(math.log2(k))
        intervals[j].append(k)

    js = sorted(intervals.keys())

    # === Per-interval analysis ===
    interval_results = []
    sum_1_alpha = 0
    sum_1_cs_at_alpha = 0
    sum_1_cs_min = 0
    sum_1_cs_greedy = 0
    sum_1_dmin = 0

    for j in js:
        I_j = sorted(intervals[j])
        t = len(I_j)
        if t == 0:
            continue

        degs = [len(adj.get(k, set())) for k in I_j]
        d_min = min(degs)
        d_max = max(degs)
        d_mean = sum(degs) / t

        # α-greedy with CS tracking
        res = alpha_greedy_with_cs(I_j, adj)
        alpha = res['alpha']
        cs_at_alpha = res['cs_at_alpha']
        cs_min_prefix = res['cs_min']

        # CS-greedy (only for small-medium intervals to avoid O(n²·d) cost)
        cs_greedy_val = None
        if t <= 3000:
            csg, csg_alpha = cs_greedy(I_j, adj)
            cs_greedy_val = csg

        # Accumulate sums
        if alpha > 0:
            sum_1_alpha += 1.0 / alpha
        if cs_at_alpha > 0:
            sum_1_cs_at_alpha += 1.0 / cs_at_alpha
        if cs_min_prefix > 0:
            sum_1_cs_min += 1.0 / cs_min_prefix
        if cs_greedy_val is not None and cs_greedy_val > 0:
            sum_1_cs_greedy += 1.0 / cs_greedy_val
        if d_min > 0:
            sum_1_dmin += 1.0 / d_min

        ir = {
            'j': j, 'size': t, 'd_min': d_min, 'd_max': d_max,
            'd_mean': round(d_mean, 2),
            'alpha': round(alpha, 4),
            'alpha_T': res['alpha_T'],
            'cs_at_alpha': round(cs_at_alpha, 4),
            'ceff_at_alpha': round(res['ceff_at_alpha'], 4),
            'cs_min_prefix': round(cs_min_prefix, 4),
            'cs_min_T': res['cs_min_T'],
            'cs_greedy': round(cs_greedy_val, 4) if cs_greedy_val is not None else None,
            'alpha_over_dmin': round(alpha / d_min, 4) if d_min > 0 else None,
            'cs_alpha_ratio': round(cs_at_alpha / alpha, 4) if alpha > 0 else None,
        }
        interval_results.append(ir)

    # d_min ratios between consecutive intervals
    dmin_ratios = []
    for i in range(1, len(interval_results)):
        d_prev = interval_results[i-1]['d_min']
        d_curr = interval_results[i]['d_min']
        if d_curr > 0:
            dmin_ratios.append(round(d_prev / d_curr, 3))

    # === Global greedy ===
    global_res = global_greedy_alpha(intervals, adj)

    return {
        'n': n, 'delta': round(delta, 4), 'B': B,
        'S_plus': len(S_plus), 'H_smooth': len(H_smooth),
        'J': len(js),
        'intervals': interval_results,
        'dmin_ratios': dmin_ratios,
        'sum_1_alpha': round(sum_1_alpha, 6),
        'sum_1_cs_at_alpha': round(sum_1_cs_at_alpha, 6),
        'sum_1_cs_min': round(sum_1_cs_min, 6),
        'sum_1_cs_greedy': round(sum_1_cs_greedy, 6),
        'sum_1_dmin': round(sum_1_dmin, 6),
        'global_alpha': round(global_res['alpha'], 4),
        'global_T': global_res['T_size'],
        'global_intervals': global_res['interval_dist'],
    }


def main():
    print("ERDŐS 710 — Z54: CS AT WORST SUBSET + GLOBAL ALPHA")
    print("=" * 90)
    print()
    print("  Q1: Is CS(T*) ≈ α? (CS tightness at worst subset)")
    print("  Q2: Is global α ≥ 1? (Direct Hall without FMC)")
    print("  Q3: Does d_min double between intervals? (Geometric convergence)")
    print("  Q4: Is α/d_min bounded below? (Analytic bound)")
    print()

    test_ns = [100, 200, 500, 1000, 2000, 3000, 5000, 7500, 10000,
               15000, 20000, 30000, 50000, 75000, 100000, 150000, 200000]

    all_results = []

    for n in test_ns:
        t0 = time.time()
        res = analyze_n(n)
        dt = time.time() - t0

        if res is None:
            continue
        all_results.append(res)

        print(f"\n  n = {n}, δ = {res['delta']}, J = {res['J']}, "
              f"global_α = {res['global_alpha']:.3f} (T={res['global_T']})")
        print(f"  Global worst T intervals: {res['global_intervals']}")
        print(f"  Sums: 1/α = {res['sum_1_alpha']:.4f}, "
              f"1/CS@α = {res['sum_1_cs_at_alpha']:.4f}, "
              f"1/CS_min = {res['sum_1_cs_min']:.4f}, "
              f"1/d_min = {res['sum_1_dmin']:.4f}")
        if res['sum_1_cs_greedy'] > 0:
            print(f"  1/CS_greedy = {res['sum_1_cs_greedy']:.4f}")
        print(f"  d_min ratios (lower→upper): {res['dmin_ratios']}")

        print(f"  {'j':>4} {'|I|':>5} {'d_min':>5} {'α':>8} {'α_T':>4} "
              f"{'CS@α':>8} {'CS/α':>6} {'CS_min':>8} {'CS_g':>8} "
              f"{'α/d':>6} {'C_eff':>6}")
        print("  " + "-" * 85)
        for iv in res['intervals']:
            csg = f"{iv['cs_greedy']:8.3f}" if iv['cs_greedy'] is not None else "    -   "
            print(f"  {iv['j']:>4} {iv['size']:>5} {iv['d_min']:>5} "
                  f"{iv['alpha']:>8.3f} {iv['alpha_T']:>4} "
                  f"{iv['cs_at_alpha']:>8.3f} {iv['cs_alpha_ratio']:>6.3f} "
                  f"{iv['cs_min_prefix']:>8.3f} {csg} "
                  f"{iv['alpha_over_dmin']:>6.3f} {iv['ceff_at_alpha']:>6.3f}")

        print(f"  ({dt:.1f}s)")
        sys.stdout.flush()

    # === Summary Tables ===
    print("\n" + "=" * 90)
    print("  SUMMARY: CS TIGHTNESS (CS@α / α)")
    print("=" * 90)
    print()
    print(f"  {'n':>8} {'J':>3} {'Σ1/α':>8} {'Σ1/CS@α':>9} {'Σ1/CS_g':>9} "
          f"{'Σ1/d_min':>9} {'glb_α':>7} {'min CS/α':>9}")
    print("  " + "-" * 70)

    for r in all_results:
        min_cs_ratio = min(iv['cs_alpha_ratio'] for iv in r['intervals']
                          if iv['cs_alpha_ratio'] is not None)
        csg = f"{r['sum_1_cs_greedy']:>9.4f}" if r['sum_1_cs_greedy'] > 0 else "        -"
        print(f"  {r['n']:>8} {r['J']:>3} {r['sum_1_alpha']:>8.4f} "
              f"{r['sum_1_cs_at_alpha']:>9.4f} {csg} "
              f"{r['sum_1_dmin']:>9.4f} {r['global_alpha']:>7.3f} "
              f"{min_cs_ratio:>9.4f}")

    # Global alpha analysis
    print("\n" + "=" * 90)
    print("  GLOBAL ALPHA ANALYSIS")
    print("=" * 90)
    print()

    all_global_ge_1 = all(r['global_alpha'] >= 1.0 for r in all_results)
    print(f"  Global α ≥ 1 at ALL n: {all_global_ge_1}")

    for r in all_results:
        dist = r['global_intervals']
        top_j = max(dist.keys()) if dist else -1
        total = sum(dist.values())
        top_frac = dist.get(top_j, 0) / total if total > 0 else 0
        print(f"  n={r['n']:>8}: α={r['global_alpha']:.3f}, T={r['global_T']}, "
              f"top_j={top_j} ({top_frac:.0%}), dist={dict(sorted(dist.items()))}")

    # d_min doubling
    print("\n" + "=" * 90)
    print("  D_MIN GEOMETRIC DOUBLING")
    print("=" * 90)
    print()

    for r in all_results:
        if r['dmin_ratios']:
            avg_ratio = sum(r['dmin_ratios']) / len(r['dmin_ratios'])
            min_ratio = min(r['dmin_ratios'])
            print(f"  n={r['n']:>8}: ratios={r['dmin_ratios']}, "
                  f"avg={avg_ratio:.3f}, min={min_ratio:.3f}")

    # α/d_min analysis
    print("\n" + "=" * 90)
    print("  ALPHA / D_MIN RATIO")
    print("=" * 90)
    print()

    for r in all_results:
        ratios = [iv['alpha_over_dmin'] for iv in r['intervals']
                  if iv['alpha_over_dmin'] is not None]
        if ratios:
            min_r = min(ratios)
            # Find which interval has the worst ratio
            worst_j = [iv['j'] for iv in r['intervals']
                      if iv['alpha_over_dmin'] == min_r][0]
            print(f"  n={r['n']:>8}: min α/d_min = {min_r:.4f} at j={worst_j}, "
                  f"all ratios = {ratios}")

    # FMC feasibility
    print("\n" + "=" * 90)
    print("  FMC FEASIBILITY: Σ 1/CS_adv < 1?")
    print("=" * 90)
    print()

    # CS_adv ≈ min(CS-greedy, CS-min-prefix)
    # If both Σ 1/CS_greedy < 1 and Σ 1/CS_min < 1: strong evidence
    all_cs_pass = True
    for r in all_results:
        # Use CS-greedy if available, else CS-min-prefix
        cs_sum = r['sum_1_cs_greedy'] if r['sum_1_cs_greedy'] > 0 else r['sum_1_cs_min']
        status = "PASS" if cs_sum < 1 else "FAIL"
        if cs_sum >= 1:
            all_cs_pass = False
        print(f"  n={r['n']:>8}: Σ 1/CS_adv ≈ {cs_sum:.4f} {status}")

    print(f"\n  ALL PASS (Σ 1/CS_adv < 1): {all_cs_pass}")

    # Write state file
    state_path = "/home/ashbringer/projects/e710_new_H/states/state_65_z54_cs_at_worst.md"
    with open(state_path, 'w') as f:
        f.write("# State 65: Z54 CS at Worst Subset + Global Alpha\n\n")

        f.write("## Key Questions\n")
        f.write("1. CS tightness at worst α subset\n")
        f.write("2. Global α ≥ 1?\n")
        f.write("3. d_min geometric doubling\n")
        f.write("4. α/d_min bounded below?\n\n")

        f.write("## Summary\n\n")
        f.write(f"| n | J | Σ1/α | Σ1/CS@α | Σ1/CSg | Σ1/dmin | glob_α | min CS/α |\n")
        f.write(f"|---|---|------|---------|--------|---------|--------|----------|\n")
        for r in all_results:
            min_csr = min(iv['cs_alpha_ratio'] for iv in r['intervals']
                         if iv['cs_alpha_ratio'] is not None)
            csg = f"{r['sum_1_cs_greedy']:.4f}" if r['sum_1_cs_greedy'] > 0 else "-"
            f.write(f"| {r['n']} | {r['J']} | {r['sum_1_alpha']:.4f} | "
                    f"{r['sum_1_cs_at_alpha']:.4f} | {csg} | "
                    f"{r['sum_1_dmin']:.4f} | {r['global_alpha']:.3f} | "
                    f"{min_csr:.4f} |\n")

        f.write("\n## Global Alpha\n\n")
        f.write(f"Global α ≥ 1 at ALL n: {all_global_ge_1}\n\n")
        for r in all_results:
            f.write(f"n={r['n']}: α={r['global_alpha']:.3f}, T={r['global_T']}, "
                    f"intervals={dict(sorted(r['global_intervals'].items()))}\n")

        f.write("\n## d_min Ratios\n\n")
        for r in all_results:
            if r['dmin_ratios']:
                f.write(f"n={r['n']}: {r['dmin_ratios']}\n")

        f.write("\n## α/d_min Ratios\n\n")
        for r in all_results:
            ratios = [iv['alpha_over_dmin'] for iv in r['intervals']
                      if iv['alpha_over_dmin'] is not None]
            if ratios:
                f.write(f"n={r['n']}: min={min(ratios):.4f}, all={ratios}\n")

        f.write("\n## Per-Interval Detail\n\n")
        for r in all_results:
            f.write(f"\n### n = {r['n']} (δ = {r['delta']})\n")
            f.write(f"Σ1/α = {r['sum_1_alpha']:.4f}\n\n")
            f.write(f"| j | |I| | d_min | α | α_T | CS@α | CS/α | CS_g | α/d | C_eff |\n")
            f.write(f"|---|-----|-------|---|-----|------|------|------|-----|-------|\n")
            for iv in r['intervals']:
                csg = f"{iv['cs_greedy']:.3f}" if iv['cs_greedy'] is not None else "-"
                f.write(f"| {iv['j']} | {iv['size']} | {iv['d_min']} | "
                        f"{iv['alpha']:.3f} | {iv['alpha_T']} | "
                        f"{iv['cs_at_alpha']:.3f} | {iv['cs_alpha_ratio']:.3f} | "
                        f"{csg} | {iv['alpha_over_dmin']:.3f} | "
                        f"{iv['ceff_at_alpha']:.3f} |\n")

    print(f"\n  State file: {state_path}")


if __name__ == '__main__':
    main()
