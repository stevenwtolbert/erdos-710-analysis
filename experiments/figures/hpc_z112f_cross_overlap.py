#!/usr/bin/env python3
"""
Z112f: Cross-interval overlap + fractional matching analysis

FMC (Σ 1/α_j ≤ 1) FAILS at all tested n. But HK confirms Hall at every n.
FMC is SUFFICIENT but not NECESSARY. We need a tighter analytic tool.

Approach 1: DEGREE-BASED FRACTIONAL MATCHING
  Assign weight w_k = 1/d(k) for each k ∈ V. Then:
  - Left coverage: each k gets total weight 1 ✓
  - Right capacity: each target h gets load = Σ_{k|h, k∈V} 1/d(k)
  Need: max_h load(h) ≤ 1.

Approach 2: CROSS-INTERVAL OVERLAP (Inclusion-Exclusion)
  For T = ∪T_j with T_j ⊆ I_j:
  |NH(T)| ≥ Σ |NH(T_j)| - Σ_{j<j'} |NH(T_j) ∩ NH(T_{j'})|
           ≥ Σ α_j|T_j| - OVERLAP

  If OVERLAP ≤ (min α_j - 1)|T|, then Hall holds.

Approach 3: MAXIMUM MULTIPLICITY
  μ(h) = |{j : h ∈ NH(I_j)}|
  |NH(T)| ≥ (Σ α_j|T_j|) / μ_max
  Need: min α_j > μ_max.

Compute all three at various n.
"""

import math
from collections import defaultdict

C_TARGET = 2 / math.exp(0.5) + 0.05

def compute_params(n):
    ln_n = math.log(n) if n > 1 else 1
    ln_ln_n = math.log(ln_n) if ln_n > 1 else 1
    L = int(math.ceil(C_TARGET * n * math.sqrt(ln_n / ln_ln_n)))
    M = L - n
    N = n // 2
    delta = 2.0 * M / n - 1
    return L, M, N, delta

def get_degree(n, k, M):
    L_val = M + n
    j_hi = (n + L_val) // k
    j_lo = (2 * n) // k + 1
    count = 0
    for j in range(j_lo, j_hi + 1):
        m = k * j
        if 2 * n < m <= n + L_val:
            count += 1
    return count

def get_targets(n, k, M):
    L_val = M + n
    targets = set()
    j_lo = (2 * n) // k + 1
    j_hi = (n + L_val) // k
    for j in range(j_lo, j_hi + 1):
        m = k * j
        if 2 * n < m <= n + L_val:
            targets.add(m)
    return targets


# ============================================================
print("=" * 110, flush=True)
print("Z112f: CROSS-INTERVAL OVERLAP + FRACTIONAL MATCHING ANALYSIS", flush=True)
print("=" * 110, flush=True)


# ============================================================
# PART 1: DEGREE-BASED FRACTIONAL MATCHING
# ============================================================
print("\n" + "=" * 90, flush=True)
print("PART 1: DEGREE-BASED FRACTIONAL MATCHING (max_h Σ 1/d(k) ≤ 1?)", flush=True)
print("=" * 90, flush=True)

for n in [5000, 10000, 15000, 20000, 30000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    L_val = M + n

    # Compute degrees for all k ∈ (B, N]
    deg_cache = {}
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)

    # For each target h ∈ (2n, n+L], compute load = Σ_{k|h, k∈V} 1/d(k)
    load = {}
    divisor_count = {}  # number of divisors of h in V

    for h in range(2 * n + 1, L_val + 1):
        total_load = 0.0
        div_count = 0
        # Find all k ∈ V that divide h
        k = 1
        while k * k <= h:
            if h % k == 0:
                # k divides h
                if B < k <= N and k in deg_cache and deg_cache[k] > 0:
                    total_load += 1.0 / deg_cache[k]
                    div_count += 1
                # h/k divides h
                kp = h // k
                if kp != k and B < kp <= N and kp in deg_cache and deg_cache[kp] > 0:
                    total_load += 1.0 / deg_cache[kp]
                    div_count += 1
            k += 1
        load[h] = total_load
        divisor_count[h] = div_count

    # Find max load
    max_load = 0
    max_h = 0
    max_div = 0
    loads = sorted(load.values(), reverse=True)

    for h, l in load.items():
        if l > max_load:
            max_load = l
            max_h = h
            max_div = divisor_count[h]

    # Distribution of loads
    load_vals = list(load.values())
    above_1 = sum(1 for l in load_vals if l > 1.0)
    above_08 = sum(1 for l in load_vals if l > 0.8)
    above_05 = sum(1 for l in load_vals if l > 0.5)
    avg_load = sum(load_vals) / len(load_vals) if load_vals else 0

    print(f"\n  n={n}: δ={delta:.3f}, |V|={N-B}, M={M}", flush=True)
    print(f"    max load = {max_load:.4f} at h={max_h} (τ_V={max_div})", flush=True)
    print(f"    avg load = {avg_load:.4f}", flush=True)
    print(f"    >1.0: {above_1}/{len(load_vals)}, >0.8: {above_08}, >0.5: {above_05}", flush=True)
    print(f"    top 5 loads: {[f'{l:.4f}' for l in loads[:5]]}", flush=True)

    # Show details for max-load target
    if max_h > 0:
        divisors_of_max = []
        k = 1
        while k * k <= max_h:
            if max_h % k == 0:
                if B < k <= N and k in deg_cache and deg_cache[k] > 0:
                    divisors_of_max.append((k, deg_cache[k]))
                kp = max_h // k
                if kp != k and B < kp <= N and kp in deg_cache and deg_cache[kp] > 0:
                    divisors_of_max.append((kp, deg_cache[kp]))
            k += 1
        divisors_of_max.sort()
        print(f"    worst h={max_h}: divisors (k, d(k)): {divisors_of_max[:10]}", flush=True)

    # Verdict
    if max_load <= 1.0:
        print(f"    FRACTIONAL MATCHING EXISTS ✓", flush=True)
    else:
        print(f"    Simple 1/d(k) assignment FAILS (max_load > 1)", flush=True)


# ============================================================
# PART 2: CROSS-INTERVAL OVERLAP (per interval pair)
# ============================================================
print("\n\n" + "=" * 90, flush=True)
print("PART 2: CROSS-INTERVAL OVERLAP", flush=True)
print("=" * 90, flush=True)

for n in [10000, 20000, 30000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    L_val = M + n

    print(f"\n{'='*70}", flush=True)
    print(f"n = {n}, δ = {delta:.3f}, N = {N}, B = {B}", flush=True)

    # Build intervals and their target sets
    j_min = int(math.ceil(math.log2(B + 1)))
    j_max = int(math.floor(math.log2(N)))

    intervals = {}
    target_sets = {}
    for j in range(j_min, j_max + 1):
        lo = max(2**j, B + 1)
        hi = min(2**(j+1) - 1, N)
        elements = list(range(lo, hi + 1))
        if elements:
            intervals[j] = elements
            # Compute union of targets for ALL elements in interval
            targets = set()
            for k in elements:
                targets.update(get_targets(n, k, M))
            target_sets[j] = targets

    print(f"\nInterval sizes and target set sizes:", flush=True)
    for j in sorted(intervals.keys()):
        elems = intervals[j]
        tgts = target_sets[j]
        avg_deg = len(tgts) / len(elems) if elems else 0
        print(f"  j={j}: [2^{j}, 2^{j+1}) |I|={len(elems):>6d}, |NH(I)|={len(tgts):>7d}, "
              f"CS={len(tgts)/len(elems):.2f}", flush=True)

    # Pairwise overlaps
    sorted_js = sorted(intervals.keys())
    print(f"\nPairwise overlaps |NH(I_j) ∩ NH(I_j')|:", flush=True)

    total_overlap = 0
    overlap_matrix = {}
    for idx, j1 in enumerate(sorted_js):
        for j2 in sorted_js[idx+1:]:
            overlap = len(target_sets[j1] & target_sets[j2])
            overlap_matrix[(j1, j2)] = overlap
            total_overlap += overlap
            if overlap > 0:
                # Fraction of each interval's targets
                frac1 = overlap / len(target_sets[j1]) if target_sets[j1] else 0
                frac2 = overlap / len(target_sets[j2]) if target_sets[j2] else 0
                print(f"  ({j1},{j2}): overlap={overlap:>6d}, "
                      f"{frac1:.3f} of NH(I_{j1}), {frac2:.3f} of NH(I_{j2})", flush=True)

    total_targets_union = len(set().union(*target_sets.values())) if target_sets else 0
    total_targets_sum = sum(len(ts) for ts in target_sets.values())

    print(f"\n  Total targets (union): {total_targets_union}", flush=True)
    print(f"  Total targets (sum):   {total_targets_sum}", flush=True)
    print(f"  Total pairwise overlap: {total_overlap}", flush=True)
    print(f"  Overlap/union ratio: {total_overlap/total_targets_union:.4f}" if total_targets_union > 0 else "", flush=True)

    # Per-interval surplus needed
    total_surplus = sum(len(target_sets[j]) - len(intervals[j]) for j in sorted_js)
    print(f"  Total surplus (Σ|NH(I_j)|-|I_j|): {total_surplus}", flush=True)
    print(f"  Surplus ≥ pairwise overlap? {total_surplus >= total_overlap}", flush=True)


# ============================================================
# PART 3: MAXIMUM TARGET MULTIPLICITY μ
# ============================================================
print("\n\n" + "=" * 90, flush=True)
print("PART 3: MAXIMUM TARGET MULTIPLICITY μ(h)", flush=True)
print("=" * 90, flush=True)

for n in [10000, 20000, 30000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    L_val = M + n

    j_min = int(math.ceil(math.log2(B + 1)))
    j_max = int(math.floor(math.log2(N)))

    # For each target h, count how many intervals have a divisor of h
    mu = {}
    for h in range(2 * n + 1, L_val + 1):
        interval_set = set()
        k = 1
        while k * k <= h:
            if h % k == 0:
                for kk in [k, h // k]:
                    if B < kk <= N:
                        # Which interval?
                        j = int(math.log2(kk)) if kk > 0 else 0
                        if j_min <= j <= j_max:
                            interval_set.add(j)
            k += 1
        mu[h] = len(interval_set)

    mu_vals = list(mu.values())
    mu_max = max(mu_vals) if mu_vals else 0
    mu_avg = sum(mu_vals) / len(mu_vals) if mu_vals else 0
    mu_dist = defaultdict(int)
    for v in mu_vals:
        mu_dist[v] += 1

    print(f"\n  n={n}: μ_max = {mu_max}, μ_avg = {mu_avg:.3f}", flush=True)
    print(f"    distribution: {dict(sorted(mu_dist.items()))}", flush=True)

    # Find the worst h (highest multiplicity)
    worst_hs = [h for h in mu if mu[h] == mu_max]
    if worst_hs:
        h_ex = worst_hs[0]
        # Show its divisors and their intervals
        divs_info = []
        k = 1
        while k * k <= h_ex:
            if h_ex % k == 0:
                for kk in [k, h_ex // k]:
                    if B < kk <= N:
                        j = int(math.log2(kk))
                        d = get_degree(n, kk, M)
                        divs_info.append((kk, j, d))
            k += 1
        divs_info.sort()
        print(f"    worst h={h_ex}: divisors (k, interval, deg): {divs_info[:8]}", flush=True)


# ============================================================
# PART 4: MODIFIED FRACTIONAL MATCHING (optimize weights)
# ============================================================
print("\n\n" + "=" * 90, flush=True)
print("PART 4: CAN WE REDISTRIBUTE WEIGHTS TO GET max_load ≤ 1?", flush=True)
print("=" * 90, flush=True)

for n in [10000, 30000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    L_val = M + n

    # Compute degrees
    deg_cache = {}
    target_cache = {}
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)

    # For each target h, find its divisors in V and their degrees
    target_divisors = {}  # h -> list of (k, d(k))
    for h in range(2 * n + 1, L_val + 1):
        divs = []
        k = 1
        while k * k <= h:
            if h % k == 0:
                for kk in [k, h // k]:
                    if B < kk <= N and kk in deg_cache and deg_cache[kk] > 0:
                        divs.append((kk, deg_cache[kk]))
            k += 1
        target_divisors[h] = divs

    # Find targets with highest load under 1/d(k) assignment
    high_load_targets = []
    for h, divs in target_divisors.items():
        load = sum(1.0 / d for _, d in divs)
        if load > 0.8:
            high_load_targets.append((load, h, divs))

    high_load_targets.sort(reverse=True)

    print(f"\n  n={n}: targets with load > 0.8 under 1/d(k):", flush=True)
    for load, h, divs in high_load_targets[:5]:
        divs_str = [(k, d) for k, d in sorted(divs)]
        print(f"    h={h}: load={load:.4f}, divisors={divs_str}", flush=True)

    # Check: among the worst-load targets, what is the structure?
    # Key question: can we reduce load on overloaded targets by
    # giving those k's LESS weight and compensating elsewhere?

    # The LP feasibility: for each k, Σ_h x_{kh} = 1 (cover k)
    # For each h, Σ_k x_{kh} ≤ 1 (capacity)
    # x_{kh} ≥ 0, x_{kh} = 0 if k∤h

    # This LP is feasible iff Hall holds. So if HK works, the LP is feasible.
    # But can we find a CLOSED-FORM feasible solution (provable)?

    if high_load_targets:
        max_l = high_load_targets[0][0]
        n_above_1 = sum(1 for l, _, _ in high_load_targets if l > 1.0)
        print(f"    max load = {max_l:.4f}, targets with load > 1: {n_above_1}", flush=True)
    else:
        print(f"    All loads ≤ 0.8 → FRACTIONAL MATCHING EXISTS ✓", flush=True)


# ============================================================
# PART 5: KEY DIAGNOSTIC — what makes the worst targets bad?
# ============================================================
print("\n\n" + "=" * 90, flush=True)
print("PART 5: STRUCTURE OF WORST-CASE TARGETS", flush=True)
print("=" * 90, flush=True)

for n in [30000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    L_val = M + n

    deg_cache = {}
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)

    # Find top-20 worst targets by load
    worst = []
    for h in range(2 * n + 1, L_val + 1):
        load = 0.0
        divs = []
        k = 1
        while k * k <= h:
            if h % k == 0:
                for kk in [k, h // k]:
                    if B < kk <= N and kk in deg_cache and deg_cache[kk] > 0:
                        load += 1.0 / deg_cache[kk]
                        divs.append((kk, deg_cache[kk]))
            k += 1
        worst.append((load, h, divs))

    worst.sort(reverse=True)

    print(f"\n  n={n}: Top 20 worst-load targets:", flush=True)
    for idx, (load, h, divs) in enumerate(worst[:20]):
        divs.sort()
        # Factorize h
        hh = h
        factors = []
        p = 2
        while p * p <= hh:
            while hh % p == 0:
                factors.append(p)
                hh //= p
            p += 1
        if hh > 1:
            factors.append(hh)

        print(f"    #{idx+1}: h={h} = {'·'.join(str(f) for f in factors)}, load={load:.4f}, "
              f"divisors: {[(k, d) for k, d in divs]}", flush=True)


print("\n\nDONE.", flush=True)
