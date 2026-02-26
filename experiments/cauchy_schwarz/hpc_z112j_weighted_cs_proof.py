#!/usr/bin/env python3
"""
Z112j: WEIGHTED CS PROOF — w_k = 1/d(k)

The weighted CS bound with w_k = 1/d(k) gives:
  |NH(T)| ≥ |T|² / Σ_h load_T(h)²

where load_T(h) = Σ_{k∈T: k|h} 1/d(k).

Hall ⟺ |T|/Σ load² ≥ 1 for all T, i.e., Σ load² ≤ |T|.

Expanding: Σ load² = Σ_k 1/d(k) + 2·Σ_{k<k'} codeg(k,k')/(d(k)·d(k'))

Per-element condition: 1/d(k) + ρ̃(k)/d(k) ≤ 1
where ρ̃(k) = Σ_{k'∈V} codeg(k,k')/d(k')

i.e., ρ̃(k) ≤ d(k) - 1

PART 1: Compute ρ̃(k) for all k, check per-element condition
PART 2: Adversarial greedy to find worst T for weighted CS
PART 3: Compute Σ load² for various T choices
PART 4: Analytic structure — how does ρ̃(k) scale with n?
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
# PART 1: Per-element condition for weighted CS
# ============================================================
print("=" * 100, flush=True)
print("PART 1: PER-ELEMENT CONDITION ρ̃(k) ≤ d(k) - 1", flush=True)
print("=" * 100, flush=True)

print("""
ρ̃(k) = Σ_{k'∈V} codeg(k,k')/d(k')
     = Σ_{h∈NH(k)} Σ_{k'∈V, k'≠k, k'|h} 1/d(k')
     = Σ_{h∈NH(k)} (load(h) - 1/d(k))
     = Σ_{h∈NH(k)} load(h) - 1
     where load(h) = Σ_{k'∈V:k'|h} 1/d(k')
""", flush=True)

for n in [5000, 10000, 20000, 30000, 50000, 75000, 100000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    # Build graph
    deg_cache = {}
    target_cache = {}
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)

    # Compute load(h) = Σ_{k∈V:k|h} 1/d(k) for all targets h
    load = defaultdict(float)
    for k in deg_cache:
        w = 1.0 / deg_cache[k]
        for h in target_cache[k]:
            load[h] += w

    # Compute ρ̃(k) = Σ_{h∈NH(k)} load(h) - 1
    rho_tilde = {}
    for k in deg_cache:
        rho_tilde[k] = sum(load[h] for h in target_cache[k]) - 1.0

    # Per-element condition: ρ̃(k) ≤ d(k) - 1
    # Equivalently: ρ̃(k) - d(k) + 1 ≤ 0
    # Or: margin = d(k) - 1 - ρ̃(k) ≥ 0
    margins = {k: deg_cache[k] - 1 - rho_tilde[k] for k in deg_cache}

    violations = [(k, margins[k]) for k in deg_cache if margins[k] < 0]
    violations.sort(key=lambda x: x[1])

    min_margin = min(margins.values())
    max_rho_tilde = max(rho_tilde.values())
    max_rho_k = max(rho_tilde, key=rho_tilde.get)

    print(f"\n  n={n:>6d}: |V|={len(deg_cache):>6d}, δ={delta:.3f}", flush=True)
    print(f"    max ρ̃ = {max_rho_tilde:.4f} at k={max_rho_k} (d={deg_cache[max_rho_k]})", flush=True)
    print(f"    min margin d-1-ρ̃ = {min_margin:.4f}", flush=True)
    print(f"    violations: {len(violations)}/{len(deg_cache)}", flush=True)

    if violations:
        print(f"    Worst 5:", flush=True)
        for k, m in violations[:5]:
            d = deg_cache[k]
            rt = rho_tilde[k]
            print(f"      k={k}: d={d}, ρ̃={rt:.4f}, margin={m:.4f}", flush=True)

    # Distribution by degree
    by_deg = defaultdict(list)
    for k in deg_cache:
        by_deg[deg_cache[k]].append(margins[k])

    print(f"    Per-degree summary:", flush=True)
    for d in sorted(by_deg.keys())[:15]:
        vals = by_deg[d]
        mn = min(vals)
        avg = sum(vals) / len(vals)
        viol = sum(1 for v in vals if v < 0)
        print(f"      d={d:>3d}: n={len(vals):>5d}, min_margin={mn:>8.4f}, "
              f"avg_margin={avg:>8.4f}, violations={viol}", flush=True)

    # Load distribution
    loads = sorted(load.values(), reverse=True)
    print(f"    Target load stats: max={loads[0]:.4f}, "
          f"p90={loads[len(loads)//10]:.4f}, "
          f"p50={loads[len(loads)//2]:.4f}, "
          f"min={loads[-1]:.4f}", flush=True)
    high_load = sum(1 for l in loads if l > 1.0)
    print(f"    Targets with load > 1: {high_load}/{len(loads)}", flush=True)


# ============================================================
# PART 2: Adversarial greedy for weighted CS
# ============================================================
print("\n\n" + "=" * 100, flush=True)
print("PART 2: ADVERSARIAL GREEDY — find worst T for weighted CS", flush=True)
print("=" * 100, flush=True)

for n in [10000, 30000, 50000, 100000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)

    # Greedy: start empty, add element that minimizes weighted CS ratio
    # Ratio = |T| / Σ_h load_T(h)² for current T
    # Lower ratio = worse for Hall

    T = set()
    load_T = defaultdict(float)
    sum_load_sq = 0.0
    T_size = 0
    candidates = set(deg_cache.keys())

    best_ratio_seen = float('inf')
    best_T_size_at_min = 0

    # Greedy: try adding each candidate, pick the one that gives worst ratio
    # For efficiency, limit to first 2000 elements added
    max_steps = min(2000, len(candidates))

    for step in range(max_steps):
        best_k = None
        best_ratio = float('inf')

        # Sample candidates for speed at large n
        if len(candidates) > 5000:
            import random
            random.seed(42 + step)
            sample = random.sample(list(candidates), 5000)
        else:
            sample = list(candidates)

        for k in sample:
            d_k = deg_cache[k]
            w_k = 1.0 / d_k
            new_size = T_size + 1

            # Change in Σ load²: for each target h of k:
            # load_T(h) increases by w_k
            # load²(h) changes by 2·load_T(h)·w_k + w_k²
            delta_load_sq = 0.0
            for h in target_cache[k]:
                old_load = load_T.get(h, 0.0)
                delta_load_sq += 2 * old_load * w_k + w_k * w_k

            new_sum_load_sq = sum_load_sq + delta_load_sq
            new_ratio = new_size / new_sum_load_sq if new_sum_load_sq > 0 else float('inf')

            if new_ratio < best_ratio:
                best_ratio = new_ratio
                best_k = k

        if best_k is None:
            break

        # Add best_k to T
        T.add(best_k)
        candidates.discard(best_k)
        w_k = 1.0 / deg_cache[best_k]
        for h in target_cache[best_k]:
            old_load = load_T.get(h, 0.0)
            sum_load_sq += 2 * old_load * w_k + w_k * w_k
            load_T[h] += w_k
        T_size += 1

        ratio = T_size / sum_load_sq if sum_load_sq > 0 else float('inf')
        if ratio < best_ratio_seen:
            best_ratio_seen = ratio
            best_T_size_at_min = T_size

        if step < 20 or step % 200 == 0 or step == max_steps - 1:
            if step < 20 or step % 200 == 0:
                print(f"  n={n}, step {step:>4d}: |T|={T_size:>5d}, "
                      f"Σload²={sum_load_sq:.2f}, ratio={ratio:.5f}, "
                      f"added k={best_k} (d={deg_cache[best_k]})", flush=True)

    print(f"  n={n}: WORST weighted CS ratio = {best_ratio_seen:.5f} at |T|={best_T_size_at_min} "
          f"{'≥ 1 ✓' if best_ratio_seen >= 1 else '< 1 ✗'}", flush=True)
    print(flush=True)


# ============================================================
# PART 3: Σ load² for T₀ and T = V
# ============================================================
print("\n\n" + "=" * 100, flush=True)
print("PART 3: Σ load² FOR VARIOUS T CHOICES", flush=True)
print("=" * 100, flush=True)

for n in [10000, 30000, 50000, 100000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)

    # Global load
    load_V = defaultdict(float)
    for k in deg_cache:
        w = 1.0 / deg_cache[k]
        for h in target_cache[k]:
            load_V[h] += w

    sum_load_sq_V = sum(l * l for l in load_V.values())

    # T = all V
    V_size = len(deg_cache)
    ratio_V = V_size / sum_load_sq_V

    # T₀ = weighted-negative elements (ρ̃ > d-1)
    rho_tilde = {k: sum(load_V[h] for h in target_cache[k]) - 1.0 for k in deg_cache}
    T0_w = [k for k in deg_cache if rho_tilde[k] > deg_cache[k] - 1]

    if T0_w:
        load_T0 = defaultdict(float)
        for k in T0_w:
            w = 1.0 / deg_cache[k]
            for h in target_cache[k]:
                load_T0[h] += w
        sum_load_sq_T0 = sum(l * l for l in load_T0.values())
        ratio_T0 = len(T0_w) / sum_load_sq_T0
    else:
        ratio_T0 = float('inf')

    # T = degree-3 only
    T3 = [k for k in deg_cache if deg_cache[k] == 3]
    load_T3 = defaultdict(float)
    for k in T3:
        w = 1.0 / 3
        for h in target_cache[k]:
            load_T3[h] += w
    sum_load_sq_T3 = sum(l * l for l in load_T3.values())
    ratio_T3 = len(T3) / sum_load_sq_T3 if sum_load_sq_T3 > 0 else float('inf')

    print(f"\n  n={n}: |V|={V_size}", flush=True)
    print(f"    T = V:    Σload²={sum_load_sq_V:.2f}, ratio = {ratio_V:.5f} "
          f"{'✓' if ratio_V >= 1 else '✗'}", flush=True)
    print(f"    T = T₀_w: |T₀|={len(T0_w)}, ratio = {ratio_T0:.5f} "
          f"{'✓' if ratio_T0 >= 1 else '✗'}", flush=True)
    print(f"    T = V₃:   |V₃|={len(T3)}, ratio = {ratio_T3:.5f} "
          f"{'✓' if ratio_T3 >= 1 else '✗'}", flush=True)


# ============================================================
# PART 4: How does max load scale?
# ============================================================
print("\n\n" + "=" * 100, flush=True)
print("PART 4: LOAD SCALING — max_h load(h) vs n", flush=True)
print("=" * 100, flush=True)

print("""
load(h) = Σ_{k∈V: k|h} 1/d(k)

If max load → some limit < ∞ as n → ∞, then for any T:
Σ load_T² ≤ max_load · Σ load_T = max_load · |T|
So weighted CS ratio = |T|/Σload² ≥ 1/max_load.
Need max_load ≤ 1 for this simple bound.
""", flush=True)

for n in [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)

    load = defaultdict(float)
    for k in deg_cache:
        w = 1.0 / deg_cache[k]
        for h in target_cache[k]:
            load[h] += w

    if not load:
        continue

    loads = sorted(load.values(), reverse=True)
    max_load = loads[0]
    max_h = max(load, key=load.get)

    # Find which elements contribute to max_h
    contribs = [(k, 1.0/deg_cache[k]) for k in deg_cache if max_h in target_cache[k]]
    contribs.sort(key=lambda x: -x[1])

    print(f"  n={n:>6d}: δ={delta:.2f}, max_load={max_load:.4f} at h={max_h}, "
          f"τ(h)={len(contribs)}, p95={loads[len(loads)//20]:.4f}", flush=True)
    if len(contribs) <= 10:
        detail = ", ".join(f"k={k}(d={deg_cache[k]},w={w:.3f})" for k, w in contribs[:5])
        print(f"    contributors: {detail}", flush=True)


# ============================================================
# PART 5: DEFINITIVE TEST — weighted CS for all T via per-element + aggregate
# ============================================================
print("\n\n" + "=" * 100, flush=True)
print("PART 5: AGGREGATE WEIGHTED CS — Σ_{k∈T} score_w(k) ≥ 0?", flush=True)
print("=" * 100, flush=True)

print("""
For weighted CS: Σ_h load_T²(h) ≤ |T| iff
Σ_{k∈T} [1 - 1/d(k) - (ρ̃(k) - (d(k)-1)·(counting within T vs V))] ≥ 0

Actually, the condition Σ load_T² ≤ |T| depends on T, not just individual elements.
But: Σ load_T²(h) = Σ_k 1/d(k) + 2·Σ_{k<k'∈T} codeg(k,k')/(d(k)d(k'))
                  ≤ Σ_k 1/d(k) + Σ_k ρ̃(k)/d(k)  [upper bound using ρ̃ with global V]
                  = Σ_k (1 + ρ̃(k))/d(k)

So: |T| / Σload² ≥ |T| / Σ_k (1+ρ̃(k))/d(k)
    = |T| · d̃_harm / (1 + ρ̃_avg/d̃)

For Hall: need Σ_k (1+ρ̃(k))/d(k) ≤ |T|
i.e., Σ_k ((1+ρ̃(k))/d(k) - 1) ≤ 0
i.e., Σ_k (1+ρ̃(k) - d(k))/d(k) ≤ 0

score_w(k) = (d(k) - 1 - ρ̃(k))/d(k)
Need: Σ_{k∈T} score_w(k) ≥ 0 for all T.
This holds iff score_w(k) ≥ 0 for all k (per-element) OR worst T has positive sum.
""", flush=True)

for n in [5000, 10000, 20000, 30000, 50000, 75000, 100000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)

    # Global load
    load_V = defaultdict(float)
    for k in deg_cache:
        w = 1.0 / deg_cache[k]
        for h in target_cache[k]:
            load_V[h] += w

    # ρ̃(k) using global load (OVERESTIMATE for T ⊂ V)
    rho_tilde = {k: sum(load_V[h] for h in target_cache[k]) - 1.0 for k in deg_cache}
    score_w = {k: (deg_cache[k] - 1 - rho_tilde[k]) / deg_cache[k] for k in deg_cache}

    neg_elems = sorted([(k, score_w[k]) for k in score_w if score_w[k] < 0], key=lambda x: x[1])
    pos_elems = [(k, score_w[k]) for k in score_w if score_w[k] >= 0]

    total_neg = sum(s for _, s in neg_elems)
    total_pos = sum(s for _, s in pos_elems)
    total_all = total_neg + total_pos

    # Worst T = all negative-score elements (using GLOBAL ρ̃ as overestimate)
    # If total_neg + correction from restricted T gives ≥ 0, we're done

    print(f"\n  n={n:>6d}: |V|={len(deg_cache)}", flush=True)
    print(f"    Σ score_w(all V) = {total_all:.4f}", flush=True)
    print(f"    Σ score_w(neg only) = {total_neg:.4f} ({len(neg_elems)} elements)", flush=True)
    print(f"    Σ score_w(pos only) = {total_pos:.4f} ({len(pos_elems)} elements)", flush=True)
    print(f"    NOTE: neg uses GLOBAL ρ̃ (overestimate). Real Σload² for T⊂V is smaller.", flush=True)

    # More precise: compute Σ load_T² for T = negative-score elements
    if neg_elems:
        T_neg = [k for k, s in neg_elems]
        load_T = defaultdict(float)
        for k in T_neg:
            w = 1.0 / deg_cache[k]
            for h in target_cache[k]:
                load_T[h] += w
        sum_load_sq = sum(l * l for l in load_T.values())
        actual_ratio = len(T_neg) / sum_load_sq
        print(f"    ACTUAL ratio for T_neg: {actual_ratio:.5f} "
              f"{'✓' if actual_ratio >= 1 else '✗'}", flush=True)


print("\n\nDONE.", flush=True)
