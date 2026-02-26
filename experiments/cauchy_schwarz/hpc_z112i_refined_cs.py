#!/usr/bin/env python3
"""
Z112i: REFINED CS BOUNDS — closing the ~1% gap

The standard CS bound fails for T₀ = {k : d(d-1) < ρ(k)} at moderate n:
  CS ratio ≈ 0.991 at n=10K, improving to 0.998 at n=50K.

This script tries several approaches to close the gap:

PART 1: CS gap at many n values — track the trend
PART 2: Exclude high-τ targets — modified CS on filtered target set
PART 3: Remove worst smooth elements — does removing O(1) elements fix CS?
PART 4: Weighted CS — optimize weights per element
PART 5: Two-round argument — match degree-3 first (disjoint), then CS on residual
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
# PART 1: CS gap at many n values
# ============================================================
print("=" * 100, flush=True)
print("PART 1: CS gap for T₀ = {k : d(d-1) < ρ(k)} at many n values", flush=True)
print("=" * 100, flush=True)

results = []

for n in [5000, 7500, 10000, 15000, 20000, 30000, 50000, 75000, 100000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    # Build graph
    deg_cache = {}
    target_cache = {}
    tau = defaultdict(int)
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)
        for h in target_cache[k]:
            tau[h] += 1

    # Compute ρ(k) = Σ_{h∈NH(k)} (τ(h) - 1)
    rho = {}
    for k in deg_cache:
        rho[k] = sum(tau[h] - 1 for h in target_cache[k])

    # Per-element scores
    scores = {k: deg_cache[k] * (deg_cache[k] - 1) - rho[k] for k in deg_cache}

    # T₀ = all negative-score elements
    T0 = [k for k in scores if scores[k] < 0]
    T0_pos = [k for k in scores if scores[k] >= 0]

    if not T0:
        print(f"  n={n:>6d}: |V|={len(deg_cache):>6d}, T₀ empty — ALL per-element conditions hold ✓", flush=True)
        results.append((n, len(deg_cache), 0, float('inf'), 0))
        continue

    # CS bound for T₀
    sum_d = sum(deg_cache[k] for k in T0)
    tau_T0 = defaultdict(int)
    for k in T0:
        for h in target_cache[k]:
            tau_T0[h] += 1
    sum_tau_sq = sum(t * t for t in tau_T0.values())
    cs_bound = sum_d * sum_d / sum_tau_sq if sum_tau_sq > 0 else float('inf')
    cs_ratio = cs_bound / len(T0) if T0 else float('inf')

    # Also compute actual |NH(T₀)|
    actual_nh = len(tau_T0)

    # Negative score stats
    neg_scores = sorted(scores[k] for k in T0)
    total_neg = sum(neg_scores)

    print(f"  n={n:>6d}: |V|={len(deg_cache):>6d}, |T₀|={len(T0):>6d}, "
          f"CS={cs_bound:.1f}, ratio={cs_ratio:.5f}, "
          f"|NH|={actual_nh:>6d}, actual_ratio={actual_nh/len(T0):.3f}, "
          f"Σneg_score={total_neg}", flush=True)
    results.append((n, len(deg_cache), len(T0), cs_ratio, actual_nh / len(T0)))

print(f"\nTrend of CS ratio:", flush=True)
for n, vsize, t0size, ratio, act_ratio in results:
    if t0size == 0:
        print(f"  n={n:>6d}: T₀ empty", flush=True)
    else:
        deficit = 1 - ratio if ratio < 1 else 0
        print(f"  n={n:>6d}: CS ratio = {ratio:.6f}, deficit = {deficit:.6f}, "
              f"actual = {act_ratio:.3f}", flush=True)


# ============================================================
# PART 2: Exclude high-τ targets — modified CS
# ============================================================
print("\n\n" + "=" * 100, flush=True)
print("PART 2: EXCLUDE HIGH-τ TARGETS — modified CS on filtered targets", flush=True)
print("=" * 100, flush=True)

for n in [10000, 30000, 50000, 100000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    tau = defaultdict(int)
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)
        for h in target_cache[k]:
            tau[h] += 1

    rho = {k: sum(tau[h] - 1 for h in target_cache[k]) for k in deg_cache}
    scores = {k: deg_cache[k] * (deg_cache[k] - 1) - rho[k] for k in deg_cache}
    T0 = [k for k in scores if scores[k] < 0]

    if not T0:
        print(f"  n={n}: T₀ empty ✓", flush=True)
        continue

    # Compute tau restricted to T₀
    tau_T0 = defaultdict(int)
    for k in T0:
        for h in target_cache[k]:
            tau_T0[h] += 1

    # Try excluding targets with τ ≥ threshold
    print(f"\n  n={n}: |T₀|={len(T0)}", flush=True)
    for tau_max in [2, 3, 4, 5, 6, 7]:
        # Filtered targets: only keep h with τ_T0(h) ≤ tau_max
        # Compute restricted degrees
        d_filt = {}
        sum_d_filt = 0
        sum_tau_sq_filt = 0
        nh_filt = 0
        for h, t in tau_T0.items():
            if t <= tau_max:
                sum_tau_sq_filt += t * t
                nh_filt += 1
        for k in T0:
            d_k = sum(1 for h in target_cache[k] if tau_T0.get(h, 0) <= tau_max)
            d_filt[k] = d_k
            sum_d_filt += d_k

        if sum_tau_sq_filt == 0:
            continue

        cs_filt = sum_d_filt * sum_d_filt / sum_tau_sq_filt
        cs_ratio_filt = cs_filt / len(T0)

        # Also: since we keep nh_filt targets, the bound on |NH_filtered(T₀)| ≥ cs_filt
        # And |NH(T₀)| ≥ |NH_filtered(T₀)| ≥ cs_filt
        print(f"    τ_max={tau_max}: nh_filt={nh_filt}, Σd_filt={sum_d_filt}, "
              f"Στ²_filt={sum_tau_sq_filt}, CS_filt={cs_filt:.1f}, "
              f"ratio={cs_ratio_filt:.5f} {'✓' if cs_ratio_filt >= 1 else '✗'}", flush=True)


# ============================================================
# PART 3: Remove worst smooth elements — does removing O(√n) fix CS?
# ============================================================
print("\n\n" + "=" * 100, flush=True)
print("PART 3: REMOVE WORST ELEMENTS — how many to remove for CS to work?", flush=True)
print("=" * 100, flush=True)

for n in [10000, 30000, 50000, 100000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    tau_global = defaultdict(int)
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)
        for h in target_cache[k]:
            tau_global[h] += 1

    rho = {k: sum(tau_global[h] - 1 for h in target_cache[k]) for k in deg_cache}
    scores = {k: deg_cache[k] * (deg_cache[k] - 1) - rho[k] for k in deg_cache}

    # Sort by score (worst first)
    sorted_elems = sorted(scores.items(), key=lambda x: x[1])

    # Try removing the worst r elements and check CS on the rest
    print(f"\n  n={n}: |V|={len(deg_cache)}", flush=True)

    for r in [0, 10, 50, 100, 200, 500, 1000]:
        if r >= len(deg_cache):
            break
        removed = set(k for k, s in sorted_elems[:r])
        remaining = [k for k in deg_cache if k not in removed]

        # Compute tau restricted to remaining
        tau_rem = defaultdict(int)
        for k in remaining:
            for h in target_cache[k]:
                tau_rem[h] += 1

        # rho_rem for remaining elements
        rho_rem = {k: sum(tau_rem[h] - 1 for h in target_cache[k]) for k in remaining}
        scores_rem = {k: deg_cache[k] * (deg_cache[k] - 1) - rho_rem[k] for k in remaining}

        T0_rem = [k for k in remaining if scores_rem[k] < 0]

        if not T0_rem:
            print(f"    remove {r:>5d}: T₀ empty → CS proves Hall for remaining ✓", flush=True)
            # But we need Hall for ALL of V, not just V \ removed
            # The removed elements need separate handling
            # Check: can removed elements be matched in the residual graph?
            removed_list = [k for k, s in sorted_elems[:r]]
            min_d_rem = min(deg_cache[k] for k in removed_list) if removed_list else 0
            max_d_rem = max(deg_cache[k] for k in removed_list) if removed_list else 0
            print(f"      Removed: deg range [{min_d_rem}, {max_d_rem}]", flush=True)
            break

        # CS for T₀_rem
        sum_d = sum(deg_cache[k] for k in T0_rem)
        tau_T0 = defaultdict(int)
        for k in T0_rem:
            for h in target_cache[k]:
                tau_T0[h] += 1
        sum_tau_sq = sum(t * t for t in tau_T0.values())
        cs_bound = sum_d**2 / sum_tau_sq if sum_tau_sq > 0 else float('inf')
        cs_ratio = cs_bound / len(T0_rem)

        print(f"    remove {r:>5d}: |T₀|={len(T0_rem):>6d}, CS ratio={cs_ratio:.5f} "
              f"{'✓' if cs_ratio >= 1 else '✗'}", flush=True)


# ============================================================
# PART 4: WEIGHTED CS — optimize element weights
# ============================================================
print("\n\n" + "=" * 100, flush=True)
print("PART 4: WEIGHTED CS BOUND — w_k = d(k)^α for various α", flush=True)
print("=" * 100, flush=True)

print("""
Weighted CS: |NH(T)| ≥ (Σ w_k d_k)² / Σ_h (Σ_{k∈T: k|h} w_k)²

With w_k = d^α:
  Numerator = (Σ d^{α+1})²
  Denom = Σ_h (Σ_{k:k|h} d(k)^α)²
""", flush=True)

for n in [10000, 30000, 50000, 100000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    tau_global = defaultdict(int)
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)
        for h in target_cache[k]:
            tau_global[h] += 1

    rho = {k: sum(tau_global[h] - 1 for h in target_cache[k]) for k in deg_cache}
    scores = {k: deg_cache[k] * (deg_cache[k] - 1) - rho[k] for k in deg_cache}
    T0 = [k for k in scores if scores[k] < 0]

    if not T0:
        print(f"  n={n}: T₀ empty ✓", flush=True)
        continue

    print(f"\n  n={n}: |T₀|={len(T0)}", flush=True)

    for alpha in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
        # Compute weighted CS for T₀
        weights = {k: deg_cache[k] ** alpha for k in T0}

        # Numerator: (Σ w_k d_k)²
        numer = sum(weights[k] * deg_cache[k] for k in T0) ** 2

        # Denominator: |T₀| · Σ_h (Σ_{k∈T₀: k|h} w_k)²
        weighted_tau = defaultdict(float)
        for k in T0:
            w = weights[k]
            for h in target_cache[k]:
                weighted_tau[h] += w

        denom = len(T0) * sum(wt * wt for wt in weighted_tau.values())

        ratio = numer / denom if denom > 0 else float('inf')
        print(f"    α={alpha:>5.1f}: ratio = {ratio:.5f} {'✓' if ratio >= 1 else '✗'}", flush=True)


# ============================================================
# PART 5: TWO-ROUND MATCHING — match low-deg first, then CS on residual
# ============================================================
print("\n\n" + "=" * 100, flush=True)
print("PART 5: TWO-ROUND MATCHING — degree-3 first, then CS on residual", flush=True)
print("=" * 100, flush=True)

print("""
Strategy: Since degree-3 elements have (nearly) disjoint targets,
match them first (each uses 1 of 3 targets). Then analyze the
residual graph for remaining elements.

If after removing 1 target per degree-3 element, the remaining
elements still satisfy CS, we have a proof.
""", flush=True)

for n in [10000, 20000, 30000, 50000, 100000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)

    V3 = [k for k in deg_cache if deg_cache[k] == 3]
    V_rest = [k for k in deg_cache if deg_cache[k] > 3]

    # Compute pairwise codegree among V3 elements
    tau_V3 = defaultdict(int)
    for k in V3:
        for h in target_cache[k]:
            tau_V3[h] += 1
    max_tau_V3 = max(tau_V3.values()) if tau_V3 else 0
    codeg_targets_V3 = sum(1 for t in tau_V3.values() if t >= 2)

    # After matching V3: each matched target is "used"
    # Worst case: each V3 element's matched target has max overlap with V_rest
    # Compute: for each V3 target h, how many V_rest elements also target h?
    targets_V3_set = set(tau_V3.keys())

    # For V_rest, compute residual degree (targets NOT shared with V3)
    tau_all = defaultdict(int)
    for k in deg_cache:
        for h in target_cache[k]:
            tau_all[h] += 1

    # Residual after V3 matching: each V3 element uses 1 target, removing it from availability
    # Best case: V3 uses targets NOT shared with V_rest
    # Worst case: V3 uses targets shared with maximum V_rest elements
    # For CS: we don't remove targets, we just need to show V_rest has Hall in residual

    # Actually: after V3 matching, the targets used by V3 are gone.
    # V_rest has degree d_rest(k) = d(k) - (number of k's targets used by V3)

    # BUT: V3 targets that are also V_rest targets get removed, lowering V_rest degrees.
    # Let's compute: how many V_rest targets overlap with V3 targets?

    shared_targets = 0
    v3_target_set = set()
    for k in V3:
        v3_target_set.update(target_cache[k])

    rest_targets = set()
    rest_shared = defaultdict(int)  # per V_rest element: how many targets in V3's target set
    for k in V_rest:
        for h in target_cache[k]:
            rest_targets.add(h)
            if h in v3_target_set:
                rest_shared[k] = rest_shared.get(k, 0) + 1

    overlap = len(v3_target_set & rest_targets)

    # In the worst case, matching V3 removes all shared targets from V_rest
    # But V3 only matches |V3| targets, not all 3|V3|
    # Actually: a matching uses exactly |V3| targets, one per V3 element

    # For V_rest: their degree is unchanged since we only remove |V3| specific targets
    # The question is: can we choose which V3 targets to use to MINIMIZE damage to V_rest?

    # Best strategy: match each V3 element to the target with LOWEST tau_all(h)
    # This minimizes overlap with other elements

    # Compute: for each V3 element, which target has lowest tau_all?
    v3_match_targets = set()
    for k in V3:
        # Pick target with min tau_all (fewest other divisors in V)
        best_h = min(target_cache[k], key=lambda h: tau_all[h])
        v3_match_targets.add(best_h)

    # How many V_rest elements lose a target?
    rest_deg_loss = defaultdict(int)
    for k in V_rest:
        for h in target_cache[k]:
            if h in v3_match_targets:
                rest_deg_loss[k] += 1

    max_loss = max(rest_deg_loss.values()) if rest_deg_loss else 0
    avg_loss = sum(rest_deg_loss.values()) / len(V_rest) if V_rest else 0
    affected = sum(1 for v in rest_deg_loss.values() if v > 0)

    # Residual degrees for V_rest
    res_deg = {k: deg_cache[k] - rest_deg_loss.get(k, 0) for k in V_rest}
    min_res_deg = min(res_deg.values()) if res_deg else 0

    # CS for V_rest in residual graph
    # tau_rest: for each target not in v3_match_targets, count V_rest divisors
    tau_rest = defaultdict(int)
    for k in V_rest:
        for h in target_cache[k]:
            if h not in v3_match_targets:
                tau_rest[h] += 1

    sum_d_rest = sum(res_deg[k] for k in V_rest)
    sum_tau_sq_rest = sum(t * t for t in tau_rest.values())
    cs_rest = sum_d_rest**2 / sum_tau_sq_rest if sum_tau_sq_rest > 0 else float('inf')
    cs_ratio_rest = cs_rest / len(V_rest) if V_rest else float('inf')

    # rho for V_rest in residual
    rho_rest = {k: sum(tau_rest[h] - 1 for h in target_cache[k] if h not in v3_match_targets) for k in V_rest}
    scores_rest = {k: res_deg[k] * (res_deg[k] - 1) - rho_rest[k] for k in V_rest}
    T0_rest = [k for k in V_rest if scores_rest[k] < 0]

    # CS for T₀ of V_rest
    if T0_rest:
        sum_d_t0 = sum(res_deg[k] for k in T0_rest)
        tau_t0 = defaultdict(int)
        for k in T0_rest:
            for h in target_cache[k]:
                if h not in v3_match_targets:
                    tau_t0[h] += 1
        sum_tau_sq_t0 = sum(t * t for t in tau_t0.values())
        cs_t0 = sum_d_t0**2 / sum_tau_sq_t0 if sum_tau_sq_t0 > 0 else float('inf')
        cs_ratio_t0 = cs_t0 / len(T0_rest) if T0_rest else float('inf')
    else:
        cs_ratio_t0 = float('inf')

    print(f"\n  n={n}: |V|={len(deg_cache)}, |V3|={len(V3)}, |V_rest|={len(V_rest)}", flush=True)
    print(f"    V3: max_τ={max_tau_V3}, shared τ≥2 targets: {codeg_targets_V3}", flush=True)
    print(f"    V3 matched targets: {len(v3_match_targets)}, V_rest affected: {affected}/{len(V_rest)}", flush=True)
    print(f"    V_rest max deg loss: {max_loss}, avg loss: {avg_loss:.2f}, min residual deg: {min_res_deg}", flush=True)
    print(f"    CS for ALL V_rest: ratio = {cs_ratio_rest:.5f} {'✓' if cs_ratio_rest >= 1 else '✗'}", flush=True)
    print(f"    CS for T₀(V_rest): |T₀|={len(T0_rest)}, ratio = {cs_ratio_t0:.5f} "
          f"{'✓' if cs_ratio_t0 >= 1 else '✗'}", flush=True)


# ============================================================
# PART 6: INCLUSION-EXCLUSION CORRECTION — second-order term
# ============================================================
print("\n\n" + "=" * 100, flush=True)
print("PART 6: ACTUAL |NH(T₀)| vs bounds", flush=True)
print("=" * 100, flush=True)

for n in [10000, 30000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    tau_global = defaultdict(int)
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)
        for h in target_cache[k]:
            tau_global[h] += 1

    rho = {k: sum(tau_global[h] - 1 for h in target_cache[k]) for k in deg_cache}
    scores = {k: deg_cache[k] * (deg_cache[k] - 1) - rho[k] for k in deg_cache}
    T0 = [k for k in scores if scores[k] < 0]

    if not T0:
        continue

    # Compute τ restricted to T₀
    tau_T0 = defaultdict(int)
    for k in T0:
        for h in target_cache[k]:
            tau_T0[h] += 1

    actual_nh = len(tau_T0)
    sum_d = sum(deg_cache[k] for k in T0)
    sum_tau_sq = sum(t * t for t in tau_T0.values())
    cs = sum_d**2 / sum_tau_sq

    # First-order inclusion-exclusion: Σd - Σ codeg
    # codeg within T₀ = (Σ τ(τ-1)/2) = (Στ² - Σd)/2
    codeg_total = (sum_tau_sq - sum_d) // 2

    ie1 = sum_d - codeg_total  # first-order IE lower bound

    # Triple overlap term (for IE2)
    # Σ_h C(τ(h),3) = Σ_h τ(τ-1)(τ-2)/6
    triple = sum(t * (t-1) * (t-2) // 6 for t in tau_T0.values())
    ie2 = sum_d - codeg_total + triple  # second-order IE lower bound

    # Fourth term
    quad = sum(t * (t-1) * (t-2) * (t-3) // 24 for t in tau_T0.values())
    ie3 = ie2 - quad

    # Bonferroni: odd terms are upper bounds, even terms are lower bounds
    # IE1 = Σd - Σcodeg (UPPER bound on |NH|) - wait no:
    # Actually: |NH| = Σ 1_{τ≥1} and by IE:
    # |NH| ≥ Σd/|T₀| ... no that's wrong

    # Correct IE: |NH(T₀)| = Σ_h 1_{τ(h)≥1}
    # By Bonferroni:
    # ≥ Σ_h [τ ≥ 1] ≥ Σ (τ/1) - Σ C(τ,2) + Σ C(τ,3) - ...
    # = |{h:τ>0}| trivially
    # That's circular.

    # The sieve bound: |NH| = Σ_h 1_{τ≥1}
    # Sieve methods give lower bounds on this.

    # Actually, the standard Bonferroni lower bound is:
    # S₁ - S₂ ≤ |NH| where S₁ = Σ_h τ(h), S₂ = Σ_h C(τ,2)
    # i.e., |NH| ≥ S₁ - S₂ = Σd - codeg_total (THIS IS A LOWER BOUND)
    # And |NH| ≤ S₁ - S₂ + S₃ = Σd - codeg + triple (UPPER BOUND)

    # τ distribution
    tau_dist = defaultdict(int)
    for t in tau_T0.values():
        tau_dist[t] += 1

    print(f"\n  n={n}: |T₀|={len(T0)}", flush=True)
    print(f"    |NH(T₀)| = {actual_nh} (actual)", flush=True)
    print(f"    CS bound = {cs:.1f} (ratio {cs/len(T0):.5f})", flush=True)
    print(f"    Bonf: S₁-S₂ = {ie1} (lower), S₁-S₂+S₃ = {ie2} (upper), S₁-S₂+S₃-S₄ = {ie3} (lower)", flush=True)
    print(f"    S₁=Σd={sum_d}, S₂=codeg={codeg_total}, S₃=triple={triple}, S₄=quad={quad}", flush=True)
    print(f"    τ distribution in T₀: {dict(sorted(tau_dist.items()))}", flush=True)

    # Compare: which bound is tightest?
    print(f"    Hall needs ≥ {len(T0)}:", flush=True)
    print(f"      actual:     {actual_nh:>8d} ({'✓' if actual_nh >= len(T0) else '✗'})", flush=True)
    print(f"      CS:         {cs:>8.1f} ({'✓' if cs >= len(T0) else '✗'})", flush=True)
    print(f"      Bonf(S1-S2):{ie1:>8d} ({'✓' if ie1 >= len(T0) else '✗'})", flush=True)


print("\n\nDONE.", flush=True)
