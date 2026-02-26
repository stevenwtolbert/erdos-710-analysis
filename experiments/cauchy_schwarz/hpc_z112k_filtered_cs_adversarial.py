#!/usr/bin/env python3
"""
Z112k: FILTERED CS — adversarial test

Key insight from Z112i Part 2: excluding targets with τ_T(h) ≥ 3 makes CS work.

Filtered CS bound:
  |NH(T)| ≥ |{h: τ_T(h) ≤ τ_max}| ≥ (Σ d_filt(k))² / Σ_{h: τ≤τ_max} τ²

This is a VALID lower bound on |NH(T)| since NH(T) ⊇ {h: τ≤τ_max}.

PART 1: Adversarial greedy for filtered CS (τ_max=2)
PART 2: Two-round approach (match V₃, then filtered CS on V_rest)
PART 3: Is there a CLEAN per-element condition for filtered CS?
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
    targets = []
    j_lo = (2 * n) // k + 1
    j_hi = (n + L_val) // k
    for j in range(j_lo, j_hi + 1):
        m = k * j
        if 2 * n < m <= n + L_val:
            targets.append(m)
    return targets


# ============================================================
# PART 1: Adversarial greedy for filtered CS
# ============================================================
print("=" * 100, flush=True)
print("PART 1: ADVERSARIAL GREEDY for filtered CS (τ_max=2)", flush=True)
print("=" * 100, flush=True)

for n in [10000, 30000, 50000, 100000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    # For each target, list of elements that hit it
    target_to_elems = defaultdict(list)

    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        targets = get_targets(n, k, M)
        target_cache[k] = targets
        for h in targets:
            target_to_elems[h].append(k)

    # Greedy: build T to minimize filtered CS ratio
    # Track: τ_T(h) for each target h
    # W₂ = {h : τ_T(h) ≤ 2}
    # d₂(k) = |NH(k) ∩ W₂| for k ∈ T
    # CS_filt = (Σ d₂)² / (Σ d₂ + 2|NH₂|)  [where NH₂ = {h∈W₂: τ=2}]

    tau_T = defaultdict(int)  # τ_T(h) for current T
    T = set()
    T_size = 0

    # For efficiency, precompute which targets each element has
    # Track running sums
    sum_d2 = 0  # Σ d₂(k) for k ∈ T
    nh1 = 0     # |{h: τ=1}|
    nh2 = 0     # |{h: τ=2}|

    # d₂(k) for elements in T
    d2_of = {}  # d₂[k] for k ∈ T

    candidates = set(deg_cache.keys())
    best_ratio_seen = float('inf')
    best_T_size = 0

    max_steps = min(3000, len(candidates))

    for step in range(max_steps):
        # Try each candidate: compute change in filtered CS ratio
        best_k = None
        best_new_ratio = float('inf')

        # Sample for speed
        if len(candidates) > 3000:
            import random
            random.seed(42 + step)
            cand_list = random.sample(list(candidates), 3000)
        else:
            cand_list = list(candidates)

        for k in cand_list:
            # Adding k: for each target h of k:
            # - tau_T(h) increases by 1
            # - If was 0→1: h enters W₂ as τ=1. nh1 += 1, d₂(k) += 1
            # - If was 1→2: h transitions from τ=1 to τ=2. nh1 -= 1, nh2 += 1
            #   d₂(k) += 1 (h is still in W₂)
            #   No change to other elements' d₂
            # - If was 2→3: h LEAVES W₂. nh2 -= 1.
            #   d₂ of the OTHER 2 elements targeting h each decrease by 1
            #   d₂(k) does NOT increase (h not in W₂ after adding k)
            #   sum_d2 decreases by 2 (the two other elements)
            # - If was ≥3→more: no change to W₂ or d₂

            delta_sum_d2 = 0
            delta_nh1 = 0
            delta_nh2 = 0
            new_d2_k = 0

            for h in target_cache[k]:
                t = tau_T[h]
                if t == 0:
                    delta_nh1 += 1
                    new_d2_k += 1
                    delta_sum_d2 += 1  # k gains this target
                elif t == 1:
                    delta_nh1 -= 1
                    delta_nh2 += 1
                    new_d2_k += 1
                    # No change to other d₂ (h was τ=1, still in W₂ at τ=2)
                    delta_sum_d2 += 1  # k gains, no one loses
                elif t == 2:
                    delta_nh2 -= 1
                    # h leaves W₂. The 2 existing elements in T that hit h lose it
                    delta_sum_d2 -= 2  # two elements lose this target
                    # k does NOT gain it (h now has τ=3, excluded)
                # else: t ≥ 3, no change

            new_sum_d2 = sum_d2 + delta_sum_d2
            new_nh1 = nh1 + delta_nh1
            new_nh2 = nh2 + delta_nh2
            new_T_size = T_size + 1

            # CS_filt = (new_sum_d2)² / (new_sum_d2 + 2*new_nh2) if new_sum_d2 > 0
            denom = new_sum_d2 + 2 * new_nh2
            if denom > 0 and new_sum_d2 > 0:
                cs_filt = new_sum_d2 * new_sum_d2 / denom
                new_ratio = cs_filt / new_T_size
            else:
                new_ratio = float('inf')

            if new_ratio < best_new_ratio:
                best_new_ratio = new_ratio
                best_k = k

        if best_k is None:
            break

        # Add best_k to T
        k = best_k
        T.add(k)
        candidates.discard(k)

        # Update tau_T, nh1, nh2, sum_d2, d₂
        new_d2_k = 0
        for h in target_cache[k]:
            t = tau_T[h]
            if t == 0:
                nh1 += 1
                new_d2_k += 1
                sum_d2 += 1
            elif t == 1:
                nh1 -= 1
                nh2 += 1
                new_d2_k += 1
                sum_d2 += 1
            elif t == 2:
                nh2 -= 1
                # Remove d₂ from the two existing elements
                for k2 in target_to_elems[h]:
                    if k2 in T and k2 != k:
                        d2_of[k2] -= 1
                        sum_d2 -= 1
            tau_T[h] += 1

        d2_of[k] = new_d2_k
        T_size += 1

        # Compute current ratio
        denom = sum_d2 + 2 * nh2
        if denom > 0 and sum_d2 > 0:
            cs_filt = sum_d2 * sum_d2 / denom
            ratio = cs_filt / T_size
        else:
            ratio = 0

        if ratio < best_ratio_seen:
            best_ratio_seen = ratio
            best_T_size = T_size

        if step < 20 or step % 500 == 0 or step == max_steps - 1:
            print(f"  n={n}, step {step:>4d}: |T|={T_size:>5d}, "
                  f"Σd₂={sum_d2:>7d}, nh1={nh1:>6d}, nh2={nh2:>6d}, "
                  f"ratio={ratio:.5f}", flush=True)

    print(f"  n={n}: WORST filtered CS ratio = {best_ratio_seen:.5f} at |T|={best_T_size} "
          f"{'≥ 1 ✓' if best_ratio_seen >= 1 else '< 1 ✗'}", flush=True)

    # Also test T = V (all elements)
    tau_all = defaultdict(int)
    for k in deg_cache:
        for h in target_cache[k]:
            tau_all[h] += 1
    all_nh1 = sum(1 for t in tau_all.values() if t == 1)
    all_nh2 = sum(1 for t in tau_all.values() if t == 2)
    all_d2 = sum(min(t, 2) for t in tau_all.values())
    all_denom = all_d2 + 2 * all_nh2
    all_cs = all_d2**2 / all_denom if all_denom > 0 else 0
    all_ratio = all_cs / len(deg_cache) if len(deg_cache) > 0 else 0
    print(f"  n={n}: T=V: Σd₂={all_d2}, nh1={all_nh1}, nh2={all_nh2}, "
          f"ratio={all_ratio:.5f} {'✓' if all_ratio >= 1 else '✗'}", flush=True)

    # Test T = T₀ (unweighted negatives)
    rho = {}
    tau_V = defaultdict(int)
    for k in deg_cache:
        for h in target_cache[k]:
            tau_V[h] += 1
    for k in deg_cache:
        rho[k] = sum(tau_V[h] - 1 for h in target_cache[k])
    scores = {k: deg_cache[k]*(deg_cache[k]-1) - rho[k] for k in deg_cache}
    T0 = [k for k in scores if scores[k] < 0]

    tau_t0 = defaultdict(int)
    for k in T0:
        for h in target_cache[k]:
            tau_t0[h] += 1
    t0_nh1 = sum(1 for t in tau_t0.values() if t == 1)
    t0_nh2 = sum(1 for t in tau_t0.values() if t == 2)
    t0_d2 = sum(min(t, 2) for t in tau_t0.values())
    t0_denom = t0_d2 + 2 * t0_nh2
    t0_cs = t0_d2**2 / t0_denom if t0_denom > 0 else 0
    t0_ratio = t0_cs / len(T0) if len(T0) > 0 else float('inf')
    print(f"  n={n}: T=T₀: |T₀|={len(T0)}, Σd₂={t0_d2}, ratio={t0_ratio:.5f} "
          f"{'✓' if t0_ratio >= 1 else '✗'}", flush=True)
    print(flush=True)


# ============================================================
# PART 2: Minimum d₂(k) analysis — how small can filtered degree get?
# ============================================================
print("\n" + "=" * 100, flush=True)
print("PART 2: MINIMUM FILTERED DEGREE — min d₂(k) for T = V and T = T₀", flush=True)
print("=" * 100, flush=True)

for n in [10000, 30000, 50000, 100000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    tau_V = defaultdict(int)
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)
        for h in target_cache[k]:
            tau_V[h] += 1

    # d₂(k) for T = V: targets h of k with τ_V(h) ≤ 2
    d2_V = {}
    for k in deg_cache:
        d2_V[k] = sum(1 for h in target_cache[k] if tau_V[h] <= 2)

    min_d2 = min(d2_V.values())
    min_d2_k = min(d2_V, key=d2_V.get)
    avg_d2 = sum(d2_V.values()) / len(d2_V)

    # Elements with d₂ < 2
    low_d2 = [(k, d2_V[k], deg_cache[k]) for k in d2_V if d2_V[k] < 3]
    low_d2.sort(key=lambda x: x[1])

    print(f"\n  n={n}: T=V", flush=True)
    print(f"    min d₂ = {min_d2} at k={min_d2_k} (d={deg_cache[min_d2_k]})", flush=True)
    print(f"    avg d₂ = {avg_d2:.2f}", flush=True)
    print(f"    elements with d₂ < 3: {len(low_d2)}", flush=True)
    for k, d2, d in low_d2[:10]:
        print(f"      k={k}: d={d}, d₂={d2}", flush=True)

    # Distribution of d₂ vs d
    by_deg = defaultdict(list)
    for k in deg_cache:
        by_deg[deg_cache[k]].append(d2_V[k])
    print(f"    d₂ by degree:", flush=True)
    for d in sorted(by_deg.keys())[:8]:
        vals = by_deg[d]
        print(f"      d={d:>3d}: n={len(vals):>5d}, min_d₂={min(vals)}, "
              f"avg_d₂={sum(vals)/len(vals):.2f}, max_d₂={max(vals)}", flush=True)


# ============================================================
# PART 3: What if we use τ_max=3 instead of 2?
# ============================================================
print("\n\n" + "=" * 100, flush=True)
print("PART 3: Filtered CS with τ_max = 1 (completely disjoint targets)", flush=True)
print("=" * 100, flush=True)

for n in [10000, 30000, 50000, 100000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    tau_V = defaultdict(int)
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)
        for h in target_cache[k]:
            tau_V[h] += 1

    # For T = V: only count τ=1 targets
    d1_V = {}
    for k in deg_cache:
        d1_V[k] = sum(1 for h in target_cache[k] if tau_V[h] == 1)

    nh1 = sum(1 for t in tau_V.values() if t == 1)
    sum_d1 = sum(d1_V.values())
    # CS_filt at τ_max=1: ratio = (Σd₁)² / (|V| · Σ_{h:τ=1} 1) = (Σd₁)²/(|V|·nh1)
    # But Σd₁ = nh1 (each τ=1 target contributes 1 to exactly one element's d₁)
    # So CS = nh1² / (|V| · nh1) = nh1/|V|
    ratio_1 = nh1 / len(deg_cache) if len(deg_cache) > 0 else 0

    min_d1 = min(d1_V.values())
    avg_d1 = sum(d1_V.values()) / len(d1_V)
    zero_d1 = sum(1 for v in d1_V.values() if v == 0)

    print(f"\n  n={n}: nh1={nh1}, |V|={len(deg_cache)}, ratio=nh1/|V|={ratio_1:.4f} "
          f"{'✓' if ratio_1 >= 1 else '✗'}", flush=True)
    print(f"    min d₁={min_d1}, avg d₁={avg_d1:.2f}, zero d₁: {zero_d1}", flush=True)


# ============================================================
# PART 4: Can we PROVE Σd₂ ≥ 2|T| for any T?
# ============================================================
print("\n\n" + "=" * 100, flush=True)
print("PART 4: ANALYSIS — is Σd₂ ≥ 2|T| always true?", flush=True)
print("=" * 100, flush=True)

print("""
If Σd₂ ≥ 2|T| for any T ⊆ V, then since NH₂ ≤ Σd₂/2:
  CS_filt = (Σd₂)² / (Σd₂ + 2|NH₂|) ≥ (Σd₂)² / (2·Σd₂) = Σd₂/2 ≥ |T|

So Σd₂ ≥ 2|T| is SUFFICIENT for filtered CS to prove Hall.
Equivalent to: avg d₂(k) ≥ 2.
""", flush=True)

for n in [10000, 30000, 50000, 100000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)

    # For T = V: compute d₂
    tau_V = defaultdict(int)
    for k in deg_cache:
        for h in target_cache[k]:
            tau_V[h] += 1

    d2_V = {k: sum(1 for h in target_cache[k] if tau_V[h] <= 2) for k in deg_cache}
    sum_d2_V = sum(d2_V.values())
    ratio_d2 = sum_d2_V / (2 * len(deg_cache))

    # For adversarial T: find T that minimizes avg d₂
    # Adding k to T increases τ for k's targets, potentially reducing d₂ of others
    # Greedy: add element that MOST reduces Σd₂ - 2 (net contribution)
    # net contribution of adding k:
    #   k gains d₂(k) targets (those with current τ_T ≤ 1, since after adding τ becomes ≤ 2)
    #   Some targets go from τ=2 to τ=3, removed from W₂
    #   Each such target removes 1 from d₂ of the 2 existing elements targeting it
    #   Net = d₂_gain(k) - 2*(number of τ=2→3 transitions) - 2 (for the term 2|T|)

    tau_T = defaultdict(int)
    T = set()
    sum_d2_T = 0
    T_size = 0
    d2_of = {}

    candidates = list(deg_cache.keys())
    worst_ratio = float('inf')
    worst_T_size = 0

    for step in range(min(3000, len(candidates))):
        # Find candidate with smallest net contribution
        best_k = None
        best_net = float('inf')

        if len(candidates) > 3000:
            import random
            random.seed(42 + step)
            sample = random.sample(candidates, 3000)
        else:
            sample = candidates

        for k in sample:
            net = 0
            d2_gain = 0
            for h in target_cache[k]:
                t = tau_T[h]
                if t <= 1:
                    d2_gain += 1  # k gets this as d₂ target
                    net += 1
                elif t == 2:
                    net -= 2  # two elements lose this target from d₂
            net -= 2  # subtract 2 for the denominator increase
            if net < best_net:
                best_net = net
                best_k = k

        if best_k is None:
            break

        k = best_k
        T.add(k)
        candidates.remove(k)

        d2_k = 0
        for h in target_cache[k]:
            t = tau_T[h]
            if t <= 1:
                d2_k += 1
                sum_d2_T += 1
            elif t == 2:
                # Remove from d₂ of existing elements
                sum_d2_T -= 2
            tau_T[h] += 1

        d2_of[k] = d2_k
        T_size += 1

        avg_d2_T = sum_d2_T / T_size if T_size > 0 else 0
        ratio = avg_d2_T / 2

        if ratio < worst_ratio:
            worst_ratio = ratio
            worst_T_size = T_size

        if step < 15 or step % 500 == 0:
            print(f"  n={n}, step {step:>4d}: |T|={T_size:>5d}, "
                  f"Σd₂={sum_d2_T:>7d}, avg_d₂={avg_d2_T:.4f}, "
                  f"ratio(avg_d₂/2)={ratio:.5f}", flush=True)

    print(f"  n={n}: WORST avg_d₂/2 = {worst_ratio:.5f} at |T|={worst_T_size} "
          f"{'≥ 1 ✓' if worst_ratio >= 1 else '< 1 ✗'}", flush=True)
    print(flush=True)


print("\n\nDONE.", flush=True)
