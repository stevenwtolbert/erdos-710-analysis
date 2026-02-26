#!/usr/bin/env python3
"""
Z112g: CAUCHY-SCHWARZ (Second Moment) BOUND for Hall's condition

For ANY T ⊆ V, by Cauchy-Schwarz on indicator variables:
  |NH(T)| ≥ (Σ_{k∈T} d(k))² / Σ_h τ_T(h)²

where τ_T(h) = |{k ∈ T : k | h}| is the "load" on target h.

Rewriting: Σ_h τ² = Σ d(k) + 2·Σ_{pairs} codeg(k₁,k₂)

So: |NH(T)| ≥ (Σ d)² / (Σ d + 2·Σ codeg)

For Hall: need (Σ d)² ≥ |T|·(Σ d + 2·Σ codeg)

KEY INSIGHT for degree-3 elements:
  - codeg(k₁,k₂) = 1 when k₁/k₂ ∈ {5/6,6/5,5/7,7/5,6/7,7/6}, else 0
  - Each element has at most 6 codeg-neighbors
  - Σ codeg ≤ 3|T|
  - |NH(T)| ≥ 9|T|²/(3|T|+6|T|) = |T|  (barely works!)

But in practice, max codeg-degree < 6 (not all neighbors have degree 3).
Compute EXACT codeg graph and verify Cauchy-Schwarz bound.
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


print("=" * 100, flush=True)
print("Z112g: CAUCHY-SCHWARZ BOUND FOR HALL'S CONDITION", flush=True)
print("=" * 100, flush=True)


# ============================================================
# PART 1: Full V — Cauchy-Schwarz bound
# ============================================================
print("\nPART 1: Cauchy-Schwarz bound for full V", flush=True)
print("=" * 80, flush=True)

for n in [10000, 20000, 30000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    L_val = M + n

    # Compute degrees and targets
    deg_cache = {}
    target_cache = {}
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)

    # For full V: compute Σ d(k) and Σ τ(h)²
    sum_deg = sum(deg_cache.values())
    V_size = len(deg_cache)

    # Count τ(h) for each h
    tau = defaultdict(int)
    for k, targets in target_cache.items():
        for h in targets:
            tau[h] += 1

    sum_tau_sq = sum(t*t for t in tau.values())
    NH_size = len(tau)  # actual |NH(V)|

    # Cauchy-Schwarz bound
    cs_bound = sum_deg * sum_deg / sum_tau_sq if sum_tau_sq > 0 else 0

    # Also compute total codeg
    total_codeg = (sum_tau_sq - sum_deg) // 2

    print(f"\n  n={n}: δ={delta:.3f}, |V|={V_size}, |NH(V)|={NH_size}, M={M}", flush=True)
    print(f"    Σ d(k) = {sum_deg}", flush=True)
    print(f"    Σ τ(h)² = {sum_tau_sq}", flush=True)
    print(f"    Total codeg = {total_codeg}", flush=True)
    print(f"    CS bound = (Σd)²/Στ² = {cs_bound:.1f}", flush=True)
    print(f"    Actual |NH(V)| = {NH_size}", flush=True)
    print(f"    CS bound ≥ |V|? {cs_bound:.1f} ≥ {V_size}? {'YES ✓' if cs_bound >= V_size else 'NO ✗'}", flush=True)

    # Distribution of τ values
    tau_dist = defaultdict(int)
    for t in tau.values():
        tau_dist[t] += 1
    print(f"    τ distribution: {dict(sorted(tau_dist.items())[:10])}", flush=True)


# ============================================================
# PART 2: Degree-3 elements — codeg graph analysis
# ============================================================
print("\n\nPART 2: Codeg graph for degree-3 elements", flush=True)
print("=" * 80, flush=True)

for n in [10000, 20000, 30000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    L_val = M + n

    # Find degree-3 elements
    V3 = []
    target_cache = {}
    for k in range(B + 1, N + 1):
        d = get_degree(n, k, M)
        if d == 3:
            V3.append(k)
            target_cache[k] = get_targets(n, k, M)

    if not V3:
        continue

    # Build codeg graph on V3
    V3_set = set(V3)
    # For each k in V3, find its codeg-neighbors in V3
    codeg_neighbors = defaultdict(list)
    total_edges = 0
    max_deg_graph = 0

    for k in V3:
        targets = target_cache[k]
        # For each target h, find other k' in V3 that also have h as target
        for h in targets:
            # k' divides h and is in V3
            d = 1
            while d * d <= h:
                if h % d == 0:
                    for kp in [d, h // d]:
                        if kp != k and kp in V3_set:
                            if h in target_cache[kp]:
                                if kp not in [x[0] for x in codeg_neighbors[k]]:
                                    codeg_neighbors[k].append((kp, 1))  # codeg ≥ 1
                d += 1

    # Count actual codeg (more carefully)
    codeg_pairs = {}
    for k in V3:
        targets_k = target_cache[k]
        for kp in V3:
            if kp <= k:
                continue
            shared = targets_k & target_cache[kp]
            if shared:
                codeg_pairs[(k, kp)] = len(shared)

    # Stats
    codeg_deg = defaultdict(int)  # degree in codeg graph
    for (k1, k2), c in codeg_pairs.items():
        codeg_deg[k1] += 1
        codeg_deg[k2] += 1

    max_codeg_d = max(codeg_deg.values()) if codeg_deg else 0
    avg_codeg_d = sum(codeg_deg.values()) / len(V3) if V3 else 0
    num_edges = len(codeg_pairs)

    # Codeg values
    codeg_val_dist = defaultdict(int)
    for c in codeg_pairs.values():
        codeg_val_dist[c] += 1

    total_codeg_sum = sum(codeg_pairs.values())

    # CS bound for degree-3 elements
    sum_d = 3 * len(V3)
    sum_tau_sq = sum_d + 2 * total_codeg_sum
    cs_bound = sum_d * sum_d / sum_tau_sq if sum_tau_sq > 0 else 0

    print(f"\n  n={n}: |V₃|={len(V3)}, edges={num_edges}, max_codeg_degree={max_codeg_d}, avg={avg_codeg_d:.2f}", flush=True)
    print(f"    codeg value dist: {dict(sorted(codeg_val_dist.items()))}", flush=True)
    print(f"    Σ codeg = {total_codeg_sum}", flush=True)
    print(f"    Σ d = {sum_d}, Σ τ² = {sum_tau_sq}", flush=True)
    print(f"    CS bound = {cs_bound:.2f} vs |V₃| = {len(V3)}: {'HALL ✓' if cs_bound >= len(V3) else 'FAIL ✗'}", flush=True)
    print(f"    Expansion ratio ≥ {cs_bound/len(V3):.4f}", flush=True)

    # Show degree distribution in codeg graph
    codeg_d_dist = defaultdict(int)
    for v in V3:
        codeg_d_dist[codeg_deg.get(v, 0)] += 1
    print(f"    Codeg-graph degree dist: {dict(sorted(codeg_d_dist.items())[:8])}", flush=True)


# ============================================================
# PART 3: Per-degree-class CS bound
# ============================================================
print("\n\nPART 3: Per-degree-class CS bound", flush=True)
print("=" * 80, flush=True)

for n in [30000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    L_val = M + n

    # Group elements by degree
    by_degree = defaultdict(list)
    target_cache = {}
    for k in range(B + 1, N + 1):
        d = get_degree(n, k, M)
        by_degree[d].append(k)
        target_cache[k] = get_targets(n, k, M)

    print(f"\n  n={n}: δ={delta:.3f}", flush=True)

    for deg in sorted(by_degree.keys())[:8]:  # first 8 degrees
        elems = by_degree[deg]
        if len(elems) < 2:
            continue

        # Compute total codeg within this degree class
        # (For large classes, sample)
        if len(elems) > 2000:
            # Sample
            import random
            random.seed(42)
            sample = random.sample(elems, min(2000, len(elems)))
        else:
            sample = elems

        sample_set = set(sample)
        total_codeg = 0
        num_pairs = 0
        max_codeg = 0
        codeg_deg_count = defaultdict(int)

        for i, k1 in enumerate(sample):
            t1 = target_cache[k1]
            count = 0
            for k2 in sample[i+1:]:
                shared = len(t1 & target_cache[k2])
                if shared > 0:
                    total_codeg += shared
                    num_pairs += 1
                    max_codeg = max(max_codeg, shared)
                    count += 1
            codeg_deg_count[count] += 1

        # CS bound
        t = len(sample)
        sum_d = deg * t
        sum_tau_sq = sum_d + 2 * total_codeg
        cs_bound = sum_d * sum_d / sum_tau_sq if sum_tau_sq > 0 else float('inf')
        expansion = cs_bound / t if t > 0 else float('inf')

        avg_codeg_per_elem = 2 * total_codeg / t if t > 0 else 0

        print(f"    deg={deg}: |V_d|={len(elems):>5d} (sample {t}), "
              f"codeg_pairs={num_pairs}, max_codeg={max_codeg}, "
              f"avg_codeg/elem={avg_codeg_per_elem:.2f}, "
              f"CS expansion ≥ {expansion:.4f} {'✓' if expansion >= 1 else '✗'}", flush=True)


# ============================================================
# PART 4: WORST-CASE subset — adversarial greedy for CS bound
# ============================================================
print("\n\nPART 4: Worst-case subset via greedy (minimize CS bound)", flush=True)
print("=" * 80, flush=True)

for n in [10000, 30000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    L_val = M + n

    deg_cache = {}
    target_cache = {}
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)

    # Greedy: start with element of min degree, then add elements that
    # minimize the CS expansion ratio (Σd)²/(|T|·Στ²)
    # CS expansion = (Σd)² / (|T| · Στ²) ≥ 1 needed for Hall

    # Start with min-degree elements
    min_d = min(deg_cache.values())
    candidates = sorted(deg_cache.keys(), key=lambda k: (deg_cache[k], k))

    # Build subset greedily
    T = []
    T_set = set()
    tau_T = defaultdict(int)  # τ_T(h) for h
    sum_d = 0
    sum_tau_sq = 0
    best_ratio = float('inf')
    best_T_size = 0

    for step in range(min(3000, len(candidates))):
        # Try each candidate, pick the one giving worst CS bound
        best_k = None
        best_new_ratio = float('inf')

        # Limit candidates to check (for speed)
        check_candidates = candidates[:min(500, len(candidates))]
        if step > 0 and step % 100 == 0:
            # Also include some random candidates
            import random
            random.seed(step)
            extra = random.sample(candidates, min(200, len(candidates)))
            check_candidates = list(set(check_candidates + extra))

        for k in check_candidates:
            if k in T_set:
                continue

            # Compute change in sum_d and sum_tau_sq
            d_k = deg_cache[k]
            new_sum_d = sum_d + d_k

            # Change in Σ τ²: for each target h of k,
            # τ(h) increases by 1, so τ² increases by 2τ+1
            delta_tau_sq = 0
            for h in target_cache[k]:
                old_tau = tau_T[h]
                delta_tau_sq += 2 * old_tau + 1

            new_sum_tau_sq = sum_tau_sq + delta_tau_sq
            new_T_size = len(T) + 1

            new_ratio = new_sum_d * new_sum_d / (new_T_size * new_sum_tau_sq) if new_sum_tau_sq > 0 else float('inf')

            if new_ratio < best_new_ratio:
                best_new_ratio = new_ratio
                best_k = k

        if best_k is None:
            break

        # Add best_k
        T.append(best_k)
        T_set.add(best_k)
        sum_d += deg_cache[best_k]
        for h in target_cache[best_k]:
            old_tau = tau_T[h]
            sum_tau_sq += 2 * old_tau + 1
            tau_T[h] += 1

        candidates = [k for k in candidates if k not in T_set]  # remove added

        new_ratio = sum_d * sum_d / (len(T) * sum_tau_sq) if sum_tau_sq > 0 else float('inf')
        if new_ratio < best_ratio:
            best_ratio = new_ratio
            best_T_size = len(T)

        if step < 20 or step % 100 == 0 or new_ratio < 1.1:
            actual_NH = len([h for h in tau_T if tau_T[h] > 0])
            print(f"  n={n} step {step}: |T|={len(T)}, Σd={sum_d}, Στ²={sum_tau_sq}, "
                  f"CS_ratio={new_ratio:.4f}, |NH|={actual_NH}, "
                  f"actual_ratio={actual_NH/len(T):.4f}", flush=True)

        if new_ratio < 0.95:
            print(f"  *** CS BOUND FAILS: ratio {new_ratio:.4f} < 1 ***", flush=True)
            break

    print(f"  n={n}: FINAL best CS ratio = {best_ratio:.4f} at |T|={best_T_size}", flush=True)
    print(f"  {'HALL HOLDS (CS ≥ 1) ✓' if best_ratio >= 1.0 else 'CS BOUND INCONCLUSIVE ✗'}", flush=True)


print("\n\nDONE.", flush=True)
