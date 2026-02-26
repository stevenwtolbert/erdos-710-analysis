#!/usr/bin/env python3
"""
Z44: Multiplicative Energy and Adversarial Structure
=====================================================
Adapting Koukoulopoulos-Maynard multiplicative energy techniques to Erdős 710.

Parts A-F: multiplicative energy, GCD partition, K-M compression,
additive energy, quality functions, GCD graph compression.
"""

import numpy as np
from math import isqrt, log, sqrt, gcd
from collections import defaultdict, Counter
import time
import random

# ─────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────

def sieve_smooth(limit, B):
    """Return boolean array where is_smooth[k] = True iff k is B-smooth."""
    is_smooth = np.ones(limit + 1, dtype=bool)
    is_smooth[0] = False
    if limit < 2:
        return is_smooth
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for p in range(2, isqrt(limit) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = False
    for p in range(B + 1, limit + 1):
        if is_prime[p]:
            is_smooth[p::p] = False
    return is_smooth


def compute_graph(n):
    C_TARGET = 2 / np.sqrt(np.e)
    EPS = 0.05
    lnN = log(n)
    lnlnN = log(lnN) if lnN > 1 else 1
    L = (C_TARGET + EPS) * n * sqrt(lnN / lnlnN)
    M = n + L
    B = isqrt(int(M))

    limit = int(M)
    is_smooth = sieve_smooth(limit, B)

    S_plus = [k for k in range(B + 1, n // 2 + 1) if is_smooth[k]]

    adj = {}
    H_set = set()
    lo, hi = n + 1, limit
    for k in S_plus:
        targets = []
        m_lo = max(2, (lo + k - 1) // k)
        m_hi = hi // k
        for m in range(m_lo, m_hi + 1):
            h = m * k
            if is_smooth[h]:
                targets.append(h)
                H_set.add(h)
        adj[k] = targets

    return S_plus, sorted(H_set), adj, B, L, M


def greedy_adversarial(S_plus, adj):
    """Greedy Hall ratio minimizer: iteratively add vertex that minimizes |NH(T)|/|T|."""
    # Start with vertex of smallest degree
    if not S_plus:
        return [], 0, 0
    best_start = min(S_plus, key=lambda k: len(adj.get(k, [])))
    T = [best_start]
    NH_T = set(adj.get(best_start, []))

    for _ in range(min(len(S_plus) - 1, 200)):  # cap iterations
        best_k = None
        best_ratio = float('inf')
        candidates = [k for k in S_plus if k not in set(T)]
        # Sample if too many candidates
        if len(candidates) > 500:
            candidates = random.sample(candidates, 500)
        for k in candidates:
            new_targets = set(adj.get(k, []))
            new_NH = len(NH_T | new_targets)
            ratio = new_NH / (len(T) + 1)
            if ratio < best_ratio:
                best_ratio = ratio
                best_k = k
                best_new_NH = NH_T | new_targets
        if best_k is None:
            break
        # Only add if it improves or maintains ratio
        if best_ratio > len(NH_T) / len(T) + 0.5 and len(T) > 3:
            break
        T.append(best_k)
        NH_T = best_new_NH

    return T, NH_T, len(NH_T) / len(T) if T else float('inf')


def random_subset(S_plus, size):
    """Random subset of S_plus of given size."""
    if size >= len(S_plus):
        return list(S_plus)
    return random.sample(S_plus, size)


# ─────────────────────────────────────────────────────────────────
# Part A: Multiplicative Energy
# ─────────────────────────────────────────────────────────────────

def multiplicative_energy(T):
    """Compute E_mult(T) = Σ_p r(p)² where r(p) = |{(a,b): a*b=p, a,b ∈ T}|."""
    T_set = set(T)
    product_count = Counter()
    for i, a in enumerate(T):
        for b in T:  # ordered pairs
            product_count[a * b] += 1
    E_mult = sum(r * r for r in product_count.values())
    # Note: this counts ordered quadruples (a,b,c,d) with a*b = c*d
    return E_mult, product_count


def multiplicative_energy_fast(T):
    """Faster: count unordered pairs, then E_mult = Σ r(p)^2 for ordered pairs."""
    product_count = Counter()
    n = len(T)
    for i in range(n):
        for j in range(n):
            product_count[T[i] * T[j]] += 1
    E_mult = sum(r * r for r in product_count.values())
    return E_mult


# ─────────────────────────────────────────────────────────────────
# Part B: GCD Partition Structure
# ─────────────────────────────────────────────────────────────────

def gcd_partition_analysis(T, B):
    """Analyze GCD structure of adversarial T."""
    n = len(T)
    gcd_dist = Counter()
    reduced_a = Counter()
    reduced_b = Counter()
    large_gcd_edges = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(T[i], T[j])
            gcd_dist[g] += 1
            a, b = T[i] // g, T[j] // g
            reduced_a[a] += 1
            reduced_b[b] += 1
            total_pairs += 1
            if g > B:
                large_gcd_edges += 1

    return gcd_dist, reduced_a, reduced_b, large_gcd_edges, total_pairs


# ─────────────────────────────────────────────────────────────────
# Part C: K-M Compression (Type grouping)
# ─────────────────────────────────────────────────────────────────

SMALL_PRIMES = [2, 3, 5, 7, 11]

def get_type(k, primes=SMALL_PRIMES):
    """Get the 'type' of k: tuple of exponents of small primes."""
    exps = []
    for p in primes:
        e = 0
        while k % p == 0:
            k //= p
            e += 1
        exps.append(e)
    return tuple(exps)


def km_compression_analysis(T, adj):
    """Group T by type and analyze."""
    type_groups = defaultdict(list)
    for k in T:
        tp = get_type(k)
        type_groups[tp].append(k)

    # Sort by group size
    sorted_types = sorted(type_groups.items(), key=lambda x: -len(x[1]))

    results = []
    for tp, members in sorted_types[:20]:  # top 20 types
        degs = [len(adj.get(k, [])) for k in members]
        mean_deg = np.mean(degs) if degs else 0
        results.append({
            'type': tp,
            'size': len(members),
            'mean_deg': mean_deg,
            'members_sample': members[:5]
        })

    total_types = len(type_groups)
    top_3_frac = sum(len(v) for _, v in sorted_types[:3]) / len(T) if T else 0

    return results, total_types, top_3_frac


# ─────────────────────────────────────────────────────────────────
# Part D: Additive Energy
# ─────────────────────────────────────────────────────────────────

def additive_energy(T):
    """Compute E_add(T) = |{(a,b,c,d) ∈ T⁴ : a+b = c+d}|."""
    sum_count = Counter()
    n = len(T)
    for i in range(n):
        for j in range(n):
            sum_count[T[i] + T[j]] += 1
    E_add = sum(r * r for r in sum_count.values())
    return E_add


# ─────────────────────────────────────────────────────────────────
# Part E: Quality Functions
# ─────────────────────────────────────────────────────────────────

def quality_functions(T, adj, E_mult_val):
    """Compute Q1-Q4 quality measures for T."""
    if not T:
        return {}

    T_set = set(T)
    NH_T = set()
    for k in T:
        NH_T.update(adj.get(k, []))

    # Q1: Hall ratio
    Q1 = len(NH_T) / len(T) if T else 0

    # Q2: Weighted CS with f=1/k
    E1_weighted = sum(1.0 / k for k in T)
    # For each h in NH(T), compute sum of 1/k for k|h, k in T
    h_weight = defaultdict(float)
    for k in T:
        for h in adj.get(k, []):
            h_weight[h] += 1.0 / k
    E2_weighted = sum(w * w for w in h_weight.values())
    Q2 = (E1_weighted ** 2) / E2_weighted if E2_weighted > 0 else float('inf')

    # Q3: log-quality
    Q3 = sum(log(len(adj.get(k, [])) / len(T)) for k in T if len(adj.get(k, [])) > 0 and len(T) > 0) / len(T) if T else 0

    # Q4: energy-penalized expansion
    E1 = sum(len(adj.get(k, [])) for k in T)
    Q4 = (E1 ** 2) / (len(T) * sqrt(E_mult_val)) if E_mult_val > 0 and T else 0

    return {'Q1': Q1, 'Q2': Q2, 'Q3': Q3, 'Q4': Q4, 'E1': E1, 'NH_size': len(NH_T)}


# ─────────────────────────────────────────────────────────────────
# Part F: GCD Graph Compression
# ─────────────────────────────────────────────────────────────────

def gcd_graph_components(T, adj):
    """Build GCD graph (edge if codeg > 0) and find connected components."""
    n = len(T)
    T_set = set(T)
    idx = {k: i for i, k in enumerate(T)}

    # Build target-to-sources map
    target_sources = defaultdict(set)
    for k in T:
        for h in adj.get(k, []):
            target_sources[h].add(k)

    # Build GCD graph: edge if two elements share a target
    adj_gcd = defaultdict(set)
    for h, sources in target_sources.items():
        src_list = list(sources)
        for i in range(len(src_list)):
            for j in range(i + 1, len(src_list)):
                adj_gcd[src_list[i]].add(src_list[j])
                adj_gcd[src_list[j]].add(src_list[i])

    # BFS to find connected components
    visited = set()
    components = []
    for k in T:
        if k not in visited:
            comp = []
            queue = [k]
            visited.add(k)
            while queue:
                v = queue.pop(0)
                comp.append(v)
                for u in adj_gcd.get(v, set()):
                    if u not in visited:
                        visited.add(u)
                        queue.append(u)
            components.append(comp)

    # Analyze each component
    comp_analysis = []
    for comp in sorted(components, key=len, reverse=True)[:10]:
        # Check density
        m = len(comp)
        edges = 0
        for i in range(m):
            for j in range(i + 1, m):
                if comp[j] in adj_gcd.get(comp[i], set()):
                    edges += 1
        max_edges = m * (m - 1) // 2
        density = edges / max_edges if max_edges > 0 else 0

        # Check Hall within component
        NH_comp = set()
        for k in comp:
            NH_comp.update(adj.get(k, []))
        hall_ratio = len(NH_comp) / len(comp) if comp else 0

        # Dyadic interval distribution
        dyadic = Counter()
        for k in comp:
            j = int(log(k) / log(2)) if k > 0 else 0
            dyadic[j] += 1

        comp_analysis.append({
            'size': m,
            'edges': edges,
            'density': density,
            'hall_ratio': hall_ratio,
            'dyadic_dist': dict(dyadic)
        })

    return components, comp_analysis, adj_gcd


# ─────────────────────────────────────────────────────────────────
# Main Experiment
# ─────────────────────────────────────────────────────────────────

def run_experiment(n_val):
    print(f"\n{'='*70}")
    print(f"n = {n_val}")
    print(f"{'='*70}")
    t0 = time.time()

    S_plus, H_smooth, adj, B, L, M = compute_graph(n_val)
    print(f"  B={B}, L={L:.1f}, |S+|={len(S_plus)}, |H_smooth|={len(H_smooth)}")

    if len(S_plus) == 0:
        print("  No S+ elements, skipping.")
        return None

    # Greedy adversarial T
    random.seed(42)
    T, NH_T, hall_ratio = greedy_adversarial(S_plus, adj)
    print(f"  Adversarial T: |T|={len(T)}, |NH(T)|={len(NH_T)}, ratio={hall_ratio:.4f}")

    results = {
        'n': n_val, 'B': B, 'L': L, 'S_plus_size': len(S_plus),
        'H_smooth_size': len(H_smooth), 'T_size': len(T),
        'NH_T_size': len(NH_T), 'hall_ratio': hall_ratio
    }

    # Cap T size for O(|T|^2) computations
    T_comp = T[:150] if len(T) > 150 else T

    # ── Part A: Multiplicative Energy ──
    print(f"\n  Part A: Multiplicative Energy (|T_comp|={len(T_comp)})")
    E_mult_T = multiplicative_energy_fast(T_comp)
    t_size = len(T_comp)
    trivial_bound = t_size ** 2  # diagonal contribution
    max_structured = t_size ** 3
    print(f"    E_mult(T) = {E_mult_T}")
    print(f"    |T|^2 = {trivial_bound}, |T|^3 = {max_structured}")
    print(f"    E_mult/|T|^2 = {E_mult_T/trivial_bound:.4f}")
    print(f"    E_mult/|T|^3 = {E_mult_T/max_structured:.6f}")

    # Random subset for comparison
    R = random_subset(S_plus, len(T_comp))
    E_mult_R = multiplicative_energy_fast(R)
    print(f"    E_mult(random) = {E_mult_R}")
    print(f"    E_mult(random)/|R|^2 = {E_mult_R/len(R)**2:.4f}")
    print(f"    Ratio T/random = {E_mult_T/E_mult_R:.4f}" if E_mult_R > 0 else "")

    results['E_mult_T'] = E_mult_T
    results['E_mult_R'] = E_mult_R
    results['E_mult_ratio_sq'] = E_mult_T / trivial_bound
    results['E_mult_ratio_cube'] = E_mult_T / max_structured
    results['E_mult_T_over_R'] = E_mult_T / E_mult_R if E_mult_R > 0 else None

    # ── Part B: GCD Partition ──
    print(f"\n  Part B: GCD Partition Structure")
    gcd_dist, red_a, red_b, large_gcd_edges, total_pairs = gcd_partition_analysis(T_comp, B)
    top_gcds = gcd_dist.most_common(10)
    print(f"    Total pairs: {total_pairs}")
    print(f"    Large GCD edges (gcd > B={B}): {large_gcd_edges} ({100*large_gcd_edges/max(1,total_pairs):.2f}%)")
    print(f"    Distinct GCDs: {len(gcd_dist)}")
    print(f"    Top 10 GCDs: {top_gcds}")

    # Reduced quotient analysis
    all_reduced = Counter()
    for i in range(len(T_comp)):
        for j in range(i + 1, len(T_comp)):
            g = gcd(T_comp[i], T_comp[j])
            all_reduced[T_comp[i] // g] += 1
            all_reduced[T_comp[j] // g] += 1
    top_reduced = all_reduced.most_common(10)
    print(f"    Top reduced quotients: {top_reduced}")

    # How many reduced quotients are ≤ 10?
    small_reduced = sum(v for k, v in all_reduced.items() if k <= 10)
    total_reduced = sum(all_reduced.values())
    print(f"    Reduced quotients ≤ 10: {small_reduced}/{total_reduced} ({100*small_reduced/max(1,total_reduced):.1f}%)")

    results['large_gcd_edges'] = large_gcd_edges
    results['large_gcd_frac'] = large_gcd_edges / max(1, total_pairs)
    results['distinct_gcds'] = len(gcd_dist)
    results['top_gcds'] = top_gcds
    results['small_reduced_frac'] = small_reduced / max(1, total_reduced)

    # ── Part C: K-M Compression ──
    print(f"\n  Part C: K-M Compression (Type Grouping)")
    type_results, total_types, top3_frac = km_compression_analysis(T_comp, adj)
    print(f"    Total distinct types: {total_types}")
    print(f"    Top 3 types cover: {100*top3_frac:.1f}% of T")
    for tr in type_results[:8]:
        print(f"      type={tr['type']}, size={tr['size']}, mean_deg={tr['mean_deg']:.1f}")

    results['total_types'] = total_types
    results['top3_type_frac'] = top3_frac
    results['type_details'] = type_results[:8]

    # Also do S+ type analysis for comparison
    S_type_groups = defaultdict(list)
    for k in S_plus:
        tp = get_type(k)
        S_type_groups[tp].append(k)
    S_total_types = len(S_type_groups)
    S_sorted = sorted(S_type_groups.items(), key=lambda x: -len(x[1]))
    S_top3_frac = sum(len(v) for _, v in S_sorted[:3]) / len(S_plus) if S_plus else 0
    print(f"    S+ total types: {S_total_types}, top 3 cover: {100*S_top3_frac:.1f}%")

    results['S_total_types'] = S_total_types
    results['S_top3_frac'] = S_top3_frac

    # ── Part D: Additive Energy ──
    print(f"\n  Part D: Additive Energy")
    E_add_T = additive_energy(T_comp)
    E_add_R = additive_energy(R[:len(T_comp)])
    print(f"    E_add(T) = {E_add_T}")
    print(f"    E_add(T)/|T|^2 = {E_add_T/max(1,len(T_comp)**2):.4f}")
    print(f"    E_add(T)/|T|^3 = {E_add_T/max(1,len(T_comp)**3):.6f}")
    print(f"    E_add(random) = {E_add_R}")
    print(f"    Ratio T/random = {E_add_T/max(1,E_add_R):.4f}")

    results['E_add_T'] = E_add_T
    results['E_add_R'] = E_add_R
    results['E_add_ratio_sq'] = E_add_T / max(1, len(T_comp) ** 2)
    results['E_add_ratio_cube'] = E_add_T / max(1, len(T_comp) ** 3)

    # ── Part E: Quality Functions ──
    print(f"\n  Part E: Quality Functions")
    Q = quality_functions(T_comp, adj, E_mult_T)
    for name, val in Q.items():
        print(f"    {name} = {val:.6f}" if isinstance(val, float) else f"    {name} = {val}")

    results['qualities'] = Q

    # ── Part F: GCD Graph Compression ──
    print(f"\n  Part F: GCD Graph Components")
    # Use smaller T for this expensive computation
    T_f = T_comp[:80] if len(T_comp) > 80 else T_comp
    components, comp_analysis, adj_gcd = gcd_graph_components(T_f, adj)
    print(f"    Vertices: {len(T_f)}, Components: {len(components)}")
    for i, ca in enumerate(comp_analysis[:5]):
        print(f"      Comp {i}: size={ca['size']}, edges={ca['edges']}, "
              f"density={ca['density']:.3f}, hall_ratio={ca['hall_ratio']:.3f}, "
              f"dyadic={ca['dyadic_dist']}")

    # Is there one giant component?
    if components:
        giant = max(len(c) for c in components)
        print(f"    Giant component: {giant}/{len(T_f)} ({100*giant/len(T_f):.1f}%)")
        isolated = sum(1 for c in components if len(c) == 1)
        print(f"    Isolated vertices: {isolated}/{len(T_f)} ({100*isolated/len(T_f):.1f}%)")

    results['n_components'] = len(components)
    results['comp_analysis'] = comp_analysis[:5]
    results['giant_frac'] = max(len(c) for c in components) / len(T_f) if components else 0
    results['isolated_frac'] = sum(1 for c in components if len(c) == 1) / len(T_f) if components else 0

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")
    results['time'] = elapsed
    return results


# ─────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    N_VALUES = [1000, 2000, 5000, 10000, 20000, 50000]

    all_results = []
    for n_val in N_VALUES:
        res = run_experiment(n_val)
        if res is not None:
            all_results.append(res)

    # ─────────────────────────────────────────────────────────────
    # Summary Table
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    print(f"\n{'n':>7} | {'|S+|':>6} | {'|T|':>5} | {'ratio':>6} | "
          f"{'Emult/t2':>9} | {'Emult/t3':>9} | {'Emult T/R':>9} | "
          f"{'Eadd/t2':>9} | {'Eadd/t3':>9}")
    print("-" * 100)
    for r in all_results:
        print(f"{r['n']:>7} | {r['S_plus_size']:>6} | {r['T_size']:>5} | "
              f"{r['hall_ratio']:>6.3f} | "
              f"{r['E_mult_ratio_sq']:>9.3f} | {r['E_mult_ratio_cube']:>9.5f} | "
              f"{r.get('E_mult_T_over_R', 0) or 0:>9.3f} | "
              f"{r['E_add_ratio_sq']:>9.3f} | {r['E_add_ratio_cube']:>9.5f}")

    print(f"\n{'n':>7} | {'types':>6} | {'top3%':>6} | {'S_types':>7} | {'S_top3%':>7} | "
          f"{'Q1':>6} | {'Q2':>6} | {'Q3':>7} | {'Q4':>8}")
    print("-" * 90)
    for r in all_results:
        Q = r['qualities']
        print(f"{r['n']:>7} | {r['total_types']:>6} | {100*r['top3_type_frac']:>5.1f}% | "
              f"{r['S_total_types']:>7} | {100*r['S_top3_frac']:>5.1f}% | "
              f"{Q['Q1']:>6.3f} | {Q['Q2']:>6.3f} | {Q['Q3']:>7.3f} | {Q['Q4']:>8.2f}")

    print(f"\n{'n':>7} | {'lgcd%':>6} | {'#gcds':>6} | {'red≤10%':>7} | "
          f"{'#comp':>6} | {'giant%':>7} | {'isol%':>6}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['n']:>7} | {100*r['large_gcd_frac']:>5.1f}% | {r['distinct_gcds']:>6} | "
              f"{100*r['small_reduced_frac']:>5.1f}% | "
              f"{r['n_components']:>6} | {100*r['giant_frac']:>5.1f}% | "
              f"{100*r['isolated_frac']:>5.1f}%")

    # ─────────────────────────────────────────────────────────────
    # Key Findings
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Multiplicative energy classification
    print("\nMultiplicative Energy Classification:")
    for r in all_results:
        ratio = r['E_mult_ratio_sq']
        t_over_r = r.get('E_mult_T_over_R', 0) or 0
        if r['E_mult_ratio_cube'] > 0.5:
            struct = "HIGHLY STRUCTURED (E_mult ~ |T|^3)"
        elif ratio > 3:
            struct = "MODERATELY STRUCTURED"
        else:
            struct = "NEAR-INDEPENDENT (E_mult ~ |T|^2)"
        print(f"  n={r['n']:>6}: E_mult/|T|^2 = {ratio:.3f}, T/random = {t_over_r:.3f} => {struct}")

    # Type concentration
    print("\nType Concentration (K-M Compression):")
    for r in all_results:
        frac = r['top3_type_frac']
        if frac > 0.5:
            conc = "CONCENTRATED (top 3 types > 50%)"
        elif frac > 0.3:
            conc = "MODERATELY CONCENTRATED"
        else:
            conc = "SPREAD"
        print(f"  n={r['n']:>6}: top 3 types = {100*frac:.1f}%, total types = {r['total_types']} => {conc}")

    # GCD graph structure
    print("\nGCD Graph Structure:")
    for r in all_results:
        gf = r['giant_frac']
        iso = r['isolated_frac']
        if gf > 0.8:
            struct = "GIANT COMPONENT DOMINATES"
        elif iso > 0.5:
            struct = "MOSTLY ISOLATED"
        else:
            struct = "MIXED"
        print(f"  n={r['n']:>6}: giant={100*gf:.1f}%, isolated={100*iso:.1f}%, components={r['n_components']} => {struct}")

    # Quality function comparison
    print("\nQuality Functions (which correlates with Hall ratio?):")
    print(f"  {'n':>7} | {'Q1(Hall)':>8} | {'Q2(wCS)':>8} | {'Q3(logQ)':>8} | {'Q4(E-pen)':>8}")
    for r in all_results:
        Q = r['qualities']
        print(f"  {r['n']:>7} | {Q['Q1']:>8.3f} | {Q['Q2']:>8.3f} | {Q['Q3']:>8.3f} | {Q['Q4']:>8.2f}")

    print("\nDone.")
