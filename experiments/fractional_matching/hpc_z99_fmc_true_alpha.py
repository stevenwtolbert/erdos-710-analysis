#!/usr/bin/env python3
"""
Z99: TRUE FMC Sum — Σ 1/α_j for BOTH S₊ and Full V partitions

Key question: Is Σ 1/α_j > 1 for the full V partition?
If yes: FMC with dyadic intervals is dead for moderate n.
If no: we were wrong about the S₊ result extending to full V.

Also tests: at what n does Σ 1/α_j drop below 1?
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
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    delta = 2.0 * M / n - 1
    return L, M, N, B, delta

def build_graph(n, elements, M):
    """Build bipartite graph for a set of left-elements on H = (2n, n+L]."""
    L_val = M + n
    adj = {}
    for k in elements:
        targets = []
        j_lo = (2 * n) // k + 1
        j_hi = (n + L_val) // k
        for j in range(j_lo, j_hi + 1):
            m = k * j
            if 2 * n < m <= n + L_val:
                targets.append(m)
        adj[k] = targets
    return adj

def is_smooth(k, B):
    """Check if k is B-smooth."""
    if k <= 1:
        return True
    n = k
    for p in range(2, min(B + 1, n + 1)):
        while n % p == 0:
            n //= p
        if n == 1:
            return True
    return n == 1

def compute_cs(T, adj):
    """Cauchy-Schwarz expansion bound for subset T."""
    if not T:
        return float('inf'), 0
    t = len(T)
    E1 = sum(len(adj[k]) for k in T)
    tau = defaultdict(int)
    for k in T:
        for m in adj[k]:
            tau[m] += 1
    E2 = sum(v * v for v in tau.values())
    NH = len(tau)
    if E2 == 0 or t == 0 or E1 == 0:
        return float('inf'), 0
    cs_ratio = E1 * E1 / (t * E2)
    actual_ratio = NH / t
    ceff = E2 / E1
    return actual_ratio, cs_ratio

def greedy_adversarial_fast(elements, adj, target_size):
    """Fast greedy |NH|-minimizer."""
    if not elements or target_size <= 0:
        return []
    T = []
    covered = set()
    remaining = list(elements)
    # Sort by degree (ascending) as initial heuristic
    remaining.sort(key=lambda k: len(adj.get(k, [])))

    for _ in range(min(target_size, len(remaining))):
        best_k = None
        best_new = float('inf')
        for k in remaining:
            if k in set(T):
                continue
            new = sum(1 for m in adj.get(k, []) if m not in covered)
            if new < best_new:
                best_new = new
                best_k = k
        if best_k is None:
            break
        T.append(best_k)
        remaining.remove(best_k)
        for m in adj.get(best_k, []):
            covered.add(m)
    return T

def compute_alpha(elements, adj, max_el=500):
    """Compute alpha_j via greedy adversarial at multiple scales.
    For large intervals, subsample to keep computation fast."""
    if not elements:
        return float('inf')
    # For very large intervals, use a subsample of elements for the greedy
    elems = list(elements)
    if len(elems) > max_el:
        # Take the elements most likely to be adversarial:
        # sort by degree (low degree = likely bad expansion)
        elems.sort(key=lambda k: len(adj.get(k, [])))
        elems = elems[:max_el]

    n_el = len(elems)
    min_ratio = float('inf')

    sizes = set([1, 2, 3, 5])
    for pct in range(5, 105, 5):
        sizes.add(max(1, n_el * pct // 100))
    sizes.add(n_el)

    for sz in sorted(sizes):
        if sz > n_el:
            continue
        T = greedy_adversarial_fast(elems, adj, sz)
        if not T:
            continue
        actual, cs = compute_cs(T, adj)
        if actual < min_ratio:
            min_ratio = actual
    return min_ratio

print("=" * 80)
print("Z99: TRUE FMC SUM — Σ 1/α_j FOR FULL V AND S₊ PARTITIONS")
print("=" * 80)

# Test at various n values — focus on where the sum crosses 1
test_ns = [500, 1000, 2000, 5000, 10000, 20000, 50000]

for n in test_ns:
    L, M, N_val, B, delta = compute_params(n)

    # Build full graph for ALL elements
    all_elements = list(range(1, N_val + 1))
    adj_full = build_graph(n, all_elements, M)

    # Identify smooth elements
    smooth = [k for k in all_elements if is_smooth(k, B)]

    # Partition into dyadic intervals
    intervals_full = defaultdict(list)
    intervals_smooth = defaultdict(list)
    for k in all_elements:
        j_val = int(math.log2(k)) if k > 0 else 0
        intervals_full[j_val].append(k)
        if k in set(smooth):
            intervals_smooth[j_val].append(k)

    # Compute Σ 1/α for both partitions
    sum_alpha_full = 0
    sum_cs_full = 0
    sum_alpha_smooth = 0
    sum_cs_smooth = 0

    print(f"\nn={n:,d}, delta={delta:.3f}, N={N_val}, B={B}, |S₊|={len(smooth)}")
    print(f"  {'j':>3} {'|I_full|':>8} {'|I_sm|':>7} {'α_full':>8} {'α_sm':>8} {'CS_full':>8} {'CS_sm':>8}")
    print("  " + "-" * 70)

    for j_val in sorted(intervals_full.keys(), reverse=True):
        I_full = intervals_full[j_val]
        I_smooth = intervals_smooth.get(j_val, [])

        if not I_full:
            continue

        adj_interval = {k: adj_full.get(k, []) for k in I_full}

        # Full interval CS
        _, cs_full_val = compute_cs(I_full, adj_interval)
        if cs_full_val > 0 and cs_full_val < 1e6:
            sum_cs_full += 1.0 / cs_full_val

        # Full interval alpha (adversarial over all elements)
        alpha_full = compute_alpha(I_full, adj_interval, max_el=300)
        if alpha_full > 0 and alpha_full < 1e6:
            sum_alpha_full += 1.0 / alpha_full

        # Smooth-only CS
        cs_sm_val = 0
        alpha_sm = float('inf')
        if I_smooth and len(I_smooth) >= 2:
            adj_sm = {k: adj_full.get(k, []) for k in I_smooth}
            _, cs_sm_val = compute_cs(I_smooth, adj_sm)
            if cs_sm_val > 0 and cs_sm_val < 1e6:
                sum_cs_smooth += 1.0 / cs_sm_val
            alpha_sm = compute_alpha(I_smooth, adj_sm, max_el=300)
            if alpha_sm > 0 and alpha_sm < 1e6:
                sum_alpha_smooth += 1.0 / alpha_sm
        elif I_smooth:
            for k in I_smooth:
                d = len(adj_full.get(k, []))
                if d > 0:
                    sum_cs_smooth += 1.0 / d
                    sum_alpha_smooth += 1.0 / d

        if len(I_full) >= 2:
            print(f"  {j_val:3d} {len(I_full):8d} {len(I_smooth):7d} {alpha_full:8.3f} {alpha_sm:8.3f} {cs_full_val:8.3f} {cs_sm_val:8.3f}")

    ok_af = "< 1 OK" if sum_alpha_full < 1 else "> 1 FAILS"
    ok_as = "< 1 OK" if sum_alpha_smooth < 1 else "> 1 FAILS"
    ok_cf = "< 1 OK" if sum_cs_full < 1 else "> 1 FAILS"
    ok_cs = "< 1 OK" if sum_cs_smooth < 1 else "> 1 FAILS"

    print(f"\n  FULL V:  Σ 1/α = {sum_alpha_full:.4f} {ok_af}   Σ 1/CS = {sum_cs_full:.4f} {ok_cf}")
    print(f"  S₊ ONLY: Σ 1/α = {sum_alpha_smooth:.4f} {ok_as}   Σ 1/CS = {sum_cs_smooth:.4f} {ok_cs}")
    print(f"  Ratio α_full/α_smooth ≈ {sum_alpha_full/sum_alpha_smooth:.3f}" if sum_alpha_smooth > 0 else "")

print("\n\n" + "=" * 80)
print("SUMMARY: WHERE DOES Σ 1/α DROP BELOW 1?")
print("=" * 80)
print("If Σ 1/α(full V) stays above 1 even at n=50000,")
print("then FMC with dyadic intervals CANNOT close the gap at moderate n.")
print("Options: (a) extend HK to cover n up to where FMC works,")
print("         (b) use finer partition (√2 intervals),")
print("         (c) different proof technique.")

print("\nDONE.")
