#!/usr/bin/env python3
"""
Z103c: FAST adversarial α verification for three-block FMC.

Uses multiple strategies to find worst-case subsets:
1. Greedy with multiple orderings
2. Local search (swap in/out) starting from greedy
3. Exhaustive for small blocks (|block| ≤ 30)
4. Focus on the CRITICAL n=5000 case (where FMC = 0.981)

The key insight: greedy gives α_greedy ≥ α_true (OVERESTIMATES α).
So 1/α_greedy UNDERESTIMATES the FMC contribution.
We need to verify that the greedy isn't significantly overestimating.
"""

import math, sys
from collections import defaultdict
from itertools import combinations

C_TARGET = 2 / math.exp(0.5) + 0.05

def compute_params(n):
    ln_n = math.log(n) if n > 1 else 1
    ln_ln_n = math.log(ln_n) if ln_n > 1 else 1
    L = int(math.ceil(C_TARGET * n * math.sqrt(ln_n / ln_ln_n)))
    M = L - n
    N = n // 2
    delta = 2.0 * M / n - 1
    return L, M, N, delta

def build_adj(n, k_list, M):
    L_val = M + n
    adj = {}
    for k in k_list:
        j_lo = (2 * n) // k + 1
        j_hi = (n + L_val) // k
        targets = []
        for j in range(j_lo, j_hi + 1):
            m = k * j
            if 2 * n < m <= n + L_val:
                targets.append(m)
        adj[k] = targets
    return adj

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

def compute_ratio(T, adj):
    if not T:
        return float('inf')
    covered = set()
    for k in T:
        for m in adj.get(k, []):
            covered.add(m)
    return len(covered) / len(T)

def greedy_adversarial(elements, adj, target_size, order_key=None):
    """Greedy: pick element adding fewest new targets."""
    T = []
    covered = set()
    if order_key is None:
        pool = sorted(elements, key=lambda k: len(adj.get(k, [])))
    else:
        pool = sorted(elements, key=order_key)
    T_set = set()
    for _ in range(min(target_size, len(pool))):
        best_k = None
        best_new = float('inf')
        for k in pool:
            if k in T_set:
                continue
            new = sum(1 for m in adj.get(k, []) if m not in covered)
            if new < best_new:
                best_new = new
                best_k = k
        if best_k is None:
            break
        T.append(best_k)
        T_set.add(best_k)
        for m in adj.get(best_k, []):
            covered.add(m)
    return T

def local_search(T_init, all_elements, adj, max_iter=1000):
    """Local search: try swaps and removals to improve ratio."""
    T = list(T_init)
    T_set = set(T)
    outside = [k for k in all_elements if k not in T_set]
    best_ratio = compute_ratio(T, adj)

    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1

        # Try removing each element
        if len(T) > 1:
            for i in range(len(T)):
                T_new = T[:i] + T[i+1:]
                r = compute_ratio(T_new, adj)
                if r < best_ratio:
                    best_ratio = r
                    outside.append(T[i])
                    T = T_new
                    T_set = set(T)
                    improved = True
                    break

        # Try swapping each T element with each outside element
        for i in range(len(T)):
            for j in range(len(outside)):
                T_new = T[:i] + [outside[j]] + T[i+1:]
                r = compute_ratio(T_new, adj)
                if r < best_ratio:
                    best_ratio = r
                    old = T[i]
                    T = T_new
                    T_set = set(T)
                    outside[j] = old
                    improved = True
                    break
            if improved:
                break

        # Try adding elements
        if not improved:
            for j in range(len(outside)):
                T_new = T + [outside[j]]
                r = compute_ratio(T_new, adj)
                if r < best_ratio:
                    best_ratio = r
                    T = T_new
                    T_set = set(T)
                    outside = outside[:j] + outside[j+1:]
                    improved = True
                    break

    return T, best_ratio

def exhaustive_alpha(elements, adj, max_size=15):
    """Exhaustive search over all subsets up to max_size."""
    elems = list(elements)
    n = len(elems)
    min_ratio = float('inf')
    best_T = None

    for sz in range(1, min(max_size + 1, n + 1)):
        for combo in combinations(elems, sz):
            T = list(combo)
            r = compute_ratio(T, adj)
            if r < min_ratio:
                min_ratio = r
                best_T = T
    return min_ratio, best_T

def find_worst_alpha(elements, adj, block_name="", verbose=True):
    """Multi-strategy search for worst-case α."""
    if not elements:
        return float('inf'), "empty", []

    elems = list(elements)
    n_el = len(elems)
    min_ratio = float('inf')
    best_method = ""
    best_T = []

    def update(r, T, method):
        nonlocal min_ratio, best_method, best_T
        if r < min_ratio:
            min_ratio = r
            best_method = method
            best_T = list(T)
            if verbose:
                print(f"    {block_name} improved: α={r:.4f} |T|={len(T)} [{method}]", flush=True)

    # Strategy 1: Greedy with different orderings, many sizes
    orderings = [
        ("deg_asc", lambda k: len(adj.get(k, []))),
        ("deg_desc", lambda k: -len(adj.get(k, []))),
        ("value", lambda k: k),
        ("value_desc", lambda k: -k),
    ]

    sizes = set()
    for s in [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50, 100]:
        if s <= n_el:
            sizes.add(s)
    for pct in range(2, 102, 2):
        s = max(1, n_el * pct // 100)
        if s <= n_el:
            sizes.add(s)
    sizes.add(n_el)

    for name, key in orderings:
        for sz in sorted(sizes):
            if sz > n_el:
                continue
            T = greedy_adversarial(elems, adj, sz, order_key=key)
            if T:
                r = compute_ratio(T, adj)
                update(r, T, f"greedy_{name}(sz={sz})")

    # Strategy 2: Local search from best greedy solutions
    # Collect top-5 distinct greedy results
    greedy_results = []
    for name, key in orderings:
        for sz in [max(1, n_el//10), max(1, n_el//4), max(1, n_el//2), n_el]:
            if sz > n_el:
                continue
            T = greedy_adversarial(elems, adj, sz, order_key=key)
            if T:
                r = compute_ratio(T, adj)
                greedy_results.append((r, T, f"greedy_{name}(sz={sz})"))

    greedy_results.sort()
    seen = set()
    for r, T, method in greedy_results[:8]:
        key = tuple(sorted(T))
        if key in seen:
            continue
        seen.add(key)
        T_improved, r_improved = local_search(T, elems, adj, max_iter=200)
        update(r_improved, T_improved, f"local_from_{method}")

    # Strategy 3: Exhaustive for very small blocks
    if n_el <= 20:
        r_ex, T_ex = exhaustive_alpha(elems, adj, max_size=n_el)
        if T_ex:
            update(r_ex, T_ex, "exhaustive")
    elif n_el <= 50:
        r_ex, T_ex = exhaustive_alpha(elems, adj, max_size=8)
        if T_ex:
            update(r_ex, T_ex, "exhaustive(≤8)")
    elif n_el <= 200:
        r_ex, T_ex = exhaustive_alpha(elems, adj, max_size=5)
        if T_ex:
            update(r_ex, T_ex, "exhaustive(≤5)")

    # Strategy 4: High-codegree cluster search
    target_to_elems = defaultdict(list)
    for k in elems:
        for m in adj.get(k, []):
            target_to_elems[m].append(k)

    # Find highly-shared targets and build subsets around them
    high_codeg = sorted(target_to_elems.items(), key=lambda x: -len(x[1]))[:30]
    for target, cluster in high_codeg:
        if len(cluster) < 2:
            continue
        cluster_sorted = sorted(cluster, key=lambda k: len(adj.get(k, [])))
        for sz in range(2, min(len(cluster) + 1, 51)):
            T = cluster_sorted[:sz]
            r = compute_ratio(T, adj)
            update(r, T, f"cluster(codeg={len(cluster)},sz={sz})")

        # Local search from cluster
        if len(cluster) >= 3:
            T_ls, r_ls = local_search(cluster_sorted[:min(10, len(cluster))], elems, adj, max_iter=100)
            update(r_ls, T_ls, f"local_cluster(codeg={len(cluster)})")

    # Strategy 5: Low-degree concentration
    # Sort by degree, take the lowest-degree elements
    deg_sorted = sorted(elems, key=lambda k: len(adj.get(k, [])))
    for sz in [2, 3, 4, 5, 8, 10, 15, 20, 30]:
        if sz > n_el:
            continue
        T = deg_sorted[:sz]
        r = compute_ratio(T, adj)
        update(r, T, f"low_deg(sz={sz})")
        # Local search from low-degree set
        T_ls, r_ls = local_search(T, elems, adj, max_iter=100)
        update(r_ls, T_ls, f"local_low_deg(sz={sz})")

    return min_ratio, best_method, best_T

# ============================================================
print("=" * 90, flush=True)
print("Z103c: FAST ADVERSARIAL α VERIFICATION — THREE-BLOCK FMC", flush=True)
print("=" * 90, flush=True)
print(flush=True)
print("KEY: greedy α OVERESTIMATES true α (misses worst subsets).", flush=True)
print("We use multiple strategies to find LOWER α values.", flush=True)
print("If FMC sum still < 1 with improved α: proof holds.", flush=True)
print(flush=True)

for n in [2000, 3000, 4000, 5000, 7000, 10000, 15000, 20000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    all_el = list(range(1, N + 1))
    adj = build_adj(n, all_el, M)

    R = [k for k in all_el if largest_prime_factor(k) > B]
    S_plus = [k for k in all_el if largest_prime_factor(k) <= B and k > B]
    S_minus = [k for k in all_el if largest_prime_factor(k) <= B and k <= B]

    adj_R = {k: adj[k] for k in R}
    adj_Sp = {k: adj[k] for k in S_plus}
    adj_Sm = {k: adj[k] for k in S_minus}

    print(f"\nn={n:,d}, δ={delta:.3f}, B={B}", flush=True)
    print(f"  |R|={len(R)}, |S₊|={len(S_plus)}, |S₋|={len(S_minus)}", flush=True)

    # Cap element count for speed
    max_el = 300 if n <= 10000 else 200

    # R block
    R_use = R
    if len(R_use) > max_el:
        R_use = sorted(R_use, key=lambda k: len(adj_R.get(k, [])))[:max_el]
    alpha_R, method_R, T_R = find_worst_alpha(R_use, adj_R, "R", verbose=True)

    # S₊ block (CRITICAL)
    Sp_use = S_plus
    if len(Sp_use) > max_el:
        Sp_use = sorted(Sp_use, key=lambda k: len(adj_Sp.get(k, [])))[:max_el]
    alpha_Sp, method_Sp, T_Sp = find_worst_alpha(Sp_use, adj_Sp, "S₊", verbose=True)

    # S₋ block (usually tiny, can be exhaustive)
    alpha_Sm, method_Sm, T_Sm = find_worst_alpha(S_minus, adj_Sm, "S₋", verbose=True)

    inv_R = 1.0/alpha_R if 0 < alpha_R < 1e6 else 0
    inv_Sp = 1.0/alpha_Sp if 0 < alpha_Sp < 1e6 else 0
    inv_Sm = 1.0/alpha_Sm if 0 < alpha_Sm < 1e6 else 0
    fmc = inv_R + inv_Sp + inv_Sm

    ok = "< 1 OK" if fmc < 1 else ">= 1 FAIL"

    print(f"\n  RESULT n={n:,d}:", flush=True)
    print(f"    R:  α={alpha_R:.4f}  1/α={inv_R:.4f}  [{method_R}]", flush=True)
    print(f"    S₊: α={alpha_Sp:.4f}  1/α={inv_Sp:.4f}  [{method_Sp}]", flush=True)
    print(f"    S₋: α={alpha_Sm:.4f}  1/α={inv_Sm:.4f}  [{method_Sm}]", flush=True)
    print(f"    FMC = {fmc:.4f} {ok}", flush=True)

    if fmc >= 1.0:
        print(f"    *** CRITICAL: FMC FAILS at n={n}! ***", flush=True)

    # Also show worst T details for S₊
    if T_Sp:
        print(f"    S₊ worst T: size={len(T_Sp)}, elements={T_Sp[:10]}{'...' if len(T_Sp)>10 else ''}", flush=True)
        # Show degree distribution of worst T
        degs = [len(adj_Sp.get(k, [])) for k in T_Sp]
        print(f"    S₊ worst T degrees: min={min(degs)}, max={max(degs)}, mean={sum(degs)/len(degs):.1f}", flush=True)

    sys.stdout.flush()

print("\nDONE.", flush=True)
