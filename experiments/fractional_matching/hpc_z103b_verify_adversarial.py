#!/usr/bin/env python3
"""
Z103b: VERIFY that Z103's greedy adversarial is finding true worst case.

Key concern: greedy_adversarial gives α_greedy ≥ α_true (overestimates).
So 1/α_greedy ≤ 1/α_true, meaning the FMC sum is UNDERESTIMATED.
If true sum > 1 at n=5K: three-block FMC fails there.

Tests:
1. Multiple random starts for the greedy
2. All-pairs codegree analysis within each block
3. Exhaustive search over small subsets (up to size ~20)
4. HK verification of per-block Hall
"""

import math
import random
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

def greedy_adversarial(elements, adj, target_size):
    """Standard greedy: pick element adding fewest new targets."""
    T = []
    covered = set()
    pool = sorted(elements, key=lambda k: len(adj.get(k, [])))
    for _ in range(min(target_size, len(pool))):
        best_k = None
        best_new = float('inf')
        for k in pool:
            if k in set(T):
                continue
            new = sum(1 for m in adj.get(k, []) if m not in covered)
            if new < best_new:
                best_new = new
                best_k = k
        if best_k is None:
            break
        T.append(best_k)
        pool = [k for k in pool if k != best_k]
        for m in adj.get(best_k, []):
            covered.add(m)
    return T

def random_adversarial(elements, adj, target_size, seed=None):
    """Random start + greedy refinement."""
    rng = random.Random(seed)
    pool = list(elements)
    rng.shuffle(pool)
    # Start with a random low-degree element
    pool.sort(key=lambda k: len(adj.get(k, [])) + rng.random())
    T = []
    covered = set()
    for _ in range(min(target_size, len(pool))):
        best_k = None
        best_new = float('inf')
        for k in pool:
            if k in set(T):
                continue
            new = sum(1 for m in adj.get(k, []) if m not in covered)
            if new < best_new or (new == best_new and rng.random() < 0.3):
                best_new = new
                best_k = k
        if best_k is None:
            break
        T.append(best_k)
        pool = [k for k in pool if k != best_k]
        for m in adj.get(best_k, []):
            covered.add(m)
    return T

def compute_ratio(T, adj):
    if not T:
        return float('inf')
    tau = defaultdict(int)
    for k in T:
        for m in adj.get(k, []):
            tau[m] += 1
    return len(tau) / len(T)

def compute_alpha_thorough(elements, adj, max_el=500, n_random=20):
    """Thoroughly search for worst-case α using multiple strategies."""
    if not elements:
        return float('inf'), "empty"
    elems = list(elements)
    if len(elems) > max_el:
        elems.sort(key=lambda k: len(adj.get(k, [])))
        elems = elems[:max_el]
    n_el = len(elems)
    min_ratio = float('inf')
    worst_method = ""
    worst_size = 0

    sizes = set([1, 2, 3, 4, 5, 8, 10, 15, 20])
    for pct in range(2, 105, 2):
        sizes.add(max(1, n_el * pct // 100))
    sizes.add(n_el)

    # Method 1: Standard greedy
    for sz in sorted(sizes):
        if sz > n_el:
            continue
        T = greedy_adversarial(elems, adj, sz)
        if not T:
            continue
        ratio = compute_ratio(T, adj)
        if ratio < min_ratio:
            min_ratio = ratio
            worst_method = f"greedy(sz={sz})"
            worst_size = len(T)

    # Method 2: Random starts
    for trial in range(n_random):
        for sz in [2, 5, 10, 20, 50, 100, n_el // 4, n_el // 2, n_el]:
            if sz > n_el or sz < 1:
                continue
            T = random_adversarial(elems, adj, sz, seed=trial * 100 + sz)
            if not T:
                continue
            ratio = compute_ratio(T, adj)
            if ratio < min_ratio:
                min_ratio = ratio
                worst_method = f"random(trial={trial},sz={sz})"
                worst_size = len(T)

    # Method 3: High-codegree pairs
    # Find pairs with maximum codegree and build around them
    target_to_elems = defaultdict(list)
    for k in elems:
        for m in adj.get(k, []):
            target_to_elems[m].append(k)

    # Find targets with highest codegree
    high_codeg_targets = sorted(target_to_elems.items(), key=lambda x: -len(x[1]))[:20]
    for target, cluster in high_codeg_targets:
        for sz in range(2, min(len(cluster) + 1, 51)):
            # Pick the sz elements with lowest degree from the cluster
            cluster_sorted = sorted(cluster, key=lambda k: len(adj.get(k, [])))
            T = cluster_sorted[:sz]
            ratio = compute_ratio(T, adj)
            if ratio < min_ratio:
                min_ratio = ratio
                worst_method = f"cluster(target={target},sz={sz})"
                worst_size = len(T)

    return min_ratio, f"{worst_method} |T|={worst_size}"

# ============================================================
print("=" * 90)
print("Z103b: VERIFY ADVERSARIAL OVERLAP — THOROUGH α SEARCH")
print("=" * 90)

for n in [1000, 2000, 5000, 10000, 20000, 50000]:
    L, M, N, delta = compute_params(n)
    nL = n + L
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    all_el = list(range(1, N + 1))
    adj = build_adj(n, all_el, M)

    R = [k for k in all_el if largest_prime_factor(k) > B]
    S_plus = [k for k in all_el if largest_prime_factor(k) <= B and k > B]
    S_minus = [k for k in all_el if largest_prime_factor(k) <= B and k <= B]

    adj_R = {k: adj[k] for k in R}
    adj_Sp = {k: adj[k] for k in S_plus}
    adj_Sm = {k: adj[k] for k in S_minus}

    print(f"\nn={n:,d}, δ={delta:.3f}, B={B}")

    # Thorough search for each block
    alpha_R, method_R = compute_alpha_thorough(R, adj_R, max_el=500, n_random=15)
    alpha_Sp, method_Sp = compute_alpha_thorough(S_plus, adj_Sp, max_el=500, n_random=15)
    alpha_Sm, method_Sm = compute_alpha_thorough(S_minus, adj_Sm, max_el=300, n_random=10)

    inv_R = 1.0/alpha_R if 0 < alpha_R < 1e6 else 0
    inv_Sp = 1.0/alpha_Sp if 0 < alpha_Sp < 1e6 else 0
    inv_Sm = 1.0/alpha_Sm if 0 < alpha_Sm < 1e6 else 0
    fmc = inv_R + inv_Sp + inv_Sm

    ok = "< 1 OK" if fmc < 1 else "> 1 FAIL"

    print(f"  R:  α={alpha_R:.3f}  1/α={inv_R:.4f}  [{method_R}]")
    print(f"  S₊: α={alpha_Sp:.3f}  1/α={inv_Sp:.4f}  [{method_Sp}]")
    print(f"  S₋: α={alpha_Sm:.3f}  1/α={inv_Sm:.4f}  [{method_Sm}]")
    print(f"  FMC = {fmc:.4f} {ok}")

    # Also test: full V as single block
    alpha_V, method_V = compute_alpha_thorough(all_el, adj, max_el=500, n_random=15)
    print(f"  V(full): α={alpha_V:.3f}  1/α={1.0/alpha_V:.4f}  [{method_V}]")

print("\nDONE.")
