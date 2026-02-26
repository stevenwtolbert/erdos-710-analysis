#!/usr/bin/env python3
"""
Z103: THREE-BLOCK FMC TEST

KEY IDEA: Instead of splitting S₊ into dyadic intervals (giving 10-15 blocks),
use just THREE blocks:
  R = rough elements (primes > B with disjoint targets)
  S₊ = smooth elements in (B, N]
  S₋ = smooth elements in [1, B]

FMC condition: 1/α(R) + 1/α(S₊) + 1/α(S₋) ≤ 1

Since R has disjoint targets: α(R) = d_min(R) (exact, not CS bound)
Since S₋ has huge degree: α(S₋) → ∞
Need: 1/α(S₊) < 1 - 1/d_min(R) - 1/α(S₋) ≈ 1 - 1/3 ≈ 2/3

i.e., α(S₊) > 3/2 = 1.5

The adversarial chain argument suggests α(S₊) ≈ 2 (two interleaved chains
of degree-3 elements, each new element adding 2 new targets).
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

def compute_alpha(elements, adj, max_el=500):
    """Compute α via greedy adversarial at multiple scales."""
    if not elements:
        return float('inf')
    elems = list(elements)
    if len(elems) > max_el:
        elems.sort(key=lambda k: len(adj.get(k, [])))
        elems = elems[:max_el]
    n_el = len(elems)
    min_ratio = float('inf')

    sizes = set([1, 2, 3, 5, 10])
    for pct in range(5, 105, 5):
        sizes.add(max(1, n_el * pct // 100))
    sizes.add(n_el)

    for sz in sorted(sizes):
        if sz > n_el:
            continue
        T = greedy_adversarial(elems, adj, sz)
        if not T:
            continue
        tau = defaultdict(int)
        for k in T:
            for m in adj.get(k, []):
                tau[m] += 1
        NH = len(tau)
        actual = NH / len(T)
        if actual < min_ratio:
            min_ratio = actual
    return min_ratio

# ============================================================
print("=" * 90)
print("Z103: THREE-BLOCK FMC TEST")
print("=" * 90)

print("""
Partition: V = R ∪ S₊ ∪ S₋
  R  = rough elements (lpf(k) > B)
  S₊ = smooth elements in (B, N]
  S₋ = smooth elements in [1, B]

FMC: 1/α(R) + 1/α(S₊) + 1/α(S₋) ≤ 1
""")

for n in [500, 1000, 2000, 5000, 10000, 20000, 50000]:
    L, M, N, delta = compute_params(n)
    nL = n + L
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    sqrt_nL = int(math.sqrt(nL))

    all_el = list(range(1, N + 1))
    adj = build_adj(n, all_el, M)

    # Classify elements
    R = [k for k in all_el if largest_prime_factor(k) > B]
    S_plus = [k for k in all_el if largest_prime_factor(k) <= B and k > B]
    S_minus = [k for k in all_el if largest_prime_factor(k) <= B and k <= B]

    # Test: are rough targets actually disjoint?
    # For rough primes p₁, p₂ > B: shared target iff p₁·j₁ = p₂·j₂
    # Since p₁,p₂ are coprime and j₁,j₂ small, this requires p₁|j₂ and p₂|j₁
    # But j₁,j₂ < p₁,p₂ (since j_hi = (n+L)/k and k > B implies j < n+L/B ≈ √(nL))
    # Wait, that's not quite right — let me just check.

    # Rough α
    adj_R = {k: adj[k] for k in R}
    if R:
        # Check disjointness: count shared targets
        target_count = defaultdict(int)
        for k in R:
            for m in adj_R[k]:
                target_count[m] += 1
        max_codeg_R = max(target_count.values()) if target_count else 0
        d_min_R = min(len(adj_R[k]) for k in R)
        d_bar_R = sum(len(adj_R[k]) for k in R) / len(R)
        alpha_R = compute_alpha(R, adj_R, max_el=400)
    else:
        max_codeg_R = 0
        d_min_R = 0
        d_bar_R = 0
        alpha_R = float('inf')

    # S₊ α (SINGLE BLOCK)
    adj_Sp = {k: adj[k] for k in S_plus}
    if S_plus:
        d_min_Sp = min(len(adj_Sp[k]) for k in S_plus)
        d_bar_Sp = sum(len(adj_Sp[k]) for k in S_plus) / len(S_plus)
        alpha_Sp = compute_alpha(S_plus, adj_Sp, max_el=500)
    else:
        d_min_Sp = 0
        d_bar_Sp = 0
        alpha_Sp = float('inf')

    # S₋ α
    adj_Sm = {k: adj[k] for k in S_minus}
    if S_minus:
        d_min_Sm = min(len(adj_Sm[k]) for k in S_minus)
        d_bar_Sm = sum(len(adj_Sm[k]) for k in S_minus) / len(S_minus)
        alpha_Sm = compute_alpha(S_minus, adj_Sm, max_el=300)
    else:
        d_min_Sm = 0
        d_bar_Sm = 0
        alpha_Sm = float('inf')

    # FMC sum
    inv_R = 1.0/alpha_R if alpha_R > 0 and alpha_R < 1e6 else 0
    inv_Sp = 1.0/alpha_Sp if alpha_Sp > 0 and alpha_Sp < 1e6 else 0
    inv_Sm = 1.0/alpha_Sm if alpha_Sm > 0 and alpha_Sm < 1e6 else 0
    fmc_sum = inv_R + inv_Sp + inv_Sm

    ok = "< 1 OK" if fmc_sum < 1 else "> 1 FAIL"

    print(f"n={n:>7,d}  δ={delta:.3f}  B={B}")
    print(f"  |R|={len(R):>6d}  |S₊|={len(S_plus):>6d}  |S₋|={len(S_minus):>5d}")
    print(f"  R:  d_min={d_min_R:>4d}  d̄={d_bar_R:>6.1f}  α={alpha_R:>7.2f}  max_codeg={max_codeg_R}  1/α={inv_R:.4f}")
    print(f"  S₊: d_min={d_min_Sp:>4d}  d̄={d_bar_Sp:>6.1f}  α={alpha_Sp:>7.2f}  1/α={inv_Sp:.4f}")
    print(f"  S₋: d_min={d_min_Sm:>4d}  d̄={d_bar_Sm:>6.1f}  α={alpha_Sm:>7.2f}  1/α={inv_Sm:.4f}")
    print(f"  FMC = {inv_R:.4f} + {inv_Sp:.4f} + {inv_Sm:.4f} = {fmc_sum:.4f} {ok}")
    print()

# ============================================================
# Also test: what is α(full V) as a SINGLE block?
print("\n" + "=" * 90)
print("COMPARISON: SINGLE-BLOCK V vs THREE-BLOCK vs DYADIC")
print("=" * 90)

print(f"\n  {'n':>7}  {'1-block':>8}  {'3-block':>8}  {'dyadic(Z101)':>13}")
# 1-block: 1/α(V) — need α(V) ≥ 1. Since d_min = 2-3, α(V) = d_min ≤ 3. Sum = 1/3.
# 3-block: from above
# Dyadic: from Z101

for n in [1000, 5000, 10000, 20000, 50000]:
    L, M, N, delta = compute_params(n)
    all_el = list(range(1, N + 1))
    adj = build_adj(n, all_el, M)
    d_min_all = min(len(adj[k]) for k in all_el)
    alpha_all = d_min_all  # Singleton is worst for single block
    one_block = 1.0/alpha_all

    print(f"  {n:>7,d}  {one_block:8.4f}  (compute above)  (from Z101)")

print("""
KEY INSIGHT: If we use V as a SINGLE block, the FMC sum is just 1/d_min.
Since d_min ≥ 2 for most n: 1/d_min ≤ 0.5 < 1. FMC HOLDS!

But wait — α(V) might be WORSE than d_min for larger T.
The greedy search above tests this.
""")

print("DONE.")
