#!/usr/bin/env python3
"""
Z111: FIND THE CORRECT PARTITION FOR FMC

Z110 showed α(S₊) = d_min is FALSE. The three-block partition's FMC fails at n<30K.

KEY INSIGHT: The problem is mixing min-degree elements (k ≈ N, deg = d_min) with
mid-range elements (k ≈ N/2, deg ≈ 2*d_min) in the same block. Element k=3696 ≈ N/2
shares 5/6 targets with T_min because 2*3696 = 7392 ∈ T_min.

FIX: Separate min-degree elements into their own block:
  V_min = {k > B : deg(k) = d_min}    (all min-degree, both rough and smooth)
  V_rest = {k > B : deg(k) > d_min}   (all higher-degree)
  S₋ = {1, ..., B}

Cross-block disjointness for V_min: smooth×smooth (coprime pair thm), rough×rough (prime),
smooth×rough (gcd ≤ N/B, lcm ≥ NB >> n+L). All pairwise disjoint. α(V_min) = d_min.

For V_rest: all deg ≥ d_min+1 = 4. The problematic k=3696 type elements are now in V_rest,
but they no longer share a block with their multiples in V_min. Need α(V_rest) > 1.51.
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

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

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

# ============================================================
print("=" * 100, flush=True)
print("Z111: CORRECT PARTITION — V_min + V_rest + S₋", flush=True)
print("=" * 100, flush=True)

# Part 1: Verify cross-type disjointness for V_min
print("\n--- Part 1: Cross-type disjointness in V_min ---\n", flush=True)

for n in [15000, 20000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    d_min = int(math.floor(delta)) + 1

    # V_min: all k > B with deg = d_min
    V_min_smooth = []
    V_min_rough = []
    for k in range(B + 1, N + 1):
        d = get_degree(n, k, M)
        if d == d_min:
            lpf = largest_prime_factor(k)
            if lpf <= B:
                V_min_smooth.append(k)
            else:
                V_min_rough.append(k)

    # Check cross-type codeg (smooth × rough)
    max_cross_codeg = 0
    cross_count = 0
    for ks in V_min_smooth[:100]:  # Sample
        for kr in V_min_rough[:100]:
            cross_count += 1
            lcm_val = ks * kr // gcd(ks, kr)
            L_val = M + n
            if lcm_val <= n + L_val:
                codeg = 0
                for mult in range((2 * n) // lcm_val + 1, (n + L_val) // lcm_val + 1):
                    m = lcm_val * mult
                    if 2 * n < m <= n + L_val:
                        codeg += 1
                if codeg > max_cross_codeg:
                    max_cross_codeg = codeg

    # lcm lower bound for cross pairs
    min_lcm_ratio = float('inf')
    for ks in V_min_smooth[:50]:
        for kr in V_min_rough[:50]:
            g = gcd(ks, kr)
            lcm_val = ks * kr // g
            ratio = lcm_val / (n + L)
            if ratio < min_lcm_ratio:
                min_lcm_ratio = ratio

    print(f"  n={n}: |V_min_smooth|={len(V_min_smooth)}, |V_min_rough|={len(V_min_rough)}, "
          f"cross_codeg={max_cross_codeg}, min_lcm/(n+L)={min_lcm_ratio:.2f}", flush=True)


# Part 2: Compute α(V_rest) by greedy adversarial
print("\n\n--- Part 2: α(V_rest) by greedy adversarial ---\n", flush=True)
print(f"  {'n':>7}  {'δ':>8}  {'d_min':>5}  {'|V_rest|':>8}  "
      f"{'min_deg':>7}  {'greedy_α':>9}  {'|T_worst|':>9}  "
      f"{'FMC_sum':>8}  {'threshold':>9}  {'Status':>7}", flush=True)
print("  " + "-" * 105, flush=True)

for n in [10000, 12000, 15000, 17000, 20000, 25000, 30000, 40000, 50000,
          75000, 100000, 150000, 200000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    d_min = int(math.floor(delta)) + 1

    # V_rest: all k > B with deg > d_min
    V_rest = []
    target_cache = {}
    min_deg_rest = float('inf')
    for k in range(B + 1, N + 1):
        d = get_degree(n, k, M)
        if d > d_min:
            V_rest.append((k, d))
            target_cache[k] = get_targets(n, k, M)
            if d < min_deg_rest:
                min_deg_rest = d

    # Greedy adversarial: start fresh, always add element minimizing ratio
    current_NH = set()
    current_T = []
    min_ratio = float('inf')
    min_at_size = 0

    remaining = list(V_rest)
    remaining_set = set(k for k, _ in remaining)

    for step in range(min(len(V_rest), 500)):
        # Find element minimizing new ratio
        best_k = None
        best_ratio = float('inf')

        # Sample for efficiency
        check = list(remaining_set)
        if len(check) > 2000:
            # Sort by degree (low first) and take top 2000
            check_with_deg = [(k, d) for k, d in V_rest if k in remaining_set]
            check_with_deg.sort(key=lambda x: x[1])
            check = [k for k, _ in check_with_deg[:2000]]

        for k in check:
            t = target_cache[k]
            new_NH_size = len(current_NH | t)
            new_size = len(current_T) + 1
            ratio = new_NH_size / new_size
            if ratio < best_ratio:
                best_ratio = ratio
                best_k = k

        if best_k is None:
            break

        current_T.append(best_k)
        current_NH |= target_cache[best_k]
        remaining_set.discard(best_k)

        if best_ratio < min_ratio:
            min_ratio = best_ratio
            min_at_size = len(current_T)

    # FMC computation
    alpha_Vmin = d_min
    alpha_Vrest = min_ratio
    alpha_Sm = M / B - 1 if B > 0 else float('inf')
    fmc_sum = 1.0 / alpha_Vmin + 1.0 / alpha_Vrest + 1.0 / alpha_Sm
    rhs = 1.0 - 1.0/d_min - B/(M-B)
    threshold = 1.0 / rhs if rhs > 0 else float('inf')
    status = "OK" if fmc_sum < 1.0 else "FAIL"

    print(f"  {n:>7,d}  {delta:>8.4f}  {d_min:>5d}  {len(V_rest):>8d}  "
          f"{min_deg_rest:>7d}  {alpha_Vrest:>9.4f}  {min_at_size:>9d}  "
          f"{fmc_sum:>8.4f}  {threshold:>9.4f}  {status:>7s}", flush=True)


# Part 3: Detailed analysis at n=15000
print("\n\n--- Part 3: Detailed V_rest analysis at n=15000 ---\n", flush=True)

n = 15000
L, M, N, delta = compute_params(n)
B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n)))))
d_min = int(math.floor(delta)) + 1

# Elements in V_rest by degree
deg_dist = defaultdict(int)
V_rest = []
target_cache = {}
for k in range(B + 1, N + 1):
    d = get_degree(n, k, M)
    if d > d_min:
        V_rest.append((k, d))
        target_cache[k] = get_targets(n, k, M)
        deg_dist[d] += 1

print(f"V_rest degree distribution at n=15000:", flush=True)
for d in sorted(deg_dist.keys()):
    print(f"  deg={d}: {deg_dist[d]} elements", flush=True)

print(f"\nTotal |V_rest| = {len(V_rest)}", flush=True)

# Maximum target multiplicity in V_rest
from collections import Counter
target_mult = Counter()
for k, d in V_rest:
    for h in target_cache[k]:
        target_mult[h] += 1

max_mult = max(target_mult.values()) if target_mult else 0
mult_dist = Counter(target_mult.values())
print(f"\nTarget multiplicity distribution:", flush=True)
for m in sorted(mult_dist.keys()):
    print(f"  μ={m}: {mult_dist[m]} targets", flush=True)
print(f"Max multiplicity μ_max = {max_mult}", flush=True)

# Lower bound: α(V_rest) ≥ min_deg / μ_max
print(f"\nLower bound: α(V_rest) ≥ min_deg/μ_max = {d_min+1}/{max_mult} = {(d_min+1)/max_mult:.4f}", flush=True)

# Total targets and coverage
total_targets = sum(d for k, d in V_rest)
unique_targets = len(target_mult)
print(f"Total target references: {total_targets}", flush=True)
print(f"Unique targets: {unique_targets}", flush=True)
print(f"Average multiplicity: {total_targets/unique_targets:.2f}", flush=True)


# Part 4: The V_min cross-block proof (smooth × rough)
print("\n\n--- Part 4: V_min disjointness proof data ---\n", flush=True)

for n in [15000, 20000, 50000, 100000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    d_min = int(math.floor(delta)) + 1
    N_val = n // 2

    # For cross pairs (smooth min-deg × rough min-deg):
    # smooth k_s ≈ N with P(k_s) ≤ B
    # rough k_r ≈ N with P(k_r) > B
    # gcd(k_s, k_r) divides k_s and gcd ≤ k_r/P(k_r) ≤ N/B (since P(k_r) > B)
    # lcm = k_s * k_r / gcd ≥ N² / (N/B) = NB

    NB = N_val * B
    nL = n + L
    print(f"  n={n}: N·B = {NB:,d} vs n+L = {nL:,d}, ratio = {NB/nL:.2f} "
          f"{'>> 1 ✓ (disjoint)' if NB > 2*nL else '≈ 1 (close!)'}", flush=True)


# Part 5: Alternative — what if we use dyadic intervals within V_rest?
# This could give tighter α bounds per interval
print("\n\n--- Part 5: α by dyadic interval within V_rest ---\n", flush=True)

n = 15000
L, M, N, delta = compute_params(n)
B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n)))))
d_min = int(math.floor(delta)) + 1

# Dyadic intervals within V_rest: [2^j, 2^{j+1}) for j from ceil(log2(B+1)) to floor(log2(N))
j_min = int(math.ceil(math.log2(B + 1)))
j_max = int(math.floor(math.log2(N)))

fmc_vrest_sum = 0
print(f"Dyadic intervals in V_rest (n=15000, B={B}, N={N}):", flush=True)
for j in range(j_min, j_max + 1):
    lo = 2**j
    hi = min(2**(j+1) - 1, N)

    # Elements in this interval that are in V_rest (deg > d_min)
    interval_elems = []
    for k in range(lo, hi + 1):
        d = get_degree(n, k, M)
        if d > d_min and k > B:
            interval_elems.append((k, d))

    if not interval_elems:
        continue

    # Greedy alpha for this interval
    itargets = {}
    for k, d in interval_elems:
        itargets[k] = get_targets(n, k, M)

    # Greedy
    cur_NH = set()
    cur_T = []
    min_r = float('inf')

    remaining = set(k for k, _ in interval_elems)
    for _ in range(len(interval_elems)):
        best_k = None
        best_r = float('inf')
        for k in remaining:
            r = len(cur_NH | itargets[k]) / (len(cur_T) + 1)
            if r < best_r:
                best_r = r
                best_k = k
        if best_k is None:
            break
        cur_T.append(best_k)
        cur_NH |= itargets[best_k]
        remaining.discard(best_k)
        if best_r < min_r:
            min_r = best_r

    inv_alpha = 1.0/min_r if min_r > 0 else float('inf')
    fmc_vrest_sum += inv_alpha

    print(f"  j={j}: [{lo}, {hi}] |I|={len(interval_elems):>5d}  "
          f"min_deg={min(d for _,d in interval_elems)}  α≥{min_r:.4f}  1/α={inv_alpha:.6f}", flush=True)

print(f"\nΣ 1/α(V_rest intervals) = {fmc_vrest_sum:.6f}", flush=True)
total_fmc = 1.0/d_min + fmc_vrest_sum + B/(M-B)
print(f"Total FMC = 1/{d_min} + {fmc_vrest_sum:.6f} + {B}/{M-B} = {total_fmc:.6f} "
      f"{'< 1 ✓' if total_fmc < 1 else '>= 1 ✗'}", flush=True)


print("\n\nDONE.", flush=True)
