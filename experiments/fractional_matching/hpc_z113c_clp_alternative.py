#!/usr/bin/env python3
"""
Z113c: Chan-Lichtman-Pomerance alternative + fractional matching analysis.

PART 1: CLP Lemma 4 approach — factor each element, use different bipartite graph.
For each a ∈ (n, 2n], factor a = m·M with m ≤ M.
Match elements to their small factors m. If injective → M's are distinct.

PART 2: Fractional matching via iterative reweighting.
Start with uniform weights x_{k,h} = 1/d(k).
Iteratively reduce weights on overloaded targets (Sinkhorn-like).
Measure how much "fractional matching mass" we lose.

PART 3: Janson's inequality test.
For T ⊆ V: μ = E[|NH(T)|] using random indicator, Δ̄ = pairwise overlap.
If μ² > 2Δ̄, Janson gives |NH(T)| > 0 with high prob.
For Hall: need |NH(T)| ≥ |T|.
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
    j_lo = (2 * n) // k + 1
    j_hi = (n + L_val) // k
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

def get_divisors_in_range(a, lo, hi):
    """Get all divisors of a in (lo, hi]."""
    divs = []
    d = 1
    while d * d <= a:
        if a % d == 0:
            if lo < d <= hi:
                divs.append(d)
            q = a // d
            if q != d and lo < q <= hi:
                divs.append(q)
        d += 1
    return sorted(divs)


# ============================================================
# PART 1: CLP alternative bipartite graph
# ============================================================
print("=" * 90, flush=True)
print("PART 1: CLP factoring bipartite graph", flush=True)
print("=" * 90, flush=True)

print("""
For each a ∈ (n, 2n], find all factorizations a = m·M with m ≤ M.
Build bipartite graph: left = elements a, right = small factors m.
Edge a ~ m iff m | a and m ≤ √(2n).
If this graph has a perfect matching, each a gets a distinct m,
and the quotients M = a/m are also distinct (since a = m·M uniquely).
""", flush=True)

for n in [1000, 5000, 10000, 50000]:
    L, M_param, N, delta = compute_params(n)

    # Left: elements in (n, 2n]
    left = list(range(n + 1, 2 * n + 1))
    sqrt_bound = int(math.isqrt(2 * n))

    # For each a, find divisors m ≤ sqrt(2n)
    # (m=1 is always a divisor, giving M=a)
    degrees = {}
    for a in left:
        divs = get_divisors_in_range(a, 0, sqrt_bound)
        degrees[a] = len(divs)

    d_vals = sorted(degrees.values())
    d_min_clp = d_vals[0]
    d_avg_clp = sum(d_vals) / len(d_vals)

    # Right side: all possible small factors
    right_set = set()
    for a in left:
        for d in get_divisors_in_range(a, 0, sqrt_bound):
            right_set.add(d)

    # Check if |right| ≥ |left| (necessary for matching)
    print(f"  n={n}: |left|={len(left)}, |right|={len(right_set)}, "
          f"d_min={d_min_clp}, d_avg={d_avg_clp:.1f}", flush=True)
    print(f"    |right|/|left| = {len(right_set)/len(left):.3f} "
          f"{'(enough targets)' if len(right_set) >= len(left) else '(NOT ENOUGH)'}", flush=True)

    # Right degree: how many a's share the same small factor m?
    right_deg = defaultdict(int)
    for a in left:
        for d in get_divisors_in_range(a, 0, sqrt_bound):
            right_deg[d] += 1

    r_vals = sorted(right_deg.values(), reverse=True)
    print(f"    Right degree: max={r_vals[0]}, top-5={r_vals[:5]}", flush=True)
    print(f"    (m=1 has right-degree {right_deg.get(1,0)}, "
          f"m=2 has {right_deg.get(2,0)})", flush=True)


# ============================================================
# PART 2: Fractional matching — Sinkhorn iteration
# ============================================================
print("\n\n" + "=" * 90, flush=True)
print("PART 2: Fractional matching via iterative reweighting", flush=True)
print("=" * 90, flush=True)

print("""
Start with x_{k,h} = 1/d(k). Load L(h) = Σ_k x_{k,h}.
For overloaded h (L>1), scale down: x_{k,h} *= 1/L(h).
This reduces left sums below 1. Iterate.
After convergence: total fractional matching = Σ_k Σ_h x_{k,h}.
Loss = |V| - fractional matching = deficiency bound.
""", flush=True)

for n in [5000, 10000, 20000, 50000]:
    L, M_param, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M_param)
        target_cache[k] = get_targets(n, k, M_param)

    V_size = len(deg_cache)

    # Initialize weights: x_{k,h} = 1/d(k)
    # Represent as: for each k, weight[k] = remaining fractional mass
    weight = {k: 1.0 for k in deg_cache}

    # Iterate: compute load, scale down overloaded
    for iteration in range(20):
        # Compute load
        load = defaultdict(float)
        for k in deg_cache:
            w = weight[k] / deg_cache[k]
            for h in target_cache[k]:
                load[h] += w

        # Scale down overloaded targets
        # For each k: new weight = weight * (1/d) * Σ_h min(1, 1/L(h)) * d
        # = weight * Σ_h min(1, 1/L(h)) / d
        new_weight = {}
        for k in deg_cache:
            cap_sum = sum(min(1.0, 1.0 / load[h]) for h in target_cache[k])
            new_weight[k] = weight[k] * cap_sum / deg_cache[k]

        total_mass = sum(new_weight.values())
        max_load = max(load.values())
        weight = new_weight

        if iteration < 5 or iteration == 19:
            loss = V_size - total_mass
            print(f"  n={n}, iter {iteration}: total_mass={total_mass:.2f}, "
                  f"loss={loss:.2f} ({loss/V_size:.2%}), max_load={max_load:.3f}", flush=True)

    print(f"  → Fractional deficiency ≤ {V_size - total_mass:.1f} = "
          f"{(V_size - total_mass)/V_size:.3%} of |V|", flush=True)


# ============================================================
# PART 3: Janson's inequality parameters
# ============================================================
print("\n\n" + "=" * 90, flush=True)
print("PART 3: Janson's inequality parameters", flush=True)
print("=" * 90, flush=True)

print("""
For each target h, define indicator I_h = 1 if some k∈T divides h.
μ = E[number of covered targets] = Σ_h (1 - Π_{k∈T: k|h} (1 - p_k))
For deterministic setting with T ⊆ V:
  μ = |NH(T)| (exact)
  Δ̄ = Σ_{h~h'} P(I_h ∧ I_{h'}) where h~h' if they share a common cause k

Actually, for Janson's inequality in the SIEVE context:
  We want to count h ∈ (2n, n+L] NOT divisible by any k ∈ T.
  |{h : ∀k∈T, k∤h}| = |W| - |NH(T)|
  By inclusion-exclusion / Janson:
  E[unsieved] = |W| · Π_{k∈T} (1 - d(k)/|W|)  [independence approximation]
  Δ̄ = Σ_{k₁<k₂∈T} M/lcm(k₁,k₂) · (stuff)

More directly: Hall needs |NH(T)| ≥ |T|.
Bonferroni gives: |NH(T)| ≥ Σd - Σ_{k<k'} codeg(k,k')
                = Σd - G_T
For Hall: Σd - G_T ≥ |T|, i.e., d̄ - G_T/|T| ≥ 1.

This is the second-order Bonferroni = CS without the squared terms.
Let's compute this.
""", flush=True)

for n in [5000, 10000, 20000, 50000, 100000]:
    L, M_param, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    tau = defaultdict(int)

    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M_param)
        target_cache[k] = get_targets(n, k, M_param)
        for h in target_cache[k]:
            tau[h] += 1

    V_size = len(deg_cache)
    d_min = min(deg_cache.values())

    # Bonferroni-2 for T = V:
    sum_d = sum(deg_cache[k] for k in deg_cache)
    G_V = sum(t * (t-1) // 2 for t in tau.values())  # = Σ_{k<k'} codeg
    # Actually: Σ τ(τ-1)/2 = number of pairs sharing each target = Σ codeg(k,k')
    # And G_V = Σ_{k<k'} codeg(k,k') = Σ_h C(τ(h), 2) = Σ τ(τ-1)/2

    bonf2_V = sum_d - G_V
    bonf2_ratio = bonf2_V / V_size if V_size > 0 else 0

    # For adversarial T (T0 = elements where d(d-1) < ρ):
    rho = {k: sum(tau[h]-1 for h in target_cache[k]) for k in deg_cache}
    T0 = [k for k in deg_cache if deg_cache[k]*(deg_cache[k]-1) < rho[k]]

    if T0:
        tau_T0 = defaultdict(int)
        for k in T0:
            for h in target_cache[k]:
                tau_T0[h] += 1
        sum_d_T0 = sum(deg_cache[k] for k in T0)
        G_T0 = sum(t*(t-1)//2 for t in tau_T0.values())
        bonf2_T0 = sum_d_T0 - G_T0
        bonf2_ratio_T0 = bonf2_T0 / len(T0) if T0 else 0
    else:
        bonf2_ratio_T0 = float('inf')
        G_T0 = 0

    print(f"  n={n:>7d}: |V|={V_size}, d_min={d_min}", flush=True)
    print(f"    T=V:  Σd={sum_d}, G_V={G_V}, Bonf2={bonf2_V}, "
          f"ratio={bonf2_ratio:.4f} {'≥1 ✓' if bonf2_ratio >= 1 else '< 1 ✗'}", flush=True)
    if T0:
        print(f"    T=T0: |T0|={len(T0)}, Σd={sum_d_T0}, G_T0={G_T0}, "
              f"Bonf2={bonf2_T0}, ratio={bonf2_ratio_T0:.4f} "
              f"{'≥1 ✓' if bonf2_ratio_T0 >= 1 else '< 1 ✗'}", flush=True)

    # Third-order Bonferroni correction:
    # |NH(T)| ≥ Σd - G_T + H_T where H_T = Σ_{k<k'<k''} triple_codeg
    # H_T = Σ_h C(τ(h), 3) = Σ τ(τ-1)(τ-2)/6
    if T0:
        H_T0 = sum(t*(t-1)*(t-2)//6 for t in tau_T0.values())
        bonf3_T0 = sum_d_T0 - G_T0 + H_T0
        bonf3_ratio_T0 = bonf3_T0 / len(T0) if T0 else 0
        Q_T0 = sum(t*(t-1)*(t-2)*(t-3)//24 for t in tau_T0.values())
        bonf4_T0 = bonf3_T0 - Q_T0
        bonf4_ratio_T0 = bonf4_T0 / len(T0) if T0 else 0
        print(f"    Bonf3: +H={H_T0}, ratio={bonf3_ratio_T0:.4f}", flush=True)
        print(f"    Bonf4: -Q={Q_T0}, ratio={bonf4_ratio_T0:.4f}", flush=True)


print("\n\nDONE.", flush=True)
