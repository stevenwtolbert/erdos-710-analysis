#!/usr/bin/env python3
"""
Z113b: Deficiency approach — does the CS ratio for worst-case T → 1?

KEY INSIGHT: We don't need α ≥ 1 exactly. If def(G) = o(|V|),
the matching gives f(n) = (2/√e + o(1))·n·√(ln n/ln ln n).

Strategy: bound def using CS on worst-case T.
If CS gives |NH(T)| ≥ (1 - ε(n))|T| with ε(n) → 0,
then def ≤ ε(n)|V| → 0 relative to |V|.

ALSO: test if INCREASING C (the constant) helps.
With C = 2/√e + c, larger c means larger δ, d_min,
and potentially CS proves Hall exactly.
"""

import math
from collections import defaultdict

def compute_params_c(n, c_val):
    ln_n = math.log(n) if n > 1 else 1
    ln_ln_n = math.log(ln_n) if ln_n > 1 else 1
    L = int(math.ceil(c_val * n * math.sqrt(ln_n / ln_ln_n)))
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


# ============================================================
# PART 1: CS ratio for greedy adversarial T, trend with n
# ============================================================
print("=" * 90, flush=True)
print("PART 1: CS ratio for worst-case T vs n", flush=True)
print("=" * 90, flush=True)

C_TARGET = 2 / math.exp(0.5) + 0.05

print(f"\nC = {C_TARGET:.6f} (2/√e + 0.05)\n", flush=True)
print(f"{'n':>7s}  {'|V|':>6s}  {'δ':>5s}  {'d_min':>5s}  {'CS(V)':>7s}  "
      f"{'CS_adv':>7s}  {'ε=1-CS':>8s}  {'def≤ε|V|':>9s}", flush=True)
print("-" * 75, flush=True)

for n in [2000, 5000, 10000, 20000, 50000, 100000, 200000]:
    L, M, N, delta = compute_params_c(n, C_TARGET)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    tau = defaultdict(int)

    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)
        for h in target_cache[k]:
            tau[h] += 1

    V_size = len(deg_cache)
    d_min = min(deg_cache.values()) if deg_cache else 0

    # Full-V CS ratio
    sum_d = sum(deg_cache[k] for k in deg_cache)
    sum_tau_sq = sum(t*t for t in tau.values())
    cs_full = sum_d**2 / (V_size * sum_tau_sq) if sum_tau_sq > 0 else 0

    # Greedy adversarial CS: add elements that maximize Στ²
    # (equivalent to minimizing CS ratio)
    # Start with elements that have highest codegree
    rho = {}
    for k in deg_cache:
        rho[k] = sum(tau[h] - 1 for h in target_cache[k])

    # Elements where per-element CS condition fails
    T0 = [k for k in deg_cache if deg_cache[k] * (deg_cache[k] - 1) < rho[k]]

    if T0:
        # CS on T0
        tau_T0 = defaultdict(int)
        for k in T0:
            for h in target_cache[k]:
                tau_T0[h] += 1
        sum_d_T0 = sum(deg_cache[k] for k in T0)
        sum_tau_sq_T0 = sum(t*t for t in tau_T0.values())
        cs_T0 = sum_d_T0**2 / (len(T0) * sum_tau_sq_T0) if sum_tau_sq_T0 > 0 else 0
    else:
        cs_T0 = cs_full

    # True greedy adversarial (expensive, only for small n)
    cs_adv = cs_T0  # use T0 as proxy
    if n <= 20000:
        # Better greedy: sort by score d(d-1) - ρ ascending, take prefix
        scored = sorted(deg_cache.keys(), key=lambda k: deg_cache[k]*(deg_cache[k]-1) - rho[k])
        best_cs = cs_full
        tau_greedy = defaultdict(int)
        sum_d_g = 0
        for i, k in enumerate(scored[:min(3000, len(scored))]):
            sum_d_g += deg_cache[k]
            for h in target_cache[k]:
                tau_greedy[h] += 1
            sum_tau_sq_g = sum(t*t for t in tau_greedy.values())
            cs_g = sum_d_g**2 / ((i+1) * sum_tau_sq_g) if sum_tau_sq_g > 0 else 0
            if cs_g < best_cs:
                best_cs = cs_g
        cs_adv = best_cs

    epsilon = max(0, 1 - cs_adv)
    def_bound = epsilon * V_size

    print(f"{n:>7d}  {V_size:>6d}  {delta:>5.2f}  {d_min:>5d}  {cs_full:>7.4f}  "
          f"{cs_adv:>7.4f}  {epsilon:>8.5f}  {def_bound:>9.1f}", flush=True)


# ============================================================
# PART 2: Effect of increasing C
# ============================================================
print("\n\n" + "=" * 90, flush=True)
print("PART 2: What constant C makes CS prove Hall exactly?", flush=True)
print("=" * 90, flush=True)

print("""
If we increase C, we get larger δ, larger d_min, better expansion.
Test: at what C does CS adversarial ratio first exceed 1.0?
This gives a PROVABLE (analytic) lower bound f(n) ≥ C·n·√(ln n/ln ln n).
""", flush=True)

n = 50000

for c_mult in [1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]:
    c_val = (2 / math.exp(0.5)) * c_mult + 0.05
    L, M, N, delta = compute_params_c(n, c_val)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    if M <= 0 or N <= B:
        continue

    deg_cache = {}
    target_cache = {}
    tau = defaultdict(int)
    for k in range(B + 1, N + 1):
        d = get_degree(n, k, M)
        if d <= 0:
            continue
        deg_cache[k] = d
        targets = get_targets(n, k, M)
        target_cache[k] = targets
        for h in targets:
            tau[h] += 1

    V_size = len(deg_cache)
    if V_size == 0:
        continue
    d_min = min(deg_cache.values())

    # CS on full V
    sum_d = sum(deg_cache[k] for k in deg_cache)
    sum_tau_sq = sum(t*t for t in tau.values())
    cs_full = sum_d**2 / (V_size * sum_tau_sq) if sum_tau_sq > 0 else 0

    # Quick adversarial: T0 = elements where d(d-1) < ρ
    rho = {k: sum(tau[h]-1 for h in target_cache[k]) for k in deg_cache}
    T0 = [k for k in deg_cache if deg_cache[k]*(deg_cache[k]-1) < rho[k]]

    if T0:
        tau_T0 = defaultdict(int)
        for k in T0:
            for h in target_cache[k]:
                tau_T0[h] += 1
        sum_d_T0 = sum(deg_cache[k] for k in T0)
        sum_tau_sq_T0 = sum(t*t for t in tau_T0.values())
        cs_T0 = sum_d_T0**2 / (len(T0) * sum_tau_sq_T0) if sum_tau_sq_T0 > 0 else 0
        T0_frac = len(T0) / V_size
    else:
        cs_T0 = cs_full
        T0_frac = 0

    print(f"  C = {c_val:.4f} ({c_mult:.1f}× base): δ={delta:.2f}, d_min={d_min}, "
          f"|V|={V_size}, CS(V)={cs_full:.3f}, CS(T0)={cs_T0:.3f}, "
          f"|T0|/|V|={T0_frac:.1%}", flush=True)


# ============================================================
# PART 3: At what n does CS(T0) cross 1.0 for C_TARGET?
# ============================================================
print("\n\n" + "=" * 90, flush=True)
print("PART 3: CS(T0) trend — does it cross 1.0?", flush=True)
print("=" * 90, flush=True)

print(f"\nTracking CS(T0) and |T0|/|V| as n → ∞:\n", flush=True)

for n in [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]:
    L, M, N, delta = compute_params_c(n, C_TARGET)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    tau = defaultdict(int)
    for k in range(B + 1, N + 1):
        d = get_degree(n, k, M)
        deg_cache[k] = d
        targets = get_targets(n, k, M)
        target_cache[k] = targets
        for h in targets:
            tau[h] += 1

    V_size = len(deg_cache)
    d_min = min(deg_cache.values()) if deg_cache else 0

    rho = {k: sum(tau[h]-1 for h in target_cache[k]) for k in deg_cache}
    T0 = [k for k in deg_cache if deg_cache[k]*(deg_cache[k]-1) < rho[k]]

    if T0:
        tau_T0 = defaultdict(int)
        for k in T0:
            for h in target_cache[k]:
                tau_T0[h] += 1
        sum_d_T0 = sum(deg_cache[k] for k in T0)
        sum_tau_sq_T0 = sum(t*t for t in tau_T0.values())
        cs_T0 = sum_d_T0**2 / (len(T0) * sum_tau_sq_T0) if sum_tau_sq_T0 > 0 else 0
    else:
        cs_T0 = float('inf')

    eps = max(0, 1 - cs_T0)
    print(f"  n={n:>7d}: δ={delta:.3f}, d_min={d_min}, |T0|={len(T0):>6d}/{V_size:>6d} "
          f"({len(T0)/V_size:.1%}), CS(T0)={cs_T0:.5f}, ε={eps:.5f}", flush=True)


print("\n\nDONE.", flush=True)
