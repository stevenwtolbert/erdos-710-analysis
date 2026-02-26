#!/usr/bin/env python3
"""
Z113a: LLL feasibility test for random injection matching.

Random injection: each k ∈ V independently picks a target h ∈ NH(k)
uniformly at random with probability 1/d(k).

Bad event B_h = "target h is chosen by ≥ 2 elements"
P(B_h) ≈ (L(h))²/2 for small L, where L(h) = Σ_{k:k|h} 1/d(k) = "load"
Dependency: B_h ~ B_{h'} iff ∃k with k|h and k|h'
D(h) = |{h' : ∃k, k|h and k|h'}| - 1

LLL condition: max P(B_h) · e · (max D(h) + 1) ≤ 1

Also test: Erdős-Spencer condition (τ_max ≤ |W|/(4e))
Also test: Haxell condition (d_min ≥ 2·τ_max)
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

print("=" * 90, flush=True)
print("Z113a: LLL / RANDOM INJECTION FEASIBILITY", flush=True)
print("=" * 90, flush=True)

for n in [5000, 10000, 20000, 50000, 100000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    for k in range(B + 1, N + 1):
        targets = get_targets(n, k, M)
        deg_cache[k] = len(targets)
        target_cache[k] = targets

    V_size = len(deg_cache)
    d_min = min(deg_cache.values())
    d_max = max(deg_cache.values())

    # Compute load L(h) = Σ_{k:k|h} 1/d(k) and τ(h) for each target
    load = defaultdict(float)
    tau = defaultdict(int)
    target_sources = defaultdict(set)  # h -> set of k dividing h in V

    for k in deg_cache:
        d = deg_cache[k]
        for h in target_cache[k]:
            load[h] += 1.0 / d
            tau[h] += 1
            target_sources[h].add(k)

    W_size = len(load)
    loads = sorted(load.values(), reverse=True)
    taus = sorted(tau.values(), reverse=True)

    max_load = loads[0]
    avg_load = sum(loads) / len(loads)  # = |V|/|W|
    max_tau = taus[0]

    # Load distribution
    load_gt1 = sum(1 for l in loads if l > 1.0)
    load_gt2 = sum(1 for l in loads if l > 2.0)

    # P(B_h) for worst h: use Poisson approximation
    # P(Poisson(λ) ≥ 2) = 1 - (1+λ)e^{-λ}
    p_worst = 1 - (1 + max_load) * math.exp(-max_load)
    p_avg = 1 - (1 + avg_load) * math.exp(-avg_load) if avg_load > 0 else 0

    # Dependency degree D(h): |{h' ≠ h : ∃k with k|h and k|h'}|
    # This is expensive to compute exactly; estimate from top-load targets
    # D(h) ≤ Σ_{k:k|h} (d(k) - 1)
    D_bound = {}
    for h in sorted(load.keys(), key=lambda x: -load[x])[:100]:
        D_bound[h] = sum(deg_cache[k] - 1 for k in target_sources[h])

    max_D = max(D_bound.values()) if D_bound else 0

    # LLL condition checks
    lll_symmetric = p_worst * math.e * (max_D + 1)
    erdos_spencer = max_tau <= W_size / (4 * math.e)
    haxell = d_min >= 2 * max_tau

    print(f"\n  n={n}: |V|={V_size}, |W|={W_size}, d_min={d_min}, d_max={d_max}", flush=True)
    print(f"  Load stats: max={max_load:.3f}, avg={avg_load:.3f}, "
          f"|L>1|={load_gt1} ({load_gt1/W_size:.1%}), |L>2|={load_gt2} ({load_gt2/W_size:.1%})", flush=True)
    print(f"  τ stats: max={max_tau}, top-5: {taus[:5]}", flush=True)
    print(f"  P(B_h) worst: {p_worst:.4f}, avg: {p_avg:.4f}", flush=True)
    print(f"  D_max (upper bound): {max_D}", flush=True)
    print(f"  LLL symmetric: P·e·(D+1) = {lll_symmetric:.2f} {'≤ 1 ✓' if lll_symmetric <= 1 else '> 1 ✗'}", flush=True)
    print(f"  Erdős-Spencer: τ_max={max_tau} vs |W|/(4e)={W_size/(4*math.e):.0f} "
          f"{'✓' if erdos_spencer else '✗'}", flush=True)
    print(f"  Haxell: d_min={d_min} vs 2·τ_max={2*max_tau} "
          f"{'✓' if haxell else '✗'}", flush=True)

    # What τ_max or load would we NEED for LLL to work?
    # Need P(B_h)·e·(D+1) ≤ 1
    # With D ≈ τ·d_avg, and P ≈ L²/2:
    # (L²/2)·e·(τ·d_avg+1) ≤ 1
    # For L = τ/d_min (worst case load): (τ²/2d_min²)·e·(τ·d_avg) ≤ 1
    d_avg = sum(deg_cache[k] for k in deg_cache) / V_size
    tau_need = (2 * d_min**2 / (math.e * d_avg))**(1/3)
    print(f"  LLL would need τ_max ≈ {tau_need:.1f} (have {max_tau})", flush=True)

    # Better estimate: what if most τ(h) ≤ τ₀?
    for tau_thresh in [2, 3, 5, 10]:
        frac_below = sum(1 for t in tau.values() if t <= tau_thresh) / W_size
        if frac_below > 0.99:
            print(f"    {frac_below:.1%} of targets have τ ≤ {tau_thresh}", flush=True)
            break


# Additional: what fraction of targets have load ≤ 1?
print("\n\nLoad distribution detail (n=50000):", flush=True)
n = 50000
L, M, N, delta = compute_params(n)
B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
load2 = defaultdict(float)
tau2 = defaultdict(int)
for k in range(B + 1, N + 1):
    d = len(get_targets(n, k, M))
    for h in get_targets(n, k, M):
        load2[h] += 1.0 / d
        tau2[h] += 1

for thresh in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    frac = sum(1 for l in load2.values() if l <= thresh) / len(load2)
    print(f"  L(h) ≤ {thresh:.1f}: {frac:.1%}", flush=True)

print("\nτ distribution (n=50000):", flush=True)
for thresh in [1, 2, 3, 5, 10, 20, 50]:
    frac = sum(1 for t in tau2.values() if t <= thresh) / len(tau2)
    print(f"  τ(h) ≤ {thresh}: {frac:.1%}", flush=True)

print("\n\nDONE.", flush=True)
