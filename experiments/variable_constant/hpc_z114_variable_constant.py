#!/usr/bin/env python3
"""
Z114: Variable-Constant CS Approach — 2D Sweep

Key insight: f(n) = (2/√e + o(1))·n·√(ln n/ln ln n) only requires a
variable C(n) → 2/√e. If CS proves Hall at C(n) for each n, and
C(n) → 2/√e, we're done.

Computes CS(T₀) over a 2D grid of (n, C_mult) to determine whether
CS(T₀) → ∞ at any fixed C > base, and tracks C_crit(n) = min C_mult
where CS(T₀) ≥ 1.0.

Grid:
  n ∈ {2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000, 50000, 70000, 100000}
  C_mult ∈ {1.00, 1.01, 1.02, 1.03, 1.05, 1.07, 1.10, 1.15, 1.20, 1.50}

For n ≤ 20000: also run greedy adversarial
For n > 20000: T₀ proxy only
"""

import math
import sys
import time
from collections import defaultdict


# ── Helpers (from Z113b) ─────────────────────────────────────────────

BASE_C = 2 / math.exp(0.5)  # 2/√e ≈ 1.2131

def compute_params(n, c_val):
    """Compute L, M, N, delta, B for given n and constant c_val."""
    ln_n = math.log(n) if n > 1 else 1
    ln_ln_n = math.log(ln_n) if ln_n > 1 else 1
    L = int(math.ceil(c_val * n * math.sqrt(ln_n / ln_ln_n)))
    M = L - n
    N = n // 2
    delta = 2.0 * M / n - 1
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    return L, M, N, delta, B


def get_degree(n, k, M):
    """Degree of vertex k: number of multiples of k in (2n, n+L]."""
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
    """Set of multiples of k in (2n, n+L]."""
    L_val = M + n
    targets = set()
    j_lo = (2 * n) // k + 1
    j_hi = (n + L_val) // k
    for j in range(j_lo, j_hi + 1):
        m = k * j
        if 2 * n < m <= n + L_val:
            targets.add(m)
    return targets


def build_graph(n, c_mult):
    """
    Build bipartite graph for given (n, c_mult).
    Returns: deg_cache, target_cache, tau, V_size, d_min, delta, B, c_val
    """
    c_val = BASE_C * c_mult + 0.05
    L, M, N, delta, B = compute_params(n, c_val)

    if M <= 0 or N <= B:
        return None

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
        return None

    d_min = min(deg_cache.values())
    return deg_cache, target_cache, tau, V_size, d_min, delta, B, c_val


def compute_cs_T0(deg_cache, target_cache, tau):
    """
    Compute T₀ = {k : d(k)(d(k)-1) < ρ(k)} and CS(T₀).
    Returns: cs_T0, T0_size, T0_frac
    """
    V_size = len(deg_cache)

    # ρ(k) = Σ_{h ∈ NH(k)} (τ(h) - 1)
    rho = {}
    for k in deg_cache:
        rho[k] = sum(tau[h] - 1 for h in target_cache[k])

    T0 = [k for k in deg_cache if deg_cache[k] * (deg_cache[k] - 1) < rho[k]]
    T0_size = len(T0)
    T0_frac = T0_size / V_size if V_size > 0 else 0

    if T0:
        tau_T0 = defaultdict(int)
        for k in T0:
            for h in target_cache[k]:
                tau_T0[h] += 1
        sum_d_T0 = sum(deg_cache[k] for k in T0)
        sum_tau_sq_T0 = sum(t * t for t in tau_T0.values())
        cs_T0 = sum_d_T0 ** 2 / (T0_size * sum_tau_sq_T0) if sum_tau_sq_T0 > 0 else float('inf')
    else:
        cs_T0 = float('inf')  # T₀ empty ⟹ Hall trivially satisfied

    return cs_T0, T0_size, T0_frac


def compute_cs_greedy(deg_cache, target_cache, tau, max_prefix=3000):
    """
    Greedy adversarial CS: sort by d(d-1)-ρ ascending, track min CS over prefixes.
    Returns: cs_greedy
    """
    rho = {}
    for k in deg_cache:
        rho[k] = sum(tau[h] - 1 for h in target_cache[k])

    scored = sorted(deg_cache.keys(),
                    key=lambda k: deg_cache[k] * (deg_cache[k] - 1) - rho[k])

    # Full-V CS as baseline
    V_size = len(deg_cache)
    sum_d_full = sum(deg_cache.values())
    sum_tau_sq_full = sum(t * t for t in tau.values())
    best_cs = sum_d_full ** 2 / (V_size * sum_tau_sq_full) if sum_tau_sq_full > 0 else float('inf')

    tau_greedy = defaultdict(int)
    sum_d_g = 0
    limit = min(max_prefix, len(scored))

    for i, k in enumerate(scored[:limit]):
        sum_d_g += deg_cache[k]
        for h in target_cache[k]:
            tau_greedy[h] += 1
        sum_tau_sq_g = sum(t * t for t in tau_greedy.values())
        cs_g = sum_d_g ** 2 / ((i + 1) * sum_tau_sq_g) if sum_tau_sq_g > 0 else 0
        if cs_g < best_cs:
            best_cs = cs_g

    return best_cs


# ── Main sweep ───────────────────────────────────────────────────────

N_VALUES = [2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000, 50000, 70000, 100000]
C_MULTS = [1.00, 1.01, 1.02, 1.03, 1.05, 1.07, 1.10, 1.15, 1.20, 1.50]
GREEDY_THRESHOLD = 20000  # only run greedy for n ≤ this

print("=" * 120, flush=True)
print("Z114: Variable-Constant CS — 2D Sweep  (n × C_mult)", flush=True)
print(f"BASE_C = 2/√e = {BASE_C:.6f}", flush=True)
print(f"n values: {N_VALUES}", flush=True)
print(f"C_mult values: {C_MULTS}", flush=True)
print(f"Greedy adversarial for n ≤ {GREEDY_THRESHOLD}", flush=True)
print("=" * 120, flush=True)

# Header
print(f"\n{'n':>7s}  {'C_mult':>6s}  {'C_val':>7s}  {'delta':>6s}  {'d_min':>5s}  "
      f"{'|V|':>6s}  {'|T0|':>6s}  {'|T0|/|V|':>8s}  {'CS_T0':>8s}  "
      f"{'CS_grdy':>8s}  {'C_crit':>6s}", flush=True)
print("-" * 120, flush=True)

# Track C_crit(n) for summary
c_crit_data = {}
all_rows = []

t_start = time.time()

for n in N_VALUES:
    c_crit = None  # minimum C_mult where CS(T₀) ≥ 1.0

    for c_mult in C_MULTS:
        result = build_graph(n, c_mult)
        if result is None:
            print(f"{n:>7d}  {c_mult:>6.2f}  {'SKIP':>7s}  (M≤0 or N≤B)", flush=True)
            continue

        deg_cache, target_cache, tau, V_size, d_min, delta, B, c_val = result

        # CS on T₀
        cs_T0, T0_size, T0_frac = compute_cs_T0(deg_cache, target_cache, tau)

        # Greedy adversarial (only for small n)
        cs_greedy = None
        if n <= GREEDY_THRESHOLD:
            cs_greedy = compute_cs_greedy(deg_cache, target_cache, tau)

        # C_crit flag
        crit_flag = ""
        if cs_T0 >= 1.0 and c_crit is None:
            c_crit = c_mult
            crit_flag = " <=="

        cs_T0_str = f"{cs_T0:>8.4f}" if cs_T0 < 1e6 else f"{'INF':>8s}"
        cs_grdy_str = f"{cs_greedy:>8.4f}" if cs_greedy is not None else f"{'---':>8s}"

        row = (n, c_mult, c_val, delta, d_min, V_size, T0_size, T0_frac,
               cs_T0, cs_greedy)
        all_rows.append(row)

        print(f"{n:>7d}  {c_mult:>6.2f}  {c_val:>7.4f}  {delta:>6.3f}  {d_min:>5d}  "
              f"{V_size:>6d}  {T0_size:>6d}  {T0_frac:>8.1%}  {cs_T0_str}  "
              f"{cs_grdy_str}{crit_flag}", flush=True)

    c_crit_data[n] = c_crit
    print(flush=True)  # blank line between n-blocks

elapsed = time.time() - t_start
print(f"\nTotal time: {elapsed:.1f}s\n", flush=True)


# ── Summary: C_crit(n) ──────────────────────────────────────────────

print("=" * 80, flush=True)
print("SUMMARY: C_crit(n) = minimum C_mult where CS(T₀) ≥ 1.0", flush=True)
print("=" * 80, flush=True)
print(f"\n{'n':>7s}  {'C_crit':>8s}  {'Trend':>10s}", flush=True)
print("-" * 30, flush=True)

prev_crit = None
for n in N_VALUES:
    cc = c_crit_data.get(n)
    if cc is not None:
        trend = ""
        if prev_crit is not None:
            if cc < prev_crit:
                trend = "DECREASING"
            elif cc > prev_crit:
                trend = "increasing"
            else:
                trend = "flat"
        print(f"{n:>7d}  {cc:>8.2f}  {trend:>10s}", flush=True)
        prev_crit = cc
    else:
        print(f"{n:>7d}  {'> 1.50':>8s}  {'---':>10s}", flush=True)
        prev_crit = None

print(flush=True)


# ── Summary: CS(T₀) vs n for selected C_mult ────────────────────────

print("=" * 80, flush=True)
print("CS(T₀) vs n for selected C_mult values", flush=True)
print("=" * 80, flush=True)

for cm_show in [1.00, 1.05, 1.10, 1.20, 1.50]:
    print(f"\n  C_mult = {cm_show:.2f}:", flush=True)
    for row in all_rows:
        n_r, cm_r = row[0], row[1]
        if abs(cm_r - cm_show) < 0.001:
            cs_T0_r = row[8]
            T0_frac_r = row[7]
            cs_str = f"{cs_T0_r:.4f}" if cs_T0_r < 1e6 else "INF"
            print(f"    n={n_r:>7d}:  CS(T₀)={cs_str:>8s}  |T₀|/|V|={T0_frac_r:.1%}", flush=True)


# ── Summary: CS_greedy vs CS_T0 gap ─────────────────────────────────

print(f"\n{'='*80}", flush=True)
print("CS_greedy vs CS_T0 gap (n ≤ 20K only)", flush=True)
print("=" * 80, flush=True)
print(f"\n{'n':>7s}  {'C_mult':>6s}  {'CS_T0':>8s}  {'CS_grdy':>8s}  {'gap':>8s}", flush=True)
print("-" * 45, flush=True)

for row in all_rows:
    n_r, cm_r, _, _, _, _, _, _, cs_T0_r, cs_grdy_r = row
    if cs_grdy_r is not None:
        gap = cs_T0_r - cs_grdy_r if cs_T0_r < 1e6 else float('inf')
        cs_str = f"{cs_T0_r:.4f}" if cs_T0_r < 1e6 else "INF"
        gap_str = f"{gap:.4f}" if gap < 1e6 else "INF"
        print(f"{n_r:>7d}  {cm_r:>6.2f}  {cs_str:>8s}  {cs_grdy_r:>8.4f}  {gap_str:>8s}", flush=True)


# ── Decision ─────────────────────────────────────────────────────────

print(f"\n{'='*80}", flush=True)
print("DECISION POINT", flush=True)
print("=" * 80, flush=True)

crits = [(n, c_crit_data[n]) for n in N_VALUES if c_crit_data[n] is not None]
if len(crits) >= 2:
    first_cc = crits[0][1]
    last_cc = crits[-1][1]
    if last_cc < first_cc:
        print(f"\nC_crit is DECREASING: {first_cc:.2f} (n={crits[0][0]}) → "
              f"{last_cc:.2f} (n={crits[-1][0]})", flush=True)
        print("→ PROCEED to analytic argument (Step 3)", flush=True)
        print("→ Plan Rackspace Spot for dense large-n sweep", flush=True)
    elif last_cc > first_cc:
        print(f"\nC_crit is INCREASING: {first_cc:.2f} → {last_cc:.2f}", flush=True)
        print("→ APPROACH LIKELY FAILS — look for alternatives", flush=True)
    else:
        print(f"\nC_crit is FLAT at {first_cc:.2f}", flush=True)
        print("→ Need more data / larger n to determine trend", flush=True)
elif len(crits) == 1:
    print(f"\nOnly one C_crit found: n={crits[0][0]}, C_crit={crits[0][1]:.2f}", flush=True)
    print("→ Need more data", flush=True)
else:
    print("\nNo C_crit found in grid (CS(T₀) < 1 everywhere)", flush=True)
    print("→ APPROACH FAILS at these C_mult values", flush=True)

print("\nDONE.", flush=True)
