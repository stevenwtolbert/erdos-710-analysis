#!/usr/bin/env python3
"""
Z112e: EXACT α per DYADIC K-INTERVAL — The definitive test

Degree-based FMC fails because Σ 1/d diverges.
K-interval FMC might work because elements in [2^j, 2^{j+1}) have
better structure: both degree AND k-range are bounded by factor 2.

Z91 showed Σ 1/CS_j ≤ 0.85 at all J-transitions. But CS_j is for
the FULL interval, not worst-case subset. We need exact α_j.

Method: For each dyadic k-interval I_j = V ∩ [2^j, 2^{j+1}),
compute α(I_j) = min_{T⊆I_j} |NH(T)|/|T| using HK on multiplied graphs.

Then check: Σ_j 1/α(I_j) ≤ 1?

Also test: V_min separated out, then k-intervals for V_rest.
"""

import math
from collections import deque, Counter

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
    j_hi = (n + L_val) // k
    j_lo = (2 * n) // k + 1
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

def hopcroft_karp(adj, left_nodes):
    match_left = {}
    match_right = {}
    INF = float('inf')

    def bfs():
        queue = deque()
        dist = {}
        for u in left_nodes:
            if u not in match_left:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                w = match_right.get(v)
                if w is None:
                    found = True
                elif dist.get(w, INF) == INF:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        return found, dist

    def dfs(u, dist):
        for v in adj[u]:
            w = match_right.get(v)
            if w is None or (dist.get(w, INF) == dist[u] + 1 and dfs(w, dist)):
                match_left[u] = v
                match_right[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while True:
        found, dist = bfs()
        if not found:
            break
        for u in left_nodes:
            if u not in match_left:
                if dfs(u, dist):
                    matching += 1
    return matching


def check_alpha_pq(elements, target_cache, p, q):
    """Check if α(elements) ≥ p/q."""
    if not elements:
        return True, 0, 0
    left_nodes = []
    adj = {}
    for k in elements:
        targets = target_cache[k]
        right_nodes = [(h, j) for h in targets for j in range(q)]
        for i in range(p):
            node = (k, i)
            left_nodes.append(node)
            adj[node] = right_nodes
    matching = hopcroft_karp(adj, left_nodes)
    total_left = p * len(elements)
    return matching == total_left, matching, total_left


def find_alpha_range(elements, target_cache, max_int=15):
    """Find α in [lo, hi) via integer and half-integer tests."""
    if not elements:
        return float('inf'), float('inf')

    # Integer thresholds
    last_pass = 0
    for copies in range(1, min(max_int, len(elements)) + 1):
        ok, _, _ = check_alpha_pq(elements, target_cache, copies, 1)
        if ok:
            last_pass = copies
        else:
            break

    if last_pass >= max_int:
        return last_pass, float('inf')

    # Half-integer test between last_pass and last_pass+1
    p_test = 2 * last_pass + 1
    ok_half, _, _ = check_alpha_pq(elements, target_cache, p_test, 2)
    if ok_half:
        lo = p_test / 2
    else:
        lo = last_pass

    # Quarter tests
    for num, den in [(4*last_pass+1, 4), (4*last_pass+3, 4)]:
        if num * len(elements) > 200000:
            continue
        if num/den > lo:
            ok_q, _, _ = check_alpha_pq(elements, target_cache, num, den)
            if ok_q:
                lo = num/den

    return lo, last_pass + 1


# ============================================================
print("=" * 110, flush=True)
print("Z112e: EXACT α PER DYADIC K-INTERVAL", flush=True)
print("=" * 110, flush=True)

for n in [10000, 15000, 20000, 30000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    d_min = int(math.floor(delta)) + 1

    print(f"\n{'='*90}", flush=True)
    print(f"n = {n}, δ = {delta:.3f}, d_min = {d_min}, N = {N}, B = {B}, M = {M}", flush=True)
    print(f"{'='*90}", flush=True)

    # Build elements and targets
    target_cache = {}
    deg_cache = {}
    for k in range(B + 1, N + 1):
        d = get_degree(n, k, M)
        deg_cache[k] = d
        target_cache[k] = get_targets(n, k, M)

    # Dyadic k-intervals: I_j = V ∩ [2^j, 2^{j+1})
    j_min = int(math.ceil(math.log2(B + 1)))
    j_max = int(math.floor(math.log2(N)))

    intervals = []
    for j in range(j_min, j_max + 1):
        lo = max(2**j, B + 1)
        hi = min(2**(j+1) - 1, N)
        elements = [k for k in range(lo, hi + 1) if k in deg_cache]
        if elements:
            intervals.append((j, lo, hi, elements))

    # Compute α per interval
    print(f"\n{'j':>3s}  {'[lo, hi]':>15s}  {'|I_j|':>6s}  {'d_min':>5s}  {'d_max':>5s}  "
          f"{'d_avg':>5s}  {'α_lo':>6s}  {'α_hi':>6s}  {'1/α':>7s}", flush=True)
    print("-" * 85, flush=True)

    fmc_sum = 0
    for j, lo, hi, elements in intervals:
        degs = [deg_cache[k] for k in elements]
        min_d = min(degs)
        max_d = max(degs)
        avg_d = sum(degs) / len(degs)

        # Find α range
        alpha_lo, alpha_hi = find_alpha_range(elements, target_cache, max_int=min(min_d + 2, 20))
        inv_alpha = 1.0 / alpha_lo if alpha_lo > 0 else float('inf')
        fmc_sum += inv_alpha

        print(f"{j:>3d}  [{lo:>6d}, {hi:>6d}]  {len(elements):>6d}  {min_d:>5d}  {max_d:>5d}  "
              f"{avg_d:>5.1f}  {alpha_lo:>6.2f}  "
              f"{'∞' if alpha_hi == float('inf') else f'{alpha_hi:.0f}':>6s}  {inv_alpha:>7.4f}", flush=True)

    # Add S₋ contribution
    bterm = B / (M - B) if M > B else 0
    fmc_total = fmc_sum + bterm

    print(f"\nΣ 1/α(I_j) = {fmc_sum:.4f}", flush=True)
    print(f"S₋ term    = {bterm:.5f}", flush=True)
    print(f"FMC TOTAL  = {fmc_total:.4f}  {'< 1 ✓' if fmc_total < 1 else '≥ 1 ✗'}", flush=True)


# Part 2: Separate V_min from each interval, then compute per-interval α for V_rest only
print("\n\n" + "=" * 110, flush=True)
print("VARIANT: V_min separated + per-interval V_rest", flush=True)
print("=" * 110, flush=True)

for n in [10000, 15000, 20000, 30000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    d_min = int(math.floor(delta)) + 1

    print(f"\nn = {n}, δ = {delta:.3f}, d_min = {d_min}", flush=True)

    target_cache = {}
    deg_cache = {}
    for k in range(B + 1, N + 1):
        d = get_degree(n, k, M)
        deg_cache[k] = d
        target_cache[k] = get_targets(n, k, M)

    # Separate V_min
    V_min = [k for k in deg_cache if deg_cache[k] == d_min]

    # Dyadic k-intervals for V_rest elements only
    j_min = int(math.ceil(math.log2(B + 1)))
    j_max = int(math.floor(math.log2(N)))

    fmc_sum = 1.0 / d_min  # V_min contribution
    bterm = B / (M - B) if M > B else 0

    print(f"  V_min: |{len(V_min)}|, α={d_min}, 1/α={1.0/d_min:.4f}", flush=True)

    for j in range(j_min, j_max + 1):
        lo = max(2**j, B + 1)
        hi = min(2**(j+1) - 1, N)
        # V_rest elements in this interval (deg > d_min)
        elements = [k for k in range(lo, hi + 1) if k in deg_cache and deg_cache[k] > d_min]
        if not elements:
            continue

        degs = [deg_cache[k] for k in elements]
        min_d = min(degs)

        # Find α
        alpha_lo, alpha_hi = find_alpha_range(elements, target_cache, max_int=min(min_d + 2, 15))
        inv_alpha = 1.0 / alpha_lo if alpha_lo > 0 else float('inf')
        fmc_sum += inv_alpha

        print(f"  j={j}: [{lo},{hi}] |{len(elements):>5d}| deg=[{min_d},{max(degs)}] "
              f"α≥{alpha_lo:.2f}  1/α={inv_alpha:.4f}", flush=True)

    fmc_total = fmc_sum + bterm
    print(f"  FMC = {1.0/d_min:.4f} + {fmc_sum - 1.0/d_min:.4f} + {bterm:.4f} = {fmc_total:.4f} "
          f"{'< 1 ✓' if fmc_total < 1 else '≥ 1 ✗'}", flush=True)


print("\n\nDONE.", flush=True)
