#!/usr/bin/env python3
"""
Z112d: DYADIC DEGREE BANDS — Exact α per band via HK

KEY INSIGHT: Instead of one V_rest block (which has α ≈ 1.25, too small),
partition V_rest into dyadic DEGREE bands:
  Band_0: deg ∈ [d_min+1, 2(d_min+1)-1]  e.g., [4, 7]
  Band_1: deg ∈ [2(d_min+1), 4(d_min+1)-1]  e.g., [8, 15]
  Band_2: deg ∈ [4(d_min+1), 8(d_min+1)-1]  e.g., [16, 31]
  ...

If each band has α ≈ min_deg(band), the FMC sum would be:
  1/d_min + Σ 1/min_deg(band_j) + B/(M-B)
  = 1/3 + (1/4 + 1/8 + 1/16 + ...) + ε
  = 1/3 + 1/2 + ε = 5/6 + ε < 1  ✓

The geometric series converges! This is the KEY: dyadic degree bands give a
CONVERGENT FMC sum, unlike per-degree blocks (which diverge as Σ 1/d).

Method: For each band, use the doubled/tripled HK to find exact α.
"""

import math
from collections import defaultdict, deque, Counter

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
    """Returns max matching size."""
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
    """Check if α(elements) ≥ p/q using (p copies left, q copies right) HK."""
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


def find_alpha_bracket(elements, target_cache, max_copies=8):
    """Find α in bracket [lo, hi) using integer-copy HK tests."""
    if not elements:
        return float('inf'), float('inf')

    # Test integer thresholds first
    last_pass = 1
    first_fail = None

    for copies in range(1, max_copies + 1):
        ok, _, _ = check_alpha_pq(elements, target_cache, copies, 1)
        if ok:
            last_pass = copies
        else:
            first_fail = copies
            break

    if first_fail is None:
        return last_pass, float('inf')

    # Now test rationals between last_pass and first_fail
    # Test p/q with small q
    best_pass = last_pass
    best_fail = first_fail

    for q in range(2, 5):
        for p in range(last_pass * q + 1, first_fail * q):
            alpha = p / q
            if alpha <= best_pass or alpha >= best_fail:
                continue
            if p * len(elements) > 100000:  # limit size
                continue
            ok, _, _ = check_alpha_pq(elements, target_cache, p, q)
            if ok:
                if alpha > best_pass:
                    best_pass = alpha
            else:
                if alpha < best_fail:
                    best_fail = alpha

    return best_pass, best_fail


# ============================================================
print("=" * 110, flush=True)
print("Z112d: DYADIC DEGREE BANDS — Exact α per band", flush=True)
print("=" * 110, flush=True)


for n in [15000, 20000, 30000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    d_min = int(math.floor(delta)) + 1

    print(f"\n{'='*80}", flush=True)
    print(f"n = {n}, δ = {delta:.3f}, d_min = {d_min}, N = {N}, B = {B}", flush=True)
    print(f"{'='*80}", flush=True)

    # Compute all elements and their degrees
    all_elements = {}
    target_cache = {}
    deg_dist = defaultdict(list)

    for k in range(B + 1, N + 1):
        d = get_degree(n, k, M)
        all_elements[k] = d
        target_cache[k] = get_targets(n, k, M)
        deg_dist[d].append(k)

    # Block 0: V_min (deg = d_min)
    V_min = deg_dist[d_min]

    # Dyadic degree bands for V_rest
    # Band j: deg ∈ [d_min+1 * 2^j, d_min+1 * 2^{j+1})
    # Special first band: just [d_min+1, 2(d_min+1)-1]
    base = d_min + 1  # = 4 for d_min = 3
    bands = []
    d_lo = base
    while True:
        d_hi = 2 * d_lo - 1
        band_elements = []
        for d in range(d_lo, d_hi + 1):
            band_elements.extend(deg_dist.get(d, []))
        if not band_elements:
            # Check if there are any remaining higher-degree elements
            remaining = []
            for d in sorted(deg_dist.keys()):
                if d > d_hi:
                    remaining.extend(deg_dist[d])
            if remaining:
                bands.append((d_lo, max(all_elements[k] for k in remaining), remaining))
            break
        bands.append((d_lo, d_hi, band_elements))
        d_lo = d_hi + 1

    print(f"\nV_min: |V_min| = {len(V_min)}, α = {d_min} (proved by coprime pair)", flush=True)

    # Test α for each band
    fmc_sum = 1.0 / d_min  # V_min contribution
    bterm = B / (M - B) if M > B else 0
    fmc_sum += bterm  # S₋ contribution

    print(f"S₋: B = {B}, α ≥ M/B - 1 = {M/B - 1:.1f}, 1/α = {bterm:.5f}", flush=True)
    print(f"\nDyadic degree bands:", flush=True)
    print(f"{'Band':>6s}  {'deg range':>12s}  {'|band|':>7s}  {'min deg':>7s}  {'max deg':>7s}  "
          f"{'α_lo':>6s}  {'α_hi':>6s}  {'1/α_lo':>7s}  Status", flush=True)
    print("-" * 95, flush=True)

    band_sum = 0
    for idx, (d_lo, d_hi, elements) in enumerate(bands):
        if not elements:
            continue

        min_d = min(all_elements[k] for k in elements)
        max_d = max(all_elements[k] for k in elements)

        # Find α bracket
        alpha_lo, alpha_hi = find_alpha_bracket(elements, target_cache, max_copies=min(min_d + 2, 12))

        inv_alpha = 1.0 / alpha_lo if alpha_lo > 0 else float('inf')
        band_sum += inv_alpha

        status = "OK" if alpha_lo >= min_d else "CHECK"
        print(f"  B{idx:>3d}  [{d_lo:>4d}, {d_hi:>4d}]  {len(elements):>7d}  {min_d:>7d}  {max_d:>7d}  "
              f"{alpha_lo:>6.2f}  {'∞' if alpha_hi == float('inf') else f'{alpha_hi:.2f}':>6s}  "
              f"{inv_alpha:>7.4f}  {status}", flush=True)

    fmc_total = 1.0/d_min + band_sum + bterm
    print(f"\nFMC sum = 1/{d_min} + Σ(1/α_band) + B/(M-B)", flush=True)
    print(f"       = {1.0/d_min:.4f} + {band_sum:.4f} + {bterm:.4f} = {fmc_total:.4f}", flush=True)
    print(f"STATUS: {'PASS ✓ (< 1)' if fmc_total < 1 else 'FAIL ✗ (≥ 1)'}", flush=True)


# Part 2: Alternative banding — strict [4,7], [8,15], [16,31], [32,63], ...
print("\n\n" + "=" * 110, flush=True)
print("ALTERNATIVE: Fixed dyadic bands [4,7], [8,15], [16,31], [32,63]", flush=True)
print("=" * 110, flush=True)

for n in [15000, 20000, 30000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    d_min = int(math.floor(delta)) + 1

    print(f"\nn = {n}, δ = {delta:.3f}, d_min = {d_min}", flush=True)

    # Compute elements
    target_cache = {}
    deg_map = {}
    for k in range(B + 1, N + 1):
        d = get_degree(n, k, M)
        deg_map[k] = d
        target_cache[k] = get_targets(n, k, M)

    # Fixed bands
    band_specs = [(4, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 255),
                  (256, 511), (512, 1023), (1024, 2047), (2048, 4095)]

    # V_min
    V_min = [k for k in deg_map if deg_map[k] == d_min]

    fmc_sum = 1.0 / d_min
    bterm = B / (M - B)
    fmc_sum += bterm

    print(f"  V_min: |{len(V_min)}|, α={d_min}, 1/α={1.0/d_min:.4f}", flush=True)

    for d_lo, d_hi in band_specs:
        elements = [k for k in deg_map if d_lo <= deg_map[k] <= d_hi and deg_map[k] > d_min]
        if not elements:
            continue

        min_d = min(deg_map[k] for k in elements)
        max_d = max(deg_map[k] for k in elements)

        # Quick tests: α ≥ 1, α ≥ d_lo, etc.
        # Just test the key thresholds for FMC
        alpha_lo = 1  # we know α ≥ 1

        # Test integer copies up to min_d + 1
        for copies in range(2, min(min_d + 2, 15)):
            ok, _, _ = check_alpha_pq(elements, target_cache, copies, 1)
            if ok:
                alpha_lo = copies
            else:
                break

        inv_alpha = 1.0 / alpha_lo
        fmc_sum += inv_alpha

        # Also try d_lo threshold specifically
        ok_dlo, _, _ = check_alpha_pq(elements, target_cache, d_lo, 1)

        print(f"  [{d_lo:>4d},{d_hi:>4d}]: |{len(elements):>5d}|, deg=[{min_d},{max_d}], "
              f"α≥{alpha_lo}, α≥{d_lo}:{ok_dlo}, 1/α={inv_alpha:.5f}", flush=True)

    print(f"  S₋: B/(M-B) = {bterm:.5f}", flush=True)
    print(f"  FMC TOTAL = {fmc_sum:.4f} {'< 1 ✓' if fmc_sum < 1 else '≥ 1 ✗'}", flush=True)


# Part 3: Detailed α for Band 0 [4,7] — this is the bottleneck
print("\n\n" + "=" * 110, flush=True)
print("DETAILED: Band [4,7] — the bottleneck band", flush=True)
print("=" * 110, flush=True)

for n in [15000, 20000, 30000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    d_min = int(math.floor(delta)) + 1

    target_cache = {}
    band47 = []
    for k in range(B + 1, N + 1):
        d = get_degree(n, k, M)
        if 4 <= d <= 7:
            band47.append(k)
            target_cache[k] = get_targets(n, k, M)

    if not band47:
        print(f"  n={n}: empty band [4,7]", flush=True)
        continue

    min_d = min(len(target_cache[k]) for k in band47)
    max_d = max(len(target_cache[k]) for k in band47)

    print(f"\n  n={n}: |band[4,7]| = {len(band47)}, deg range [{min_d}, {max_d}]", flush=True)

    # Exact α: test p/q thresholds
    for p, q in [(1,1), (3,2), (2,1), (5,2), (3,1), (7,2), (4,1)]:
        ok, match, total = check_alpha_pq(band47, target_cache, p, q)
        deficiency = total - match
        print(f"    α ≥ {p}/{q} = {p/q:.3f}: "
              f"{'PASS' if ok else f'FAIL (def={deficiency})'}", flush=True)
        if not ok and p == q:
            break  # if α < 1, something is very wrong

    # Sub-bands: deg 4 only, deg 5 only, etc.
    for d_target in [4, 5, 6, 7]:
        sub = [k for k in band47 if len(target_cache[k]) == d_target]
        if not sub:
            continue

        # Test α ≥ d_target
        ok, match, total = check_alpha_pq(sub, target_cache, d_target, 1)
        deficiency = total - match

        # Also test lower thresholds
        lower_tests = []
        for copies in range(1, d_target + 1):
            ok_c, _, _ = check_alpha_pq(sub, target_cache, copies, 1)
            if ok_c:
                lower_tests.append(copies)
            else:
                break

        print(f"    Sub deg={d_target}: |{len(sub):>4d}|, α≥{d_target}: "
              f"{'PASS' if ok else f'FAIL(def={deficiency})'}, "
              f"proven α≥{max(lower_tests) if lower_tests else 0}", flush=True)


print("\n\nDONE.", flush=True)
