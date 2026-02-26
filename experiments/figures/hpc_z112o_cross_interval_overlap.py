#!/usr/bin/env python3
"""
Z112o: Cross-interval overlap analysis.

THE critical computation: for each target h in (2n, n+L],
how many dyadic intervals I_j have divisors of h?

This determines whether per-interval Hall → global Hall.

APPROACH:
  For any T ⊆ V with T_j = T ∩ I_j:
  |NH(T)| = |∪_j NH(T_j)|

  Method A (inclusion-exclusion):
  |NH(T)| ≥ Σ_j |NH(T_j)| - Σ_{j<j'} |NH(T_j) ∩ NH(T_{j'})|
           ≥ Σ_j α_j|T_j| - OVERLAP

  Method B (multiplicity):
  |NH(T)| = |{h : c_h(T) ≥ 1}| ≥ (Σ_j |NH(T_j)|) / μ_max

  Method C (direct HK on full V with multiplied graph):
  Compute global α via HK doubling on ALL of V.
"""

import math
from collections import defaultdict, deque

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


# ============================================================
# PART 1: Target multiplicity — how many intervals reach each target?
# ============================================================
print("=" * 100, flush=True)
print("PART 1: TARGET MULTIPLICITY c(h) = |{j : ∃k ∈ I_j with k|h}|", flush=True)
print("=" * 100, flush=True)

for n in [5000, 10000, 20000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    j_min = int(math.ceil(math.log2(B + 1)))
    j_max = int(math.floor(math.log2(N)))

    # For each target h, find which intervals have divisors reaching h
    target_intervals = defaultdict(set)  # h -> set of j values
    deg_cache = {}
    target_cache = {}

    for j in range(j_min, j_max + 1):
        lo = max(2**j, B + 1)
        hi = min(2**(j+1) - 1, N)
        for k in range(lo, hi + 1):
            d = get_degree(n, k, M)
            deg_cache[k] = d
            targets = get_targets(n, k, M)
            target_cache[k] = targets
            for h in targets:
                target_intervals[h].add(j)

    # Statistics on c(h)
    c_values = [len(s) for s in target_intervals.values()]
    c_max = max(c_values)
    c_avg = sum(c_values) / len(c_values)
    c_dist = defaultdict(int)
    for c in c_values:
        c_dist[c] += 1

    # Total excess coverage
    excess = sum(c - 1 for c in c_values)  # Σ (c_h - 1)
    total_nh = len(target_intervals)  # |NH(V)| = number of targets
    V_size = len(deg_cache)

    print(f"\n  n={n}: |V|={V_size}, |NH(V)|={total_nh}, M={M}", flush=True)
    print(f"  μ_max = {c_max}, avg c = {c_avg:.3f}", flush=True)
    print(f"  Total excess = Σ(c_h-1) = {excess}", flush=True)
    print(f"  c distribution:", flush=True)
    for c in sorted(c_dist.keys()):
        frac = c_dist[c] / total_nh
        print(f"    c={c}: {c_dist[c]:>7d} ({frac:.1%})", flush=True)

    # Per-interval surplus vs overlap
    print(f"\n  Per-interval analysis:", flush=True)
    total_surplus = 0
    total_pairwise_overlap = 0

    interval_nh = {}  # j -> set of targets
    interval_elems = {}  # j -> list of elements

    for j in range(j_min, j_max + 1):
        lo = max(2**j, B + 1)
        hi = min(2**(j+1) - 1, N)
        elems = [k for k in range(lo, hi + 1) if k in deg_cache]
        if not elems:
            continue
        interval_elems[j] = elems
        nh_j = set()
        for k in elems:
            nh_j.update(target_cache[k])
        interval_nh[j] = nh_j

        # Surplus: |NH(I_j)| - |I_j| (how much above Hall threshold)
        surplus_j = len(nh_j) - len(elems)
        total_surplus += surplus_j
        ratio_j = len(nh_j) / len(elems)
        print(f"    j={j}: |I_j|={len(elems):>6d}, |NH(I_j)|={len(nh_j):>7d}, "
              f"ratio={ratio_j:.3f}, surplus={surplus_j:>7d}", flush=True)

    # Pairwise overlaps
    intervals = sorted(interval_nh.keys())
    print(f"\n  Pairwise overlaps |NH(I_j) ∩ NH(I_{chr(106)}')|:", flush=True)
    for i, j1 in enumerate(intervals):
        for j2 in intervals[i+1:]:
            overlap = len(interval_nh[j1] & interval_nh[j2])
            total_pairwise_overlap += overlap
            if overlap > 0:
                print(f"    ({j1},{j2}): {overlap:>7d}", flush=True)

    print(f"\n  Total surplus = {total_surplus}", flush=True)
    print(f"  Total pairwise overlap = {total_pairwise_overlap}", flush=True)
    print(f"  Surplus - overlap = {total_surplus - total_pairwise_overlap}", flush=True)
    # I-E bound: |NH(V)| ≥ Σ|NH(I_j)| - pairwise_overlap ≥ Σ|I_j| + surplus - overlap
    ie_bound = sum(len(interval_nh[j]) for j in intervals) - total_pairwise_overlap
    print(f"  I-E lower bound on |NH(V)|: {ie_bound} vs |V|={V_size} "
          f"({'OK' if ie_bound >= V_size else 'FAIL'})", flush=True)


# ============================================================
# PART 2: ADVERSARIAL cross-interval subsets
# ============================================================
print("\n\n" + "=" * 100, flush=True)
print("PART 2: ADVERSARIAL cross-interval subsets", flush=True)
print("=" * 100, flush=True)

print("""
The I-E bound in Part 1 uses T = V (full vertex set).
The adversary picks T ⊊ V to minimize |NH(T)|/|T|.

Key question: does a cross-interval T have BETTER or WORSE
expansion than a within-interval T?

Test: greedy adversarial on full V (not restricted to intervals).
""", flush=True)

for n in [5000, 10000, 20000, 50000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    deg_cache = {}
    target_cache = {}
    for k in range(B + 1, N + 1):
        deg_cache[k] = get_degree(n, k, M)
        target_cache[k] = get_targets(n, k, M)

    # Global greedy: add element that minimizes |NH(T)|/|T|
    # Sort by degree (low first)
    elements_sorted = sorted(deg_cache.keys(), key=lambda k: deg_cache[k])

    T = set()
    nh = set()
    best_ratio = float('inf')
    best_size = 0
    worst_ratios = []  # track (|T|, ratio) at each step

    for step, k in enumerate(elements_sorted[:min(2000, len(elements_sorted))]):
        nh.update(target_cache[k])
        T.add(k)
        ratio = len(nh) / len(T)
        if ratio < best_ratio:
            best_ratio = ratio
            best_size = len(T)

        if len(T) <= 20 or len(T) % 100 == 0:
            worst_ratios.append((len(T), ratio))

    print(f"  n={n}: greedy min ratio = {best_ratio:.4f} at |T|={best_size}", flush=True)

    # Now try: greedy that can pick from ANY interval at each step
    # True greedy: pick element minimizing |NH(T ∪ {k})|/|T ∪ {k}|
    T2 = set()
    nh2 = set()
    best_ratio2 = float('inf')
    remaining = set(deg_cache.keys())

    # Only do true greedy for small n (expensive)
    if n <= 10000:
        for step in range(min(500, len(remaining))):
            best_k = None
            best_new_ratio = float('inf')
            for k in remaining:
                new_nh_size = len(nh2 | target_cache[k])
                new_ratio = new_nh_size / (len(T2) + 1)
                if new_ratio < best_new_ratio:
                    best_new_ratio = new_ratio
                    best_k = k
            if best_k is None:
                break
            T2.add(best_k)
            nh2.update(target_cache[best_k])
            remaining.discard(best_k)
            ratio2 = len(nh2) / len(T2)
            if ratio2 < best_ratio2:
                best_ratio2 = ratio2

        print(f"    TRUE greedy min ratio = {best_ratio2:.4f}", flush=True)

        # Check which intervals the worst-case T spans
        j_min_v = int(math.ceil(math.log2(B + 1)))
        interval_counts = defaultdict(int)
        for k in T2:
            j = int(math.floor(math.log2(k)))
            interval_counts[j] += 1
        intervals_used = sorted(interval_counts.keys())
        print(f"    Worst T spans intervals: {intervals_used}", flush=True)
        for j in intervals_used:
            print(f"      j={j}: {interval_counts[j]} elements", flush=True)


# ============================================================
# PART 3: For GLOBAL Hall, what does HK doubling give?
# ============================================================
print("\n\n" + "=" * 100, flush=True)
print("PART 3: GLOBAL expansion α(V) via HK doubling", flush=True)
print("=" * 100, flush=True)

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


def check_global_alpha(n, p, q):
    """Check if α(V) ≥ p/q via HK on multiplied graph."""
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    left_nodes = []
    adj = {}
    elements = list(range(B + 1, N + 1))

    for k in elements:
        targets = get_targets(n, k, M)
        right_nodes = [(h, j) for h in targets for j in range(q)]
        for i in range(p):
            node = (k, i)
            left_nodes.append(node)
            adj[node] = right_nodes

    matching = hopcroft_karp(adj, left_nodes)
    total_left = p * len(elements)
    return matching == total_left, matching, total_left


print("""
Test α(V) ≥ p/q for the GLOBAL vertex set V (not per-interval).
This tells us the global worst-case expansion.
""", flush=True)

for n in [2000, 5000, 10000]:
    print(f"\n  n={n}:", flush=True)
    for p, q in [(1, 1), (3, 2), (2, 1), (5, 2), (3, 1)]:
        V_size_est = n // 2 - int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n)))))
        if p * V_size_est > 200000:
            print(f"    α ≥ {p}/{q}: skip (too large: {p * V_size_est} left nodes)", flush=True)
            continue
        ok, match, total = check_global_alpha(n, p, q)
        status = "✓" if ok else f"✗ ({match}/{total})"
        print(f"    α ≥ {p}/{q}: {status}", flush=True)


# ============================================================
# PART 4: Deficiency structure at n=5000
# ============================================================
print("\n\n" + "=" * 100, flush=True)
print("PART 4: WHERE does the (2,1) HK fail? (deficiency structure)", flush=True)
print("=" * 100, flush=True)

print("""
When α < 2, there exists T with |NH(T)| < 2|T|.
Extract this T from the HK (2,1) matching deficiency.
""", flush=True)

for n in [5000, 10000]:
    L, M, N, delta = compute_params(n)
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n

    # Build (2,1) expanded graph
    elements = list(range(B + 1, N + 1))
    left_nodes = []
    adj = {}
    deg_cache_local = {}
    tc_local = {}
    for k in elements:
        targets = get_targets(n, k, M)
        tc_local[k] = targets
        deg_cache_local[k] = len(targets)
        right_nodes = list(targets)
        for i in range(2):
            node = (k, i)
            left_nodes.append(node)
            adj[node] = right_nodes

    # Run HK
    match_left = {}
    match_right = {}
    INF = float('inf')

    def bfs_mk():
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

    def dfs_mk(u, dist):
        for v in adj[u]:
            w = match_right.get(v)
            if w is None or (dist.get(w, INF) == dist[u] + 1 and dfs_mk(w, dist)):
                match_left[u] = v
                match_right[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while True:
        found, dist = bfs_mk()
        if not found:
            break
        for u in left_nodes:
            if u not in match_left:
                if dfs_mk(u, dist):
                    matching += 1

    total_left = 2 * len(elements)
    deficit = total_left - matching
    print(f"\n  n={n}: (2,1) matching = {matching}/{total_left}, deficit = {deficit}", flush=True)

    if deficit > 0:
        # Find unmatched left nodes
        unmatched = [u for u in left_nodes if u not in match_left]
        unmatched_elems = set(u[0] for u in unmatched)
        print(f"    {len(unmatched_elems)} elements with at least one copy unmatched", flush=True)

        # For each unmatched element, check degree and interval
        interval_counts = defaultdict(int)
        degree_counts = defaultdict(int)
        for k in unmatched_elems:
            j = int(math.floor(math.log2(k)))
            interval_counts[j] += 1
            degree_counts[deg_cache_local[k]] += 1

        print(f"    By interval:", flush=True)
        for j in sorted(interval_counts.keys()):
            print(f"      j={j}: {interval_counts[j]}", flush=True)
        print(f"    By degree:", flush=True)
        for d in sorted(degree_counts.keys()):
            print(f"      d={d}: {degree_counts[d]}", flush=True)

        # Find the actual violating set via König's theorem
        # The min vertex cover C has |C| = matching (König).
        # The complement V\C gives the maximum independent set.
        # The left side of V\C is the deficient set.
        # From HK: build alternating tree from unmatched left nodes
        reachable_left = set()
        reachable_right = set()
        queue = deque(unmatched)
        visited_left = set(u for u in unmatched)
        while queue:
            u = queue.popleft()
            reachable_left.add(u)
            for v in adj[u]:
                if v not in reachable_right:
                    reachable_right.add(v)
                    # Follow matching edge back
                    w = match_right.get(v)
                    if w is not None and w not in visited_left:
                        visited_left.add(w)
                        queue.append(w)

        # König: T* = reachable_left elements, NH(T*) = reachable_right
        T_star_elems = set(u[0] for u in reachable_left)
        # Each element has 2 copies, so |T*| in the expanded graph = len(reachable_left)
        # But we want the original elements
        nh_star = reachable_right

        # The deficiency in original terms:
        # Each target h can serve at most 1 copy (in (2,1) graph).
        # So |NH(T*)| < 2|T*| iff the reachable set has deficiency.
        print(f"    König deficient set: |T*|={len(T_star_elems)} elements, "
              f"|NH(T*)|={len(nh_star)} targets", flush=True)
        if T_star_elems:
            ratio_star = len(nh_star) / len(T_star_elems)
            print(f"    Expansion ratio: {ratio_star:.4f}", flush=True)
            print(f"    This means α(V) ≤ {ratio_star:.4f} at n={n}", flush=True)

            # What intervals does T* span?
            t_interval_counts = defaultdict(int)
            t_degree_counts = defaultdict(int)
            for k in T_star_elems:
                j = int(math.floor(math.log2(k)))
                t_interval_counts[j] += 1
                t_degree_counts[deg_cache_local[k]] += 1
            print(f"    T* by interval:", flush=True)
            for j in sorted(t_interval_counts.keys()):
                lo_j = 2**j
                hi_j = 2**(j+1) - 1
                total_j = min(hi_j, N) - max(lo_j, B+1) + 1
                print(f"      j={j}: {t_interval_counts[j]}/{total_j} "
                      f"({t_interval_counts[j]/total_j:.1%})", flush=True)
            print(f"    T* by degree:", flush=True)
            for d in sorted(t_degree_counts.keys())[:10]:
                print(f"      d={d}: {t_degree_counts[d]}", flush=True)
    else:
        print(f"    α(V) ≥ 2 at n={n}!", flush=True)


print("\n\nDONE.", flush=True)
