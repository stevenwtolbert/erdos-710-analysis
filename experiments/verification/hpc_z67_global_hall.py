#!/usr/bin/env python3
"""
ERDŐS 710 — Z67: GLOBAL HALL VERIFICATION AT FMC FAILURE POINTS

Z66d found that the FMC condition Σ 1/α_j < 1 FAILS at certain n values
(n = 72000, 76000, 128000, 143000). But FMC failure ≠ Hall failure.

This script tests:
1. GLOBAL Hall via Hopcroft-Karp: does the bipartite graph S₊ → H_smooth have
   a perfect matching of size |S₊|? (If yes, Hall holds for ALL subsets.)
2. Cross-interval target multiplicity: μ_max = max_h |{j : h ∈ NH(I_j)}|
3. Per-interval surplus and cross-interval overlap
4. Whether min(α_j) ≥ μ_max (an alternative sufficient condition for global Hall)

Key insight: FMC is a SUFFICIENT condition derived from fractional matching.
When Σ 1/α_j > 1, the fractional certificate fails, but the integer matching
may still exist. Hopcroft-Karp gives the definitive answer via König's theorem.
"""

import math
import time
import sys
from collections import defaultdict, deque

C_TARGET = 2 / math.e**0.5
EPS = 0.05


def log(msg):
    print(msg)
    sys.stdout.flush()


def compute_params(n):
    ln_n = math.log(n)
    ln_ln_n = math.log(ln_n) if ln_n > 1 else 0.1
    L = (C_TARGET + EPS) * n * math.sqrt(ln_n / ln_ln_n)
    M = n + L
    B = int(math.sqrt(M))
    return L, M, B


def sieve_primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            for mult in range(p*p, limit + 1, p):
                sieve[mult] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]


def get_smooth_numbers_fast(B, lo, hi):
    if hi <= lo:
        return []
    size = hi - lo
    remaining = list(range(lo + 1, hi + 1))
    primes = sieve_primes(B)
    for p in primes:
        start = lo + 1
        first = start + (-start % p)
        for idx in range(first - lo - 1, size, p):
            while remaining[idx] % p == 0:
                remaining[idx] //= p
    return [lo + 1 + i for i in range(size) if remaining[i] == 1]


class HopcroftKarp:
    """Hopcroft-Karp maximum matching for bipartite graphs.

    Left vertices: 0..n_left-1
    Right vertices: 0..n_right-1
    adj[u] = list of right neighbors of left vertex u
    """

    def __init__(self, n_left, n_right, adj):
        self.n_left = n_left
        self.n_right = n_right
        self.adj = adj  # adj[u] = list of right vertices
        self.match_left = [-1] * n_left  # match_left[u] = right vertex or -1
        self.match_right = [-1] * n_right  # match_right[v] = left vertex or -1

    def bfs(self):
        """BFS to find shortest augmenting path length."""
        self.dist = [0] * self.n_left
        queue = deque()
        for u in range(self.n_left):
            if self.match_left[u] == -1:
                self.dist[u] = 0
                queue.append(u)
            else:
                self.dist[u] = float('inf')

        found = False
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                w = self.match_right[v]
                if w == -1:
                    found = True
                elif self.dist[w] == float('inf'):
                    self.dist[w] = self.dist[u] + 1
                    queue.append(w)
        return found

    def dfs(self, u):
        """DFS to find augmenting path from free left vertex u."""
        for v in self.adj[u]:
            w = self.match_right[v]
            if w == -1 or (self.dist[w] == self.dist[u] + 1 and self.dfs(w)):
                self.match_left[u] = v
                self.match_right[v] = u
                return True
        self.dist[u] = float('inf')
        return False

    def max_matching(self):
        """Return size of maximum matching."""
        matching = 0
        while self.bfs():
            for u in range(self.n_left):
                if self.match_left[u] == -1:
                    if self.dfs(u):
                        matching += 1
        return matching


class MaxFlowGraph:
    """Dinic's max-flow for computing exact α_j."""
    def __init__(self, n_nodes):
        self.n = n_nodes
        self.graph = [[] for _ in range(n_nodes)]

    def add_edge(self, u, v, cap):
        self.graph[u].append([v, cap, len(self.graph[v])])
        self.graph[v].append([u, 0.0, len(self.graph[u]) - 1])

    def bfs(self, s, t):
        level = [-1] * self.n
        level[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v, cap, _ in self.graph[u]:
                if level[v] < 0 and cap > 1e-12:
                    level[v] = level[u] + 1
                    queue.append(v)
                    if v == t:
                        return level
        return None if level[t] < 0 else level

    def dfs(self, u, t, f, level, it):
        if u == t:
            return f
        while it[u] < len(self.graph[u]):
            v, cap, rev = self.graph[u][it[u]]
            if cap > 1e-12 and level[v] == level[u] + 1:
                d = self.dfs(v, t, min(f, cap), level, it)
                if d > 1e-12:
                    self.graph[u][it[u]][1] -= d
                    self.graph[v][rev][1] += d
                    return d
            it[u] += 1
        return 0.0

    def max_flow(self, s, t):
        flow = 0.0
        while True:
            level = self.bfs(s, t)
            if level is None:
                break
            it = [0] * self.n
            while True:
                f = self.dfs(s, t, float('inf'), level, it)
                if f < 1e-12:
                    break
                flow += f
        return flow


def compute_exact_alpha(I_j_list, adj_j):
    """Compute exact α_j via binary search + max-flow."""
    left = I_j_list
    n_left = len(left)
    if n_left == 0:
        return float('inf')

    right_set = set()
    for k in left:
        right_set |= adj_j.get(k, set())
    right = sorted(right_set)
    n_right = len(right)
    if n_right == 0:
        return 0.0

    SOURCE = 0
    SINK = n_left + n_right + 1
    left_idx = {k: i + 1 for i, k in enumerate(left)}
    right_idx = {h: n_left + 1 + i for i, h in enumerate(right)}
    n_nodes = SINK + 1
    INF_CAP = float(n_right + 1)

    lo = 0.5
    hi = float(n_right) / n_left + 0.01

    edges = []
    for k in left:
        li = left_idx[k]
        for h in adj_j.get(k, set()):
            if h in right_idx:
                edges.append((li, right_idx[h]))

    for iteration in range(40):
        mid = (lo + hi) / 2.0
        G = MaxFlowGraph(n_nodes)
        for k in left:
            G.add_edge(SOURCE, left_idx[k], mid)
        for li, ri in edges:
            G.add_edge(li, ri, INF_CAP)
        for h in right:
            G.add_edge(right_idx[h], SINK, 1.0)
        flow = G.max_flow(SOURCE, SINK)
        if flow >= mid * n_left - 1e-9:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.001:
            break
    return lo


def analyze_n(n, t0):
    """Full analysis at a given n: global HK + per-interval exact + cross-interval."""
    L, M, B = compute_params(n)
    n_half = n // 2
    nL = int(n + L)
    delta = 2 * M / n - 1

    if B < 2 or n_half <= B:
        return None

    # Build bipartite graph
    log(f"\n{'─'*80}")
    log(f"  n = {n:>7} | δ = {delta:.3f} | B = {B}")
    log(f"{'─'*80}")

    t1 = time.time()
    S_plus = get_smooth_numbers_fast(B, B, n_half)
    H_smooth = get_smooth_numbers_fast(B, n, nL)

    if not S_plus or not H_smooth:
        log(f"  Empty sets: |S₊|={len(S_plus)}, |H|={len(H_smooth)}")
        return None

    H_set = set(H_smooth)
    adj = {}
    total_edges = 0
    for k in S_plus:
        targets = set()
        lo_mult = n // k + 1
        hi_mult = nL // k
        for m in range(lo_mult, hi_mult + 1):
            h = k * m
            if h in H_set:
                targets.add(h)
        adj[k] = targets
        total_edges += len(targets)

    # Dyadic intervals
    intervals = defaultdict(list)
    for k in S_plus:
        j = int(math.log2(k))
        intervals[j].append(k)
    js = sorted(intervals.keys())
    J = len(js)

    dt_graph = time.time() - t1
    log(f"  |S₊| = {len(S_plus)}, |H_smooth| = {len(H_smooth)}, |E| = {total_edges}, J = {J}")
    log(f"  Graph construction: {dt_graph:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    # PART 1: GLOBAL HOPCROFT-KARP
    # ═══════════════════════════════════════════════════════════════════
    log(f"\n  ── PART 1: GLOBAL HOPCROFT-KARP ──")

    # Map S₊ and H_smooth to contiguous indices
    s_idx = {k: i for i, k in enumerate(S_plus)}
    h_idx = {h: i for i, h in enumerate(H_smooth)}
    n_left = len(S_plus)
    n_right = len(H_smooth)

    # Build adjacency list for HK
    hk_adj = [[] for _ in range(n_left)]
    for k in S_plus:
        u = s_idx[k]
        for h in adj[k]:
            hk_adj[u].append(h_idx[h])

    t1 = time.time()
    hk = HopcroftKarp(n_left, n_right, hk_adj)
    matching = hk.max_matching()
    dt_hk = time.time() - t1

    hall_global = matching >= n_left
    deficiency = n_left - matching

    if hall_global:
        log(f"  GLOBAL HALL: ✓ PASS — max matching = {matching} = |S₊| = {n_left}")
    else:
        log(f"  GLOBAL HALL: ✗ FAIL — max matching = {matching} < |S₊| = {n_left}, deficiency = {deficiency}")
    log(f"  HK time: {dt_hk:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    # PART 2: CROSS-INTERVAL TARGET MULTIPLICITY
    # ═══════════════════════════════════════════════════════════════════
    log(f"\n  ── PART 2: CROSS-INTERVAL TARGET MULTIPLICITY ──")

    # For each target h, which intervals have a source that connects to it?
    target_intervals = defaultdict(set)
    for j in js:
        for k in intervals[j]:
            for h in adj[k]:
                target_intervals[h].add(j)

    # Multiplicity distribution
    mult_counts = defaultdict(int)
    for h in H_smooth:
        mu = len(target_intervals[h])
        mult_counts[mu] += 1

    mu_max = max(mult_counts.keys()) if mult_counts else 0
    total_targets = len(H_smooth)
    log(f"  μ_max (max # intervals sharing a target) = {mu_max}")
    log(f"  Multiplicity distribution:")
    for mu in sorted(mult_counts.keys()):
        pct = 100.0 * mult_counts[mu] / total_targets
        log(f"    μ={mu}: {mult_counts[mu]:>7} targets ({pct:>5.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # PART 3: PER-INTERVAL EXACT α_j + FMC CHECK
    # ═══════════════════════════════════════════════════════════════════
    log(f"\n  ── PART 3: PER-INTERVAL EXACT α_j ──")

    sum_fmc = 0.0
    min_alpha = float('inf')
    interval_data = []

    for j in js:
        I_j = sorted(intervals[j])
        n_ij = len(I_j)

        # Restrict adjacency to this interval
        adj_j = {k: adj[k] for k in I_j}

        t1 = time.time()
        alpha_j = compute_exact_alpha(I_j, adj_j)
        dt = time.time() - t1

        inv_alpha = 1.0 / alpha_j if alpha_j > 0 and alpha_j != float('inf') else 0.0
        sum_fmc += inv_alpha
        if alpha_j < min_alpha:
            min_alpha = alpha_j

        # Per-interval targets
        targets_j = set()
        for k in I_j:
            targets_j |= adj[k]

        interval_data.append({
            'j': j, 'n_left': n_ij, 'alpha': alpha_j,
            'inv_alpha': inv_alpha, 'n_targets': len(targets_j),
            'dt': dt,
        })

        elapsed = time.time() - t0
        log(f"    j={j:>2} |I|={n_ij:>6} α={alpha_j:>9.3f} 1/α={inv_alpha:>8.4f} "
            f"|NH(I_j)|={len(targets_j):>6} ({dt:.1f}s) [{elapsed:.0f}s total]")

    fmc_pass = sum_fmc < 1.0
    log(f"  Σ 1/α_j = {sum_fmc:.6f} — FMC: {'✓' if fmc_pass else '✗'}")
    log(f"  min α_j = {min_alpha:.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 4: CROSS-INTERVAL OVERLAP ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    log(f"\n  ── PART 4: CROSS-INTERVAL OVERLAP ──")

    # Compute pairwise overlap |NH(I_j) ∩ NH(I_{j'})|
    # and per-interval surplus (α_j - 1) * |I_j|
    interval_targets = {}
    for j in js:
        tgts = set()
        for k in intervals[j]:
            tgts |= adj[k]
        interval_targets[j] = tgts

    log(f"  Pairwise overlaps |NH(I_j) ∩ NH(I_j')|:")
    total_overlap = 0
    total_surplus = 0
    max_pairwise = 0

    for idx, j in enumerate(js):
        surplus_j = (interval_data[idx]['alpha'] - 1.0) * len(intervals[j])
        total_surplus += surplus_j

    for idx, j in enumerate(js):
        for idx2, j2 in enumerate(js):
            if j2 <= j:
                continue
            overlap = len(interval_targets[j] & interval_targets[j2])
            if overlap > 0:
                total_overlap += overlap
                if overlap > max_pairwise:
                    max_pairwise = overlap
                log(f"    j={j:>2} × j'={j2:>2}: overlap = {overlap:>6}")

    log(f"  Total pairwise overlap: {total_overlap}")
    log(f"  Max pairwise overlap: {max_pairwise}")
    log(f"  Total surplus Σ (α_j - 1)|I_j|: {total_surplus:.1f}")
    log(f"  Surplus > overlap? {'✓' if total_surplus > total_overlap else '✗'}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 5: ALTERNATIVE SUFFICIENT CONDITIONS
    # ═══════════════════════════════════════════════════════════════════
    log(f"\n  ── PART 5: ALTERNATIVE SUFFICIENT CONDITIONS ──")

    # Condition A: min(α_j) ≥ μ_max
    cond_a = min_alpha >= mu_max
    log(f"  A) min(α_j) = {min_alpha:.3f} ≥ μ_max = {mu_max}? {'✓' if cond_a else '✗'}")

    # Condition B: weighted avg ≥ μ_max
    if sum(len(intervals[j]) for j in js) > 0:
        weighted_avg = sum(
            interval_data[idx]['alpha'] * len(intervals[j])
            for idx, j in enumerate(js)
        ) / sum(len(intervals[j]) for j in js)
        cond_b = weighted_avg >= mu_max
        log(f"  B) weighted avg α = {weighted_avg:.3f} ≥ μ_max = {mu_max}? {'✓' if cond_b else '✗'}")

    # Condition C: inclusion-exclusion surplus > overlap
    cond_c = total_surplus > total_overlap
    log(f"  C) surplus > overlap: {total_surplus:.1f} > {total_overlap}? {'✓' if cond_c else '✗'}")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    log(f"\n  ── SUMMARY for n = {n} ──")
    log(f"  Global Hall (HK):     {'✓' if hall_global else '✗'}")
    log(f"  FMC (Σ 1/α < 1):     {'✓' if fmc_pass else '✗'} (Σ = {sum_fmc:.6f})")
    log(f"  Cond A (min α ≥ μ):  {'✓' if cond_a else '✗'}")
    if sum(len(intervals[j]) for j in js) > 0:
        log(f"  Cond B (avg α ≥ μ):  {'✓' if cond_b else '✗'}")
    log(f"  Cond C (surplus):     {'✓' if cond_c else '✗'}")

    return {
        'n': n, 'delta': delta, 'B': B, 'J': J,
        'n_splus': len(S_plus), 'n_hsmooth': len(H_smooth),
        'hall_global': hall_global, 'matching': matching, 'deficiency': deficiency,
        'sum_fmc': sum_fmc, 'min_alpha': min_alpha,
        'mu_max': mu_max, 'mult_counts': dict(mult_counts),
        'total_surplus': total_surplus, 'total_overlap': total_overlap,
        'interval_data': interval_data,
    }


def main():
    log("ERDŐS 710 — Z67: GLOBAL HALL VERIFICATION AT FMC FAILURE POINTS")
    log("=" * 80)
    log("Testing whether GLOBAL Hall holds at n values where FMC fails.")
    log("Method: Hopcroft-Karp (exact, by König's theorem).")
    t0 = time.time()

    # Test points: FMC failure points from Z66d + some passing points for comparison
    test_ns = [
        # FMC failures (from Z66d)
        72000,   # Σ=1.194
        76000,   # Σ=1.098
        # FMC passes (for comparison)
        74000,   # Σ=0.869
        78000,   # Σ=0.844
    ]

    results = []
    for n in test_ns:
        r = analyze_n(n, t0)
        if r:
            results.append(r)
        elapsed = time.time() - t0
        log(f"\n  Elapsed: {elapsed:.0f}s")
        if elapsed > 14400:  # 4h timeout
            log("  TIMEOUT")
            break

    # Phase 2: Larger FMC failures (slower, if time permits)
    elapsed = time.time() - t0
    if elapsed < 7200:
        log(f"\n{'='*80}")
        log("  PHASE 2: LARGER FMC FAILURE POINTS")
        log(f"{'='*80}")

        for n in [128000, 143000]:
            r = analyze_n(n, t0)
            if r:
                results.append(r)
            elapsed = time.time() - t0
            if elapsed > 14400:
                log("  TIMEOUT")
                break

    total_time = time.time() - t0

    # Final summary
    log(f"\n\n{'='*80}")
    log("  Z67 FINAL SUMMARY")
    log(f"{'='*80}\n")
    log(f"  {'n':>7} {'Global':>8} {'FMC':>5} {'Σ1/α':>8} {'min α':>7} {'μ_max':>6} {'surplus':>8} {'overlap':>8}")
    log(f"  {'─'*65}")

    for r in sorted(results, key=lambda x: x['n']):
        hall = "✓" if r['hall_global'] else "✗"
        fmc = "✓" if r['sum_fmc'] < 1.0 else "✗"
        log(f"  {r['n']:>7} {hall:>8} {fmc:>5} {r['sum_fmc']:>8.4f} "
            f"{r['min_alpha']:>7.3f} {r['mu_max']:>6} {r['total_surplus']:>8.1f} {r['total_overlap']:>8}")

    log(f"\n  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")

    all_hall = all(r['hall_global'] for r in results)
    if all_hall:
        log(f"\n  *** GLOBAL HALL HOLDS AT ALL {len(results)} TESTED VALUES ***")
        log("  *** FMC failures are SPURIOUS — the sufficient condition is too weak ***")
    else:
        log(f"\n  *** HALL FAILURES DETECTED ***")
        for r in results:
            if not r['hall_global']:
                log(f"    n = {r['n']}: deficiency = {r['deficiency']}")

    # Write state file
    state_path = "/home/ashbringer/projects/e710_new_H/states/state_79_z67_global_hall.md"
    with open(state_path, 'w') as f:
        f.write("# State 79: Z67 Global Hall Verification at FMC Failure Points\n\n")
        f.write("## Method\n\n")
        f.write("Hopcroft-Karp maximum matching on the GLOBAL bipartite graph\n")
        f.write("S₊ → H_smooth. By König's theorem, max matching = |S₊| ⟺\n")
        f.write("Hall's condition holds for ALL subsets.\n\n")
        f.write("## Results\n\n")
        f.write("| n | Global Hall | FMC | Σ1/α | min α | μ_max | surplus | overlap |\n")
        f.write("|---|-----------|-----|------|-------|-------|---------|--------|\n")
        for r in sorted(results, key=lambda x: x['n']):
            hall = "✓" if r['hall_global'] else "✗"
            fmc = "✓" if r['sum_fmc'] < 1.0 else "✗"
            f.write(f"| {r['n']} | {hall} | {fmc} | {r['sum_fmc']:.4f} | "
                    f"{r['min_alpha']:.3f} | {r['mu_max']} | {r['total_surplus']:.1f} | "
                    f"{r['total_overlap']} |\n")

        f.write(f"\n## Key Finding\n\n")
        if all_hall:
            f.write("**Global Hall holds at ALL FMC failure points.**\n")
            f.write("The FMC condition Σ 1/α_j < 1 is a sufficient but NOT necessary\n")
            f.write("condition. The bipartite graph has a perfect matching even when\n")
            f.write("the fractional matching certificate fails.\n")
        else:
            f.write("**Hall FAILS at some n values.**\n")

    log(f"  State file: {state_path}")


if __name__ == '__main__':
    main()
