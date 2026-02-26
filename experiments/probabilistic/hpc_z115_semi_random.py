#!/usr/bin/env python3
"""
Z115: Semi-Random Nibble for Erdős 710 Bipartite Matching

Last-resort experiment. The Rödl nibble / KPSY semi-random method:
  Round i: each unmatched vertex picks a random available target.
  If a target is picked by exactly one vertex, match them.
  Repeat until done or stuck.

Then run HK on the residual to check matchability.

Key questions:
  1. Does the process find a complete matching?
  2. How large is the residual after the nibble phase?
  3. Does the residual shrink with n? (Would imply def = o(|V|))
  4. Does HK succeed on the residual?
  5. How do residual degree/codegree evolve per round?
"""

import math
import random
import time
import sys
from collections import defaultdict, deque

# ── Graph construction (from Z113b/Z114) ─────────────────────────────

BASE_C = 2 / math.exp(0.5)

def compute_params(n, c_val):
    ln_n = math.log(n) if n > 1 else 1
    ln_ln_n = math.log(ln_n) if ln_n > 1 else 1
    L = int(math.ceil(c_val * n * math.sqrt(ln_n / ln_ln_n)))
    M = L - n
    N = n // 2
    delta = 2.0 * M / n - 1
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))) if n > 10 else n
    return L, M, N, delta, B


def build_graph(n, c_val):
    """Build bipartite graph. Returns adjacency lists and metadata."""
    L, M, N, delta, B = compute_params(n, c_val)
    if M <= 0 or N <= B:
        return None

    L_val = M + n  # = L = n + M

    # V = (B, N], H = (2n, n+L]
    # adj_v[k] = list of targets h with k | h
    adj_v = {}
    # adj_h[h] = list of vertices k with k | h
    adj_h = defaultdict(list)

    for k in range(B + 1, N + 1):
        targets = []
        j_lo = (2 * n) // k + 1
        j_hi = L_val // k
        for j in range(j_lo, j_hi + 1):
            m = k * j
            if 2 * n < m <= L_val:
                targets.append(m)
        if targets:
            adj_v[k] = targets
            for h in targets:
                adj_h[h].append(k)

    V = sorted(adj_v.keys())
    H_all = sorted(adj_h.keys())

    return {
        'n': n, 'L': L_val, 'M': M, 'N': N, 'B': B, 'delta': delta,
        'adj_v': adj_v, 'adj_h': dict(adj_h),
        'V': V, 'H': H_all,
        'V_size': len(V), 'H_size': len(H_all),
    }


# ── Hopcroft-Karp on residual ────────────────────────────────────────

def hopcroft_karp(adj_v_sub, adj_h_sub, V_sub, H_sub):
    """
    HK max matching on a subgraph.
    adj_v_sub[k] = list of available targets for k
    Returns matching size.
    """
    match_v = {}  # k -> h
    match_h = {}  # h -> k

    def bfs():
        dist = {}
        queue = deque()
        for k in V_sub:
            if k not in match_v:
                dist[k] = 0
                queue.append(k)
            else:
                dist[k] = float('inf')
        found = False
        while queue:
            k = queue.popleft()
            for h in adj_v_sub.get(k, []):
                k2 = match_h.get(h)
                if k2 is None:
                    found = True
                elif dist.get(k2, float('inf')) == float('inf'):
                    dist[k2] = dist[k] + 1
                    queue.append(k2)
        return found, dist

    def dfs(k, dist):
        for h in adj_v_sub.get(k, []):
            k2 = match_h.get(h)
            if k2 is None or (dist.get(k2, float('inf')) == dist[k] + 1 and dfs(k2, dist)):
                match_v[k] = h
                match_h[h] = k
                return True
        dist[k] = float('inf')
        return False

    while True:
        found, dist = bfs()
        if not found:
            break
        for k in V_sub:
            if k not in match_v:
                dfs(k, dist)

    return len(match_v)


# ── Semi-random nibble ───────────────────────────────────────────────

def semi_random_nibble(graph, seed=42, max_rounds=500, verbose=False):
    """
    Run the semi-random nibble process.

    Returns dict with per-round stats and final results.
    """
    rng = random.Random(seed)

    adj_v = graph['adj_v']
    V_all = set(graph['V'])
    H_all = set(graph['H'])

    # Current state
    unmatched = set(V_all)
    available = set(H_all)
    matching = {}  # k -> h

    # For each vertex, track available targets (as sets for fast removal)
    avail_targets = {k: set(adj_v[k]) for k in V_all}

    round_stats = []
    total_matched = 0

    for rnd in range(1, max_rounds + 1):
        if not unmatched:
            break

        # Remove stuck vertices (degree 0 in residual)
        stuck = {k for k in unmatched if not (avail_targets[k] & available)}
        stuck_count = len(stuck)

        active = unmatched - stuck

        if not active:
            # All remaining vertices are stuck
            round_stats.append({
                'round': rnd, 'matched': 0, 'stuck_new': stuck_count,
                'remaining': len(unmatched), 'active': 0,
            })
            break

        # Each active vertex picks a random available target
        picks = defaultdict(list)  # h -> [list of k that picked h]
        for k in active:
            targets_avail = list(avail_targets[k] & available)
            if not targets_avail:
                continue
            h = rng.choice(targets_avail)
            picks[h].append(k)

        # Match unique picks
        matched_this_round = 0
        newly_occupied = []
        for h, vertices in picks.items():
            if len(vertices) == 1:
                k = vertices[0]
                matching[k] = h
                unmatched.discard(k)
                available.discard(h)
                newly_occupied.append(h)
                matched_this_round += 1

        total_matched += matched_this_round

        # Compute residual stats (sample-based for speed)
        remaining = len(unmatched)
        min_deg_resid = float('inf')
        max_deg_resid = 0
        if remaining > 0:
            sample = list(unmatched)[:min(500, remaining)]
            for k in sample:
                d = len(avail_targets[k] & available)
                min_deg_resid = min(min_deg_resid, d)
                max_deg_resid = max(max_deg_resid, d)

        stats = {
            'round': rnd,
            'matched': matched_this_round,
            'total_matched': total_matched,
            'stuck_new': stuck_count,
            'remaining': remaining,
            'active': len(active),
            'min_deg_sample': min_deg_resid if remaining > 0 else 0,
            'max_deg_sample': max_deg_resid if remaining > 0 else 0,
        }
        round_stats.append(stats)

        if verbose and (rnd <= 20 or rnd % 10 == 0 or matched_this_round == 0):
            print(f"  Round {rnd:>3d}: matched={matched_this_round:>5d}  "
                  f"remaining={remaining:>6d}  active={len(active):>6d}  "
                  f"stuck={stuck_count:>5d}  "
                  f"deg=[{min_deg_resid},{max_deg_resid}]", flush=True)

        if matched_this_round == 0 and stuck_count == len(unmatched):
            break

    return {
        'matching': matching,
        'unmatched': unmatched,
        'available': available,
        'round_stats': round_stats,
        'total_rounds': len(round_stats),
        'total_matched': total_matched,
        'residual_size': len(unmatched),
    }


# ── Main experiment ──────────────────────────────────────────────────

C_VAL = BASE_C + 0.05
N_VALUES = [1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000, 50000]
SEEDS = list(range(10))
HK_THRESHOLD = 30000  # run HK on residual for n ≤ this

print("=" * 110, flush=True)
print("Z115: Semi-Random Nibble for Erdős 710 Bipartite Matching", flush=True)
print(f"C = {C_VAL:.6f} (2/√e + 0.05)", flush=True)
print(f"n values: {N_VALUES}", flush=True)
print(f"Seeds: {len(SEEDS)} per n", flush=True)
print(f"HK on residual for n ≤ {HK_THRESHOLD}", flush=True)
print("=" * 110, flush=True)

# Header
print(f"\n{'n':>6s}  {'seed':>4s}  {'|V|':>6s}  {'delta':>5s}  {'rounds':>6s}  "
      f"{'matched':>7s}  {'resid':>6s}  {'resid%':>6s}  "
      f"{'HK_resid':>8s}  {'HALL':>4s}  {'time':>5s}", flush=True)
print("-" * 110, flush=True)

summary = {}  # n -> list of result dicts

for n in N_VALUES:
    t0 = time.time()
    graph = build_graph(n, C_VAL)
    if graph is None:
        print(f"{n:>6d}  SKIP (bad params)", flush=True)
        continue

    build_time = time.time() - t0
    V_size = graph['V_size']
    delta = graph['delta']

    results_for_n = []

    for seed in SEEDS:
        t1 = time.time()

        # Run nibble
        result = semi_random_nibble(graph, seed=seed, max_rounds=500,
                                     verbose=(seed == 0 and n <= 5000))

        nibble_time = time.time() - t1
        resid_size = result['residual_size']
        resid_pct = resid_size / V_size * 100 if V_size > 0 else 0

        # Run HK on residual
        hk_result = "---"
        hall_ok = "---"
        if resid_size > 0 and n <= HK_THRESHOLD:
            # Build residual subgraph
            unmatched_list = sorted(result['unmatched'])
            avail_set = result['available']
            adj_v_sub = {}
            for k in unmatched_list:
                tgts = [h for h in graph['adj_v'][k] if h in avail_set]
                if tgts:
                    adj_v_sub[k] = tgts
            hk_match = hopcroft_karp(adj_v_sub, {}, unmatched_list, [])
            hk_result = str(hk_match)
            hall_ok = "YES" if hk_match == resid_size else "NO"
        elif resid_size == 0:
            hk_result = "0"
            hall_ok = "YES"

        total_time = time.time() - t1

        row = {
            'n': n, 'seed': seed, 'V_size': V_size, 'delta': delta,
            'rounds': result['total_rounds'],
            'matched': result['total_matched'],
            'resid': resid_size, 'resid_pct': resid_pct,
            'hk_resid': hk_result, 'hall': hall_ok,
            'time': total_time,
        }
        results_for_n.append(row)

        print(f"{n:>6d}  {seed:>4d}  {V_size:>6d}  {delta:>5.2f}  "
              f"{result['total_rounds']:>6d}  {result['total_matched']:>7d}  "
              f"{resid_size:>6d}  {resid_pct:>5.1f}%  "
              f"{hk_result:>8s}  {hall_ok:>4s}  {total_time:>5.1f}s", flush=True)

    summary[n] = results_for_n

    elapsed_n = time.time() - t0
    print(f"  [{n}: {elapsed_n:.1f}s total]", flush=True)
    print(flush=True)


# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 110, flush=True)
print("SUMMARY: Residual after semi-random nibble", flush=True)
print("=" * 110, flush=True)

print(f"\n{'n':>6s}  {'|V|':>6s}  {'delta':>5s}  "
      f"{'avg_resid':>9s}  {'avg_resid%':>10s}  {'min_resid':>9s}  {'max_resid':>9s}  "
      f"{'avg_rounds':>10s}  {'HK_all_ok':>9s}", flush=True)
print("-" * 95, flush=True)

for n in N_VALUES:
    if n not in summary:
        continue
    rows = summary[n]
    V_size = rows[0]['V_size']
    delta = rows[0]['delta']

    resids = [r['resid'] for r in rows]
    resid_pcts = [r['resid_pct'] for r in rows]
    rounds_list = [r['rounds'] for r in rows]
    hk_oks = [r['hall'] for r in rows]

    avg_resid = sum(resids) / len(resids)
    avg_pct = sum(resid_pcts) / len(resid_pcts)
    min_resid = min(resids)
    max_resid = max(resids)
    avg_rounds = sum(rounds_list) / len(rounds_list)
    hk_all = "YES" if all(h == "YES" for h in hk_oks) else \
             ("NO" if any(h == "NO" for h in hk_oks) else "---")

    print(f"{n:>6d}  {V_size:>6d}  {delta:>5.2f}  "
          f"{avg_resid:>9.1f}  {avg_pct:>9.1f}%  {min_resid:>9d}  {max_resid:>9d}  "
          f"{avg_rounds:>10.1f}  {hk_all:>9s}", flush=True)


# ── Scaling analysis ─────────────────────────────────────────────────

print("\n" + "=" * 110, flush=True)
print("SCALING: Does residual/|V| shrink with n?", flush=True)
print("=" * 110, flush=True)

print(f"\n{'n':>6s}  {'|V|':>6s}  {'avg_resid%':>10s}  {'trend':>10s}", flush=True)
print("-" * 40, flush=True)

prev_pct = None
for n in N_VALUES:
    if n not in summary:
        continue
    rows = summary[n]
    avg_pct = sum(r['resid_pct'] for r in rows) / len(rows)
    trend = ""
    if prev_pct is not None:
        if avg_pct < prev_pct - 0.5:
            trend = "SHRINKING"
        elif avg_pct > prev_pct + 0.5:
            trend = "growing"
        else:
            trend = "~flat"
    print(f"{n:>6d}  {rows[0]['V_size']:>6d}  {avg_pct:>9.1f}%  {trend:>10s}", flush=True)
    prev_pct = avg_pct


# ── Decision ─────────────────────────────────────────────────────────

print(f"\n{'='*110}", flush=True)
print("DECISION", flush=True)
print("=" * 110, flush=True)

all_hall = True
any_hall_fail = False
for n in N_VALUES:
    if n not in summary:
        continue
    for r in summary[n]:
        if r['hall'] == "NO":
            any_hall_fail = True
            all_hall = False
        elif r['hall'] == "---":
            all_hall = False  # unknown

if any_hall_fail:
    print("\nHK FAILS on some residuals. The nibble leaves an unmatchable residual.", flush=True)
    print("→ Semi-random nibble DOES NOT WORK for this graph.", flush=True)
elif all_hall:
    print("\nHK succeeds on ALL residuals at ALL tested n.", flush=True)

    # Check if residual is shrinking
    first_n = [n for n in N_VALUES if n in summary][0]
    last_n = [n for n in N_VALUES if n in summary][-1]
    first_pct = sum(r['resid_pct'] for r in summary[first_n]) / len(summary[first_n])
    last_pct = sum(r['resid_pct'] for r in summary[last_n]) / len(summary[last_n])

    if last_pct < first_pct * 0.8:
        print(f"Residual fraction SHRINKING: {first_pct:.1f}% → {last_pct:.1f}%", flush=True)
        print("→ PROMISING: nibble + HK might give def = o(|V|)", flush=True)
    else:
        print(f"Residual fraction NOT shrinking: {first_pct:.1f}% → {last_pct:.1f}%", flush=True)
        print("→ Nibble leaves Θ(|V|) unmatched; HK still needed for bulk of work", flush=True)
        print("→ This is just a randomized pre-processing, not a proof technique", flush=True)
else:
    print("\nHK not run on all residuals (large n). Check data above.", flush=True)

print("\nDONE.", flush=True)
