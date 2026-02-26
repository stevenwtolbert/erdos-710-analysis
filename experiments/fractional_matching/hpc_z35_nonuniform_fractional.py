#!/usr/bin/env python3
"""
ERDOS 710 — EXPERIMENT Z35: NON-UNIFORM FRACTIONAL MATCHING

Three weight schemes for proving Hall's condition via fractional matching:
  Scheme 1: Popularity-adjusted w(k,h) = c(k)/mu(h)
  Scheme 2: Interval-weighted w(k,h) = alpha_j/deg(k) per dyadic interval
  Scheme 3: LP-optimal (scipy linprog) — exact feasibility + structure analysis

For each edge (k, h), assign w(k, h) >= 0 such that:
  Left:  sum_h w(k, h) = 1  for each k in S_+
  Right: sum_k w(k, h) <= 1  for each h in H_smooth
Existence => Hall's condition holds (LP duality / total unimodularity).
"""

import time
import sys
from math import gcd, log, sqrt, exp, floor, ceil, log2
from collections import Counter, defaultdict
import numpy as np

C_TARGET = 2 / sqrt(exp(1))
EPS = 0.05


def target_L(n, eps=EPS):
    if n < 3:
        return 3 * n
    return int((C_TARGET + eps) * n * sqrt(log(n) / log(log(n))))


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


def compute_targets(k, n, L):
    j0 = (2 * n) // k + 1
    j1 = (n + L) // k
    return set(k * j for j in range(j0, j1 + 1))


def get_dyadic_intervals(sqrt_nL, N):
    lo = int(sqrt_nL) + 1
    hi = N
    intervals = []
    j = int(log2(lo)) if lo > 0 else 0
    while (1 << j) <= hi:
        ivl_lo = max(lo, 1 << j)
        ivl_hi = min(hi, (1 << (j + 1)) - 1)
        if ivl_lo <= ivl_hi:
            intervals.append((j, ivl_lo, ivl_hi))
        j += 1
    return intervals


def build_graph(n):
    """Build bipartite graph G = (S_+, H_smooth) with divisibility edges."""
    L = target_L(n)
    if L <= n:
        return None
    M = L - n
    N = n // 2
    nL = n + L
    sqrt_nL = nL ** 0.5
    B = int(sqrt_nL)

    # S_+ = {k in [B+1, N] : k is B-smooth}
    lpf = {}
    for k in range(1, N + 1):
        lpf[k] = largest_prime_factor(k)

    s_plus = sorted([k for k in range(B + 1, N + 1) if lpf[k] <= B])

    # H_smooth = {h in (2n, n+L] : h is B-smooth}
    # For each h in that range, check B-smoothness
    h_lo = 2 * n + 1
    h_hi = int(n + L)

    # Build via edges: for each k in S_+, find multiples in (2n, n+L]
    edges = defaultdict(set)  # k -> set of targets h
    all_targets = set()
    for k in s_plus:
        j0 = (2 * n) // k + 1
        j1 = (n + L) // k
        for j in range(j0, j1 + 1):
            h = k * j
            if h_lo <= h <= h_hi:
                edges[k].add(h)
                all_targets.add(h)

    # Only keep targets that are B-smooth
    h_smooth = set()
    for h in all_targets:
        if largest_prime_factor(h) <= B:
            h_smooth.add(h)

    # Filter edges to only B-smooth targets
    for k in s_plus:
        edges[k] = edges[k] & h_smooth

    # Remove nodes with no edges
    s_plus = [k for k in s_plus if len(edges[k]) > 0]

    # Build reverse edges
    rev_edges = defaultdict(set)  # h -> set of sources k
    for k in s_plus:
        for h in edges[k]:
            rev_edges[h].add(k)

    h_smooth = sorted(h_smooth)

    # Dyadic intervals
    intervals = get_dyadic_intervals(sqrt_nL, N)

    return {
        'n': n, 'L': L, 'M': M, 'N': N, 'B': B,
        's_plus': s_plus,
        'h_smooth': h_smooth,
        'edges': edges,
        'rev_edges': rev_edges,
        'intervals': intervals,
        'sqrt_nL': sqrt_nL,
    }


# ═══════════════════════════════════════════════════════════════
# SCHEME 1: POPULARITY-ADJUSTED WEIGHTS
# ═══════════════════════════════════════════════════════════════

def scheme1_popularity_adjusted(G):
    """
    w(k, h) = c(k) / mu(h) where mu(h) = |{k' in S_+: k'|h}|
    c(k) = 1 / sum_{h: k|h} 1/mu(h)  so that left constraint = 1.
    Right constraint: sum_{k: k|h} c(k)/mu(h) = (1/mu(h)) sum_{k: k|h} c(k) <= 1
    Equivalently: sum_{k: k|h} c(k) <= mu(h).
    """
    edges = G['edges']
    rev_edges = G['rev_edges']
    s_plus = G['s_plus']
    h_smooth = G['h_smooth']

    # mu(h) = in-degree of h
    mu = {}
    for h in h_smooth:
        mu[h] = len(rev_edges[h])

    # c(k) = 1 / sum_{h in N(k)} 1/mu(h)
    c = {}
    for k in s_plus:
        s = sum(1.0 / mu[h] for h in edges[k] if mu[h] > 0)
        c[k] = 1.0 / s if s > 0 else 0.0

    # Right load for each h: (1/mu(h)) * sum_{k: k|h} c(k)
    max_right_load = 0.0
    worst_h = None
    load_dist = []

    for h in h_smooth:
        if mu[h] == 0:
            continue
        ck_sum = sum(c[k] for k in rev_edges[h])
        right_load = ck_sum / mu[h]
        load_dist.append(right_load)
        if right_load > max_right_load:
            max_right_load = right_load
            worst_h = h

    # Also check: sum_{k: k|h} c(k) <= mu(h)?
    max_ratio = 0.0
    for h in h_smooth:
        if mu[h] == 0:
            continue
        ck_sum = sum(c[k] for k in rev_edges[h])
        ratio = ck_sum / mu[h]
        if ratio > max_ratio:
            max_ratio = ratio

    load_arr = np.array(load_dist)

    return {
        'max_right_load': max_right_load,
        'worst_h': worst_h,
        'mean_load': np.mean(load_arr) if len(load_arr) > 0 else 0,
        'median_load': np.median(load_arr) if len(load_arr) > 0 else 0,
        'p90_load': np.percentile(load_arr, 90) if len(load_arr) > 0 else 0,
        'p99_load': np.percentile(load_arr, 99) if len(load_arr) > 0 else 0,
        'feasible': max_right_load <= 1.0 + 1e-9,
    }


# ═══════════════════════════════════════════════════════════════
# SCHEME 2: INTERVAL-WEIGHTED
# ═══════════════════════════════════════════════════════════════

def scheme2_interval_weighted(G):
    """
    w(k, h) = alpha_j / deg(k) for k in interval I_j.
    Left constraint: alpha_j * sum_h 1/deg(k) = alpha_j => satisfied if alpha_j = 1.
    But right constraint: sum_j alpha_j * sum_{k in I_j: k|h} 1/deg(k) <= 1.

    We optimize alpha_j to minimize max_h (right load).
    Use binary search on lambda, checking feasibility:
      for all h: sum_j alpha_j * load_j(h) <= lambda
      with alpha_j >= 0, and we want each k's flow to be 1:
        alpha_j * (deg(k)/deg(k)) = alpha_j, so alpha_j = 1 for all j.

    Actually alpha_j = 1 is forced by the left constraint.
    The question is whether max_h sum_{k|h} 1/deg(k) <= 1, which is
    the UNIFORM fractional matching. The point of interval-weighting is
    to use deg_j(k) (interval-restricted degree) instead of deg(k).

    Let's implement: w(k, h) = alpha_j / deg_j(k) where deg_j(k) = |N(k) intersect H_j|
    within interval I_j, and H_j = targets reachable from I_j.
    Left: sum_{h in H_j} alpha_j/deg_j(k) = alpha_j.
    Right: for each h, sum_{j: exists k in I_j with k|h} alpha_j * (tau_j(h)/? )

    Actually, let's think carefully. The right load on h from interval j is:
      R_j(h) = sum_{k in I_j: k|h} alpha_j / deg_j(k)
    where deg_j(k) = |{h' : k|h', h' in H_smooth}| (full degree, not restricted).

    Wait — if we use full deg(k), then left constraint gives alpha_j = 1.
    If we use interval-restricted deg_j(k), we need alpha_j chosen so that
    sum_{h in H} w(k,h) = 1, which means alpha_j * sum_{h: k|h} 1/deg_j(k)
    = alpha_j * deg(k)/deg_j(k).
    So alpha_j = deg_j(k)/deg(k) for all k in I_j? But deg_j(k) varies within I_j.

    Let's try a simpler approach: per-interval scaling.
    For each interval j, let CS_j = per-interval Cauchy-Schwarz ratio.
    Set alpha_j = 1/CS_j (the fraction of target capacity used).
    Then the right load from interval j on target h is at most alpha_j * tau_j(h)/...

    SIMPLIFICATION: Let's directly optimize alpha_j via LP or minimize max load.
    For each interval j, for each k in I_j, define w(k, h) = alpha_j / deg(k).
    Left constraint: alpha_j * (deg(k)/deg(k)) = alpha_j. So alpha_j must = 1.
    This reduces to uniform weights. Not useful.

    ALTERNATIVE: weight by INTERVAL-RESTRICTED degree.
    For k in I_j, deg_j(k) = |{h in H_smooth : k|h and h reachable from some k' in I_j}|.
    But this is just the full degree since H_smooth doesn't depend on j.

    CORRECT APPROACH: Use per-interval fractional matching.
    Each interval I_j independently matches to H_smooth with weights w_j(k, h).
    Combined: w(k, h) = w_j(k, h) where k in I_j.
    Left: for each k in I_j, sum_h w_j(k, h) = 1.
    Right: for each h, sum_j sum_{k in I_j: k|h} w_j(k, h) <= 1.

    If each interval uses uniform weights: w_j(k,h) = 1/deg(k).
    Right load = sum_j sum_{k in I_j: k|h} 1/deg(k).
    This is the same as the global uniform scheme.

    KEY IDEA: Scale down each interval by alpha_j < 1 so right constraints hold.
    w_j(k, h) = alpha_j / deg(k).
    Left: sum_h w_j(k,h) = alpha_j.  (NOT 1!)
    But we only need sum_j alpha_j >= 1 somehow... No, left must be exactly 1.

    ACTUAL CORRECT SCHEME:
    Partition targets: H_j = targets used (primarily) by interval j.
    Within interval j, match S_j to H_j.
    Right constraint per h: only load from one interval.

    Let's implement a practical version:
    For each target h, let load_j(h) = sum_{k in I_j: k|h} 1/deg(k).
    Total load = sum_j load_j(h).
    Find alpha_j > 0 with sum_j alpha_j * load_j(h) <= 1 for all h,
    AND for each k, alpha_j >= 1 (k in I_j) [so left constraint is met].

    So: alpha_j >= 1 for all j.
    But then sum_j alpha_j * load_j(h) >= sum_j load_j(h) = uniform load > 1. Doesn't help.

    The only way non-uniform helps is if we change the weight WITHIN an interval.
    Let's use popularity-adjusted weights PER INTERVAL.
    """
    edges = G['edges']
    rev_edges = G['rev_edges']
    s_plus = G['s_plus']
    h_smooth = G['h_smooth']
    intervals = G['intervals']
    sqrt_nL = G['sqrt_nL']
    N = G['N']

    # Classify each k into its dyadic interval
    k_to_interval = {}
    interval_elements = defaultdict(list)
    for k in s_plus:
        for idx, (j, lo, hi) in enumerate(intervals):
            if lo <= k <= hi:
                k_to_interval[k] = idx
                interval_elements[idx].append(k)
                break

    # For each interval j, compute per-interval mu_j(h) = |{k in I_j : k|h}|
    # and per-interval c_j(k) = 1 / sum_{h: k|h} 1/mu_j(h)
    # Then w(k,h) = c_j(k) / mu_j(h) where k in I_j

    # But this ignores cross-interval sharing. Let's try:
    # Global load on h = sum_j [sum_{k in I_j: k|h} c_j(k) / mu_j(h)]

    interval_mu = {}  # (interval_idx, h) -> count
    for idx in interval_elements:
        for k in interval_elements[idx]:
            for h in edges[k]:
                key = (idx, h)
                interval_mu[key] = interval_mu.get(key, 0) + 1

    # Per-interval c_j(k)
    c_interval = {}
    for idx in interval_elements:
        for k in interval_elements[idx]:
            s = 0.0
            for h in edges[k]:
                mu_jh = interval_mu.get((idx, h), 1)
                s += 1.0 / mu_jh
            c_interval[k] = 1.0 / s if s > 0 else 0.0

    # Right load: for each h, sum over intervals j of
    # sum_{k in I_j: k|h} c_j(k) / mu_j(h)
    max_right_load = 0.0
    worst_h = None
    load_dist = []

    for h in h_smooth:
        total_load = 0.0
        for k in rev_edges[h]:
            if k not in k_to_interval:
                continue
            idx = k_to_interval[k]
            mu_jh = interval_mu.get((idx, h), 1)
            total_load += c_interval[k] / mu_jh
        load_dist.append(total_load)
        if total_load > max_right_load:
            max_right_load = total_load
            worst_h = h

    load_arr = np.array(load_dist)

    return {
        'max_right_load': max_right_load,
        'worst_h': worst_h,
        'mean_load': np.mean(load_arr) if len(load_arr) > 0 else 0,
        'median_load': np.median(load_arr) if len(load_arr) > 0 else 0,
        'p90_load': np.percentile(load_arr, 90) if len(load_arr) > 0 else 0,
        'p99_load': np.percentile(load_arr, 99) if len(load_arr) > 0 else 0,
        'n_intervals': len(intervals),
        'feasible': max_right_load <= 1.0 + 1e-9,
    }


# ═══════════════════════════════════════════════════════════════
# SCHEME 3: LP-OPTIMAL (scipy linprog)
# ═══════════════════════════════════════════════════════════════

def scheme3_lp_optimal(G):
    """
    Directly solve the LP:
      min 0  (feasibility)
    s.t.
      sum_h w(k,h) = 1   for each k
      sum_k w(k,h) <= 1   for each h
      w(k,h) >= 0

    Since Hall holds, this LP is feasible. We minimize the max right load:
      min lambda
    s.t.
      sum_h w(k,h) = 1      for each k
      sum_k w(k,h) <= lambda for each h
      w >= 0, lambda >= 0
    """
    from scipy.optimize import linprog
    from scipy.sparse import csc_matrix

    edges = G['edges']
    rev_edges = G['rev_edges']
    s_plus = G['s_plus']
    h_smooth = G['h_smooth']

    # Create edge list and indices
    k_idx = {k: i for i, k in enumerate(s_plus)}
    h_idx = {h: i for i, h in enumerate(h_smooth)}
    nk = len(s_plus)
    nh = len(h_smooth)

    edge_list = []
    for k in s_plus:
        for h in edges[k]:
            if h in h_idx:
                edge_list.append((k_idx[k], h_idx[h]))
    ne = len(edge_list)

    print(f"    LP size: {nk} sources, {nh} targets, {ne} edges")
    print(f"    Variables: {ne} edge weights + 1 lambda = {ne + 1}")
    print(f"    Constraints: {nk} equality + {nh} inequality")

    if ne > 200000:
        print("    SKIPPED: LP too large")
        return None

    # Variables: w_0, w_1, ..., w_{ne-1}, lambda
    # Objective: minimize lambda (last variable)
    c_obj = np.zeros(ne + 1)
    c_obj[ne] = 1.0  # minimize lambda

    # Equality constraints: sum_h w(k,h) = 1 for each k
    # Build sparse A_eq
    eq_rows = []
    eq_cols = []
    eq_data = []
    for e_idx, (ki, hi) in enumerate(edge_list):
        eq_rows.append(ki)
        eq_cols.append(e_idx)
        eq_data.append(1.0)
    A_eq = csc_matrix((eq_data, (eq_rows, eq_cols)), shape=(nk, ne + 1))
    b_eq = np.ones(nk)

    # Inequality constraints: sum_k w(k,h) - lambda <= 0 for each h
    ineq_rows = []
    ineq_cols = []
    ineq_data = []
    for e_idx, (ki, hi) in enumerate(edge_list):
        ineq_rows.append(hi)
        ineq_cols.append(e_idx)
        ineq_data.append(1.0)
    # -lambda for each h constraint
    for hi in range(nh):
        ineq_rows.append(hi)
        ineq_cols.append(ne)  # lambda variable
        ineq_data.append(-1.0)
    A_ub = csc_matrix((ineq_data, (ineq_rows, ineq_cols)), shape=(nh, ne + 1))
    b_ub = np.zeros(nh)

    # Bounds: w >= 0, lambda >= 0
    bounds = [(0, None)] * (ne + 1)

    print("    Solving LP...", flush=True)
    t0 = time.time()
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')
    dt = time.time() - t0
    print(f"    LP solved in {dt:.1f}s, status={result.status} ({result.message})")

    if result.status != 0:
        print(f"    LP FAILED: {result.message}")
        return None

    lambda_opt = result.x[ne]
    weights = result.x[:ne]

    # Analyze right loads
    right_load = np.zeros(nh)
    for e_idx, (ki, hi) in enumerate(edge_list):
        right_load[hi] += weights[e_idx]

    # Analyze weight structure
    # For each edge, compute w(k,h) * deg(k) — if uniform, this = 1/deg(k)*deg(k) = 1
    # And w(k,h) * mu(h) — if popularity-adjusted, this = c(k)
    deg = {}
    for k in s_plus:
        deg[k] = len(edges[k])
    mu = {}
    for h in h_smooth:
        mu[h] = len(rev_edges[h])

    # Analyze: for each k, what is the distribution of w(k,h)?
    # Group by k
    k_weights = defaultdict(list)  # k -> list of (h, w)
    for e_idx, (ki, hi) in enumerate(edge_list):
        if weights[e_idx] > 1e-10:
            k_weights[ki].append((hi, weights[e_idx]))

    # For each k, compute: w(k,h)*deg(k) and w(k,h)*mu(h)
    uniform_dev = []  # |w(k,h)*deg(k) - 1|
    pop_adj_dev = []  # std of w(k,h)*mu(h) across h for each k
    weight_times_mu = []  # all w(k,h)*mu(h) values

    for ki, wlist in k_weights.items():
        k = s_plus[ki]
        dk = deg[k]
        for hi, w in wlist:
            h = h_smooth[hi]
            uniform_dev.append(abs(w * dk - 1.0))
            weight_times_mu.append(w * mu[h])
        # Check if w(k,h)*mu(h) is constant across h (popularity-adjusted)
        wmu_vals = [w * mu[h_smooth[hi]] for hi, w in wlist]
        if len(wmu_vals) > 1:
            pop_adj_dev.append(np.std(wmu_vals) / (np.mean(wmu_vals) + 1e-15))

    # Analyze by interval
    intervals = G['intervals']
    sqrt_nL = G['sqrt_nL']
    interval_stats = []
    for idx, (j, lo, hi_int) in enumerate(intervals):
        int_edges = [(e_idx, ki, hi_val) for e_idx, (ki, hi_val) in enumerate(edge_list)
                     if lo <= s_plus[ki] <= hi_int]
        if not int_edges:
            continue
        int_weights = [weights[e_idx] for e_idx, _, _ in int_edges]
        # Average w * deg(k)
        wdeg = [weights[e_idx] * deg[s_plus[ki]] for e_idx, ki, _ in int_edges]
        interval_stats.append({
            'j': j, 'lo': lo, 'hi': hi_int,
            'n_edges': len(int_edges),
            'mean_w': np.mean(int_weights),
            'mean_w_deg': np.mean(wdeg),
            'std_w_deg': np.std(wdeg),
        })

    return {
        'lambda_opt': lambda_opt,
        'feasible': lambda_opt <= 1.0 + 1e-6,
        'max_right_load': np.max(right_load),
        'mean_right_load': np.mean(right_load),
        'median_right_load': np.median(right_load),
        'p90_right_load': np.percentile(right_load, 90),
        'p99_right_load': np.percentile(right_load, 99),
        'solve_time': dt,
        'uniform_dev_mean': np.mean(uniform_dev) if uniform_dev else 0,
        'uniform_dev_max': np.max(uniform_dev) if uniform_dev else 0,
        'pop_adj_cv_mean': np.mean(pop_adj_dev) if pop_adj_dev else 0,
        'pop_adj_cv_max': np.max(pop_adj_dev) if pop_adj_dev else 0,
        'weight_times_mu_mean': np.mean(weight_times_mu) if weight_times_mu else 0,
        'weight_times_mu_std': np.std(weight_times_mu) if weight_times_mu else 0,
        'interval_stats': interval_stats,
        'nk': nk, 'nh': nh, 'ne': ne,
    }


# ═══════════════════════════════════════════════════════════════
# SCHEME 4: ITERATIVE REWEIGHTING (scalable alternative to LP)
# ═══════════════════════════════════════════════════════════════

def scheme4_iterative_reweight(G, max_iter=500, lr=0.5):
    """
    Iteratively adjust weights to reduce max right load.
    Start with uniform w(k,h) = 1/deg(k).
    Each iteration: identify overloaded targets, shift weight away from them.

    Multiplicative weights / mirror descent approach:
    - Maintain dual prices p(h) for each target
    - p(h) increases when target h is overloaded
    - w(k,h) proportional to exp(-p(h)) / Z(k)  [entropy regularized]
    """
    edges = G['edges']
    rev_edges = G['rev_edges']
    s_plus = G['s_plus']
    h_smooth = G['h_smooth']

    h_idx = {h: i for i, h in enumerate(h_smooth)}
    nh = len(h_smooth)

    # Precompute edge indices for vectorized operations
    k_edge_indices = {}  # k -> list of h_idx values
    for k in s_plus:
        k_edge_indices[k] = np.array([h_idx[h] for h in edges[k]], dtype=int)

    # Initialize dual prices
    price = np.zeros(nh)

    deg = {k: len(edges[k]) for k in s_plus}

    best_max_load = float('inf')
    best_price = None
    convergence = []

    for iteration in range(max_iter):
        # Compute weights: w(k,h) proportional to exp(-price[h]) / Z(k)
        right_load = np.zeros(nh)

        for k in s_plus:
            if deg[k] == 0:
                continue
            hidxs = k_edge_indices[k]
            exp_neg_p = np.exp(-price[hidxs])
            Z = np.sum(exp_neg_p)
            if Z < 1e-300:
                continue
            w_vec = exp_neg_p / Z
            np.add.at(right_load, hidxs, w_vec)

        max_load = np.max(right_load)

        if max_load < best_max_load:
            best_max_load = max_load
            best_price = price.copy()

        convergence.append(max_load)

        # Adaptive learning rate: decay after initial phase
        cur_lr = lr / (1.0 + iteration * 0.002)

        # Update prices: increase for overloaded, decrease for underloaded
        price += cur_lr * (right_load - 1.0)
        # Project: keep prices >= 0
        price = np.maximum(price, 0.0)

    # Final load computation with best prices
    price = best_price if best_price is not None else price
    right_load = np.zeros(nh)
    for k in s_plus:
        if deg[k] == 0:
            continue
        hidxs = k_edge_indices[k]
        exp_neg_p = np.exp(-price[hidxs])
        Z = np.sum(exp_neg_p)
        if Z < 1e-300:
            continue
        w_vec = exp_neg_p / Z
        np.add.at(right_load, hidxs, w_vec)

    return {
        'max_right_load': np.max(right_load),
        'mean_right_load': np.mean(right_load),
        'median_right_load': np.median(right_load),
        'p90_right_load': np.percentile(right_load, 90),
        'p99_right_load': np.percentile(right_load, 99),
        'feasible': np.max(right_load) <= 1.0 + 1e-6,
        'iterations': max_iter,
        'convergence_first5': convergence[:5],
        'convergence_last5': convergence[-5:],
    }


# ═══════════════════════════════════════════════════════════════
# UNIFORM BASELINE
# ═══════════════════════════════════════════════════════════════

def scheme0_uniform(G):
    """Baseline: w(k,h) = 1/deg(k). Right load on h = sum_{k|h} 1/deg(k)."""
    edges = G['edges']
    rev_edges = G['rev_edges']
    s_plus = G['s_plus']
    h_smooth = G['h_smooth']

    deg = {k: len(edges[k]) for k in s_plus}

    load_dist = []
    max_load = 0
    worst_h = None

    for h in h_smooth:
        load = sum(1.0 / deg[k] for k in rev_edges[h] if deg[k] > 0)
        load_dist.append(load)
        if load > max_load:
            max_load = load
            worst_h = h

    load_arr = np.array(load_dist)
    return {
        'max_right_load': max_load,
        'worst_h': worst_h,
        'mean_load': np.mean(load_arr),
        'median_load': np.median(load_arr),
        'p90_load': np.percentile(load_arr, 90),
        'p99_load': np.percentile(load_arr, 99),
        'feasible': max_load <= 1.0 + 1e-9,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════

def run_experiment(n_values_small, n_values_large):
    print("=" * 80)
    print("  EXPERIMENT Z35: NON-UNIFORM FRACTIONAL MATCHING")
    print("=" * 80)
    print(f"  Small n (with LP): {n_values_small}")
    print(f"  Large n (no LP):   {n_values_large}")
    print()

    all_results = {}

    for n in sorted(set(n_values_small + n_values_large)):
        print(f"\n{'─' * 70}")
        print(f"  n = {n}")
        print(f"{'─' * 70}")

        t0 = time.time()
        G = build_graph(n)
        dt_build = time.time() - t0

        if G is None:
            print(f"  SKIPPED: graph construction failed")
            continue

        nk = len(G['s_plus'])
        nh = len(G['h_smooth'])
        ne = sum(len(G['edges'][k]) for k in G['s_plus'])
        surplus = nh / nk if nk > 0 else 0

        print(f"  B={G['B']}, |S_+|={nk}, |H_smooth|={nh}, |E|={ne}")
        print(f"  Surplus ratio |H|/|S| = {surplus:.3f}")
        print(f"  Graph built in {dt_build:.2f}s")

        results = {'n': n, 'nk': nk, 'nh': nh, 'ne': ne, 'surplus': surplus}

        # --- Scheme 0: Uniform baseline ---
        print(f"\n  [Scheme 0] Uniform w(k,h) = 1/deg(k):")
        t0 = time.time()
        r0 = scheme0_uniform(G)
        dt = time.time() - t0
        print(f"    Max right load: {r0['max_right_load']:.4f}  "
              f"{'FEASIBLE' if r0['feasible'] else 'INFEASIBLE'}")
        print(f"    Mean={r0['mean_load']:.4f}, Median={r0['median_load']:.4f}, "
              f"P90={r0['p90_load']:.4f}, P99={r0['p99_load']:.4f}")
        print(f"    Worst target h={r0['worst_h']}, time={dt:.2f}s")
        results['scheme0'] = r0

        # --- Scheme 1: Popularity-adjusted ---
        print(f"\n  [Scheme 1] Popularity-adjusted w(k,h) = c(k)/mu(h):")
        t0 = time.time()
        r1 = scheme1_popularity_adjusted(G)
        dt = time.time() - t0
        print(f"    Max right load: {r1['max_right_load']:.4f}  "
              f"{'FEASIBLE' if r1['feasible'] else 'INFEASIBLE'}")
        print(f"    Mean={r1['mean_load']:.4f}, Median={r1['median_load']:.4f}, "
              f"P90={r1['p90_load']:.4f}, P99={r1['p99_load']:.4f}")
        print(f"    Worst target h={r1['worst_h']}, time={dt:.2f}s")
        results['scheme1'] = r1

        # --- Scheme 2: Per-interval popularity-adjusted ---
        print(f"\n  [Scheme 2] Per-interval popularity-adjusted:")
        t0 = time.time()
        r2 = scheme2_interval_weighted(G)
        dt = time.time() - t0
        print(f"    Max right load: {r2['max_right_load']:.4f}  "
              f"{'FEASIBLE' if r2['feasible'] else 'INFEASIBLE'}")
        print(f"    Mean={r2['mean_load']:.4f}, Median={r2['median_load']:.4f}, "
              f"P90={r2['p90_load']:.4f}, P99={r2['p99_load']:.4f}")
        print(f"    {r2['n_intervals']} intervals, time={dt:.2f}s")
        results['scheme2'] = r2

        # --- Scheme 4: Iterative reweighting (mirror descent) ---
        print(f"\n  [Scheme 4] Iterative reweighting (mirror descent, 500 iters):")
        t0 = time.time()
        r4 = scheme4_iterative_reweight(G, max_iter=500, lr=0.5)
        dt = time.time() - t0
        print(f"    Max right load: {r4['max_right_load']:.4f}  "
              f"{'FEASIBLE' if r4['feasible'] else 'INFEASIBLE'}")
        print(f"    Mean={r4['mean_right_load']:.4f}, Median={r4['median_right_load']:.4f}, "
              f"P90={r4['p90_right_load']:.4f}, P99={r4['p99_right_load']:.4f}")
        print(f"    Convergence first: {[f'{x:.3f}' for x in r4['convergence_first5']]}")
        print(f"    Convergence last:  {[f'{x:.3f}' for x in r4['convergence_last5']]}")
        print(f"    time={dt:.2f}s")
        results['scheme4'] = r4

        # --- Scheme 3: LP optimal (small n only) ---
        if n in n_values_small:
            print(f"\n  [Scheme 3] LP-optimal (scipy linprog):")
            t0 = time.time()
            r3 = scheme3_lp_optimal(G)
            dt = time.time() - t0
            if r3 is not None:
                print(f"    Optimal lambda = {r3['lambda_opt']:.6f}  "
                      f"{'FEASIBLE' if r3['feasible'] else 'INFEASIBLE'}")
                print(f"    Max right load: {r3['max_right_load']:.6f}")
                print(f"    Mean={r3['mean_right_load']:.4f}, Median={r3['median_right_load']:.4f}, "
                      f"P90={r3['p90_right_load']:.4f}, P99={r3['p99_right_load']:.4f}")
                print(f"\n    Weight structure analysis:")
                print(f"      Uniform deviation: mean |w*deg-1| = {r3['uniform_dev_mean']:.4f}, "
                      f"max = {r3['uniform_dev_max']:.4f}")
                print(f"      Pop-adj CV: mean = {r3['pop_adj_cv_mean']:.4f}, "
                      f"max = {r3['pop_adj_cv_max']:.4f}")
                print(f"      w*mu: mean = {r3['weight_times_mu_mean']:.4f}, "
                      f"std = {r3['weight_times_mu_std']:.4f}")

                if r3['interval_stats']:
                    print(f"\n    Per-interval LP weight stats:")
                    print(f"      {'Interval':>12s}  {'#edges':>8s}  {'mean_w':>10s}  "
                          f"{'mean_w*deg':>10s}  {'std_w*deg':>10s}")
                    for ist in r3['interval_stats']:
                        print(f"      [2^{ist['j']},{ist['hi']}]{' ':>3s}  {ist['n_edges']:>8d}  "
                              f"{ist['mean_w']:>10.6f}  {ist['mean_w_deg']:>10.4f}  "
                              f"{ist['std_w_deg']:>10.4f}")

                results['scheme3'] = r3
        else:
            print(f"\n  [Scheme 3] LP: SKIPPED (n too large)")

        all_results[n] = results

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'═' * 80}")
    print(f"  SUMMARY: MAX RIGHT LOAD BY SCHEME")
    print(f"{'═' * 80}")
    print(f"  {'n':>8s}  {'|S_+|':>6s}  {'|H|':>6s}  {'Surplus':>7s}  "
          f"{'Uniform':>8s}  {'PopAdj':>8s}  {'IntPop':>8s}  {'MirDesc':>8s}  {'LP':>8s}")
    print(f"  {'─' * 8}  {'─' * 6}  {'─' * 6}  {'─' * 7}  "
          f"{'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")

    for n in sorted(all_results.keys()):
        r = all_results[n]
        s0 = f"{r['scheme0']['max_right_load']:.3f}" if 'scheme0' in r else "—"
        s1 = f"{r['scheme1']['max_right_load']:.3f}" if 'scheme1' in r else "—"
        s2 = f"{r['scheme2']['max_right_load']:.3f}" if 'scheme2' in r else "—"
        s4 = f"{r['scheme4']['max_right_load']:.3f}" if 'scheme4' in r else "—"
        s3 = f"{r['scheme3']['lambda_opt']:.3f}" if 'scheme3' in r else "—"
        print(f"  {n:>8d}  {r['nk']:>6d}  {r['nh']:>6d}  {r['surplus']:>7.3f}  "
              f"{s0:>8s}  {s1:>8s}  {s2:>8s}  {s4:>8s}  {s3:>8s}")

    # Feasibility summary
    print(f"\n  FEASIBILITY (max load <= 1):")
    for scheme_name, scheme_key in [('Uniform', 'scheme0'), ('PopAdj', 'scheme1'),
                                     ('IntPop', 'scheme2'), ('MirDesc', 'scheme4'),
                                     ('LP', 'scheme3')]:
        feas = []
        for n in sorted(all_results.keys()):
            r = all_results[n]
            if scheme_key in r:
                f = r[scheme_key].get('feasible', False)
                feas.append((n, f))
        if feas:
            status = ", ".join(f"n={n}:{'Y' if f else 'N'}" for n, f in feas)
            all_pass = all(f for _, f in feas)
            print(f"    {scheme_name:>8s}: {status}  {'ALL PASS' if all_pass else 'SOME FAIL'}")

    # Key insight
    print(f"\n{'═' * 80}")
    print(f"  KEY INSIGHTS")
    print(f"{'═' * 80}")
    for n in sorted(all_results.keys()):
        r = all_results[n]
        loads = []
        for sk in ['scheme0', 'scheme1', 'scheme2', 'scheme4']:
            if sk in r:
                loads.append((sk, r[sk]['max_right_load']))
        if 'scheme3' in r:
            loads.append(('scheme3', r['scheme3']['lambda_opt']))
        if loads:
            best_sk, best_load = min(loads, key=lambda x: x[1])
            worst_sk, worst_load = max(loads, key=lambda x: x[1])
            print(f"  n={n:>6d}: best={best_sk}({best_load:.4f}), "
                  f"worst={worst_sk}({worst_load:.4f}), "
                  f"improvement={worst_load/best_load:.2f}x")

    print(f"\nDone.")


def deep_lp_analysis(n_values):
    """Deep analysis of LP-optimal weights for small n values."""
    print(f"\n\n{'=' * 80}")
    print(f"  DEEP LP ANALYSIS: WEIGHT STRUCTURE")
    print(f"{'=' * 80}")

    for n in n_values:
        print(f"\n{'─' * 70}")
        print(f"  n = {n}")
        print(f"{'─' * 70}")

        G = build_graph(n)
        if G is None:
            continue

        edges = G['edges']
        rev_edges = G['rev_edges']
        s_plus = G['s_plus']
        h_smooth = G['h_smooth']
        intervals = G['intervals']

        nk = len(s_plus)
        nh = len(h_smooth)
        ne = sum(len(edges[k]) for k in s_plus)
        print(f"  |S_+|={nk}, |H_smooth|={nh}, |E|={ne}")

        # Solve LP
        from scipy.optimize import linprog
        from scipy.sparse import csc_matrix

        k_idx = {k: i for i, k in enumerate(s_plus)}
        h_idx = {h: i for i, h in enumerate(h_smooth)}

        edge_list = []
        for k in s_plus:
            for h in edges[k]:
                if h in h_idx:
                    edge_list.append((k_idx[k], h_idx[h]))
        ne_actual = len(edge_list)

        c_obj = np.zeros(ne_actual + 1)
        c_obj[ne_actual] = 1.0

        eq_rows, eq_cols, eq_data = [], [], []
        for e_idx, (ki, hi) in enumerate(edge_list):
            eq_rows.append(ki)
            eq_cols.append(e_idx)
            eq_data.append(1.0)
        A_eq = csc_matrix((eq_data, (eq_rows, eq_cols)), shape=(nk, ne_actual + 1))
        b_eq = np.ones(nk)

        ineq_rows, ineq_cols, ineq_data = [], [], []
        for e_idx, (ki, hi) in enumerate(edge_list):
            ineq_rows.append(hi)
            ineq_cols.append(e_idx)
            ineq_data.append(1.0)
        for hi in range(nh):
            ineq_rows.append(hi)
            ineq_cols.append(ne_actual)
            ineq_data.append(-1.0)
        A_ub = csc_matrix((ineq_data, (ineq_rows, ineq_cols)), shape=(nh, ne_actual + 1))
        b_ub = np.zeros(nh)
        bounds = [(0, None)] * (ne_actual + 1)

        print(f"  Solving LP ({ne_actual} variables)...", flush=True)
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')
        if result.status != 0:
            print(f"  LP FAILED")
            continue

        lambda_opt = result.x[ne_actual]
        weights = result.x[:ne_actual]
        print(f"  Optimal lambda = {lambda_opt:.6f}")

        # Compute degrees and multiplicities
        deg = {k: len(edges[k]) for k in s_plus}
        mu = {h: len(rev_edges[h]) for h in h_smooth}

        # For each k, analyze: what fraction of weight goes to low-mu vs high-mu targets?
        print(f"\n  === Weight distribution by target multiplicity ===")
        print(f"  For each k, partition weight by mu(h) of target h")

        # Collect per-k stats
        k_weight_by_mu = defaultdict(lambda: defaultdict(float))
        for e_idx, (ki, hi) in enumerate(edge_list):
            if weights[e_idx] > 1e-12:
                h = h_smooth[hi]
                m = mu[h]
                k_weight_by_mu[ki][m] += weights[e_idx]

        # Aggregate: what fraction of total weight goes to each mu bucket?
        mu_buckets = defaultdict(float)
        total_weight = 0
        for ki, mu_dist in k_weight_by_mu.items():
            for m, w in mu_dist.items():
                mu_buckets[m] += w
                total_weight += w

        print(f"  mu(h)  total_weight  frac_of_total  #targets_with_mu")
        mu_count = Counter(mu[h] for h in h_smooth)
        for m in sorted(mu_buckets.keys()):
            frac = mu_buckets[m] / total_weight if total_weight > 0 else 0
            print(f"  {m:>5d}  {mu_buckets[m]:>12.4f}  {frac:>13.4f}  {mu_count[m]:>15d}")

        # For each k: compute w(k,h)*mu(h) — is it constant?
        print(f"\n  === Is w(k,h)*mu(h) approximately constant across h for each k? ===")
        print(f"  (If yes, the LP solution is popularity-adjusted)")
        cv_list = []
        for ki, mu_dist in k_weight_by_mu.items():
            k = s_plus[ki]
            wmu_vals = []
            for e_idx, (ki2, hi) in enumerate(edge_list):
                if ki2 == ki and weights[e_idx] > 1e-12:
                    wmu_vals.append(weights[e_idx] * mu[h_smooth[hi]])
            if len(wmu_vals) > 1:
                cv = np.std(wmu_vals) / np.mean(wmu_vals)
                cv_list.append(cv)

        if cv_list:
            print(f"  CV of w(k,h)*mu(h) across h:  mean={np.mean(cv_list):.4f}, "
                  f"median={np.median(cv_list):.4f}, max={np.max(cv_list):.4f}")
            print(f"  If CV << 1, the LP is popularity-adjusted.")
            print(f"  CV < 0.1: {sum(1 for cv in cv_list if cv < 0.1)}/{len(cv_list)}")
            print(f"  CV < 0.3: {sum(1 for cv in cv_list if cv < 0.3)}/{len(cv_list)}")
            print(f"  CV < 0.5: {sum(1 for cv in cv_list if cv < 0.5)}/{len(cv_list)}")

        # For each k: compute w(k,h)*deg(k) — is it close to 1? (uniform test)
        print(f"\n  === Is w(k,h)*deg(k) approximately 1? (uniform test) ===")
        wdeg_all = []
        for e_idx, (ki, hi) in enumerate(edge_list):
            if weights[e_idx] > 1e-12:
                k = s_plus[ki]
                wdeg_all.append(weights[e_idx] * deg[k])
        if wdeg_all:
            arr = np.array(wdeg_all)
            print(f"  w*deg stats: mean={np.mean(arr):.4f}, std={np.std(arr):.4f}, "
                  f"min={np.min(arr):.4f}, max={np.max(arr):.4f}")

        # KEY: What is the relationship between w(k,h), deg(k), and mu(h)?
        # Test: w(k,h) = f(k) * g(h) — is the weight matrix rank-1?
        print(f"\n  === Rank-1 test: w(k,h) = f(k)*g(h)? ===")
        # For each k, compute average weight
        k_avg_w = {}
        for ki, mu_dist in k_weight_by_mu.items():
            k = s_plus[ki]
            total_w = sum(weights[e_idx] for e_idx, (ki2, _) in enumerate(edge_list) if ki2 == ki and weights[e_idx] > 1e-12)
            n_nonzero = sum(1 for e_idx, (ki2, _) in enumerate(edge_list) if ki2 == ki and weights[e_idx] > 1e-12)
            k_avg_w[ki] = total_w / n_nonzero if n_nonzero > 0 else 0

        # For each h, compute average weight
        h_avg_w = {}
        for hi_val in range(nh):
            wlist = [weights[e_idx] for e_idx, (_, hi2) in enumerate(edge_list) if hi2 == hi_val and weights[e_idx] > 1e-12]
            h_avg_w[hi_val] = np.mean(wlist) if wlist else 0

        # Test: w(k,h) / (k_avg_w[k] * h_avg_w[h]) close to constant?
        ratios = []
        for e_idx, (ki, hi) in enumerate(edge_list):
            if weights[e_idx] > 1e-12 and k_avg_w.get(ki, 0) > 1e-12 and h_avg_w.get(hi, 0) > 1e-12:
                expected = k_avg_w[ki] * h_avg_w[hi]
                # normalize by harmonic mean or something
                ratios.append(weights[e_idx] / expected)
        if ratios:
            arr = np.array(ratios)
            print(f"  w/(f*g) stats: mean={np.mean(arr):.4f}, std={np.std(arr):.4f}, "
                  f"CV={np.std(arr)/np.mean(arr):.4f}")

        # Analyze right loads: which targets are at lambda_opt (tight)?
        right_load = np.zeros(nh)
        for e_idx, (ki, hi) in enumerate(edge_list):
            right_load[hi] += weights[e_idx]

        tight = [h_smooth[hi] for hi in range(nh) if right_load[hi] > lambda_opt - 1e-6]
        print(f"\n  === Tight right constraints (load = lambda) ===")
        print(f"  {len(tight)} out of {nh} targets are tight ({100*len(tight)/nh:.1f}%)")
        if tight:
            tight_mu = [mu[h] for h in tight]
            nontight_mu = [mu[h] for h in h_smooth if h not in set(tight)]
            print(f"  Tight targets: mu range [{min(tight_mu)}, {max(tight_mu)}], "
                  f"mean={np.mean(tight_mu):.1f}")
            if nontight_mu:
                print(f"  Non-tight:     mu range [{min(nontight_mu)}, {max(nontight_mu)}], "
                      f"mean={np.mean(nontight_mu):.1f}")

        # What is the SPARSITY of optimal weights?
        n_nonzero = sum(1 for w in weights if w > 1e-12)
        print(f"\n  === Weight sparsity ===")
        print(f"  Non-zero weights: {n_nonzero}/{ne_actual} ({100*n_nonzero/ne_actual:.1f}%)")

        # Per-interval: what fraction of edges are used?
        print(f"\n  === Per-interval edge usage ===")
        for idx, (j, lo, hi_int) in enumerate(intervals):
            int_edge_idx = [e_idx for e_idx, (ki, _) in enumerate(edge_list)
                           if lo <= s_plus[ki] <= hi_int]
            if not int_edge_idx:
                continue
            n_used = sum(1 for e_idx in int_edge_idx if weights[e_idx] > 1e-12)
            total = len(int_edge_idx)
            print(f"  I_{j} [{lo},{hi_int}]: {n_used}/{total} edges used "
                  f"({100*n_used/total:.1f}%)")


if __name__ == '__main__':
    n_values_small = [1000, 2000, 5000, 10000, 20000]
    n_values_large = [50000, 100000]
    run_experiment(n_values_small, n_values_large)

    # Deep analysis for small n
    deep_lp_analysis([1000, 5000])
