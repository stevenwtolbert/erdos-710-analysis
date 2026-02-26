#!/usr/bin/env python3
"""
ERDOS 710 -- EXPERIMENTS Z43a-Z43e: WEIGHTED CAUCHY-SCHWARZ FOR SUBSET HALL

The CS bound with weight function f: S+ -> R+:
  |NH(T)| >= (sum_k f(k) deg(k))^2 / sum_h (sum_{k in T: k|h} f(k))^2
           = (d^T f)^2 / (f^T C f)

where d = degree vector, C = codegree matrix (C_{ij} = |NH(i) cap NH(j)|).
Optimal weight: f = C^{-1} d, giving max CS bound = d^T C^{-1} d.
For Hall: need d^T C^{-1} d >= |T| for all T subseteq S+.

Z43a: Test various weight functions on adversarial T
Z43b: Optimal weights via codegree matrix inverse (adversarial T)
Z43c: Optimal weights on full S+ (small n)
Z43d: Analyze structure of optimal f = C^{-1} d
Z43e: Per-interval weighted CS (within-interval vs global codegrees)
"""

import time
import sys
import argparse
import numpy as np
from math import gcd, log, sqrt, exp, floor, ceil, log2
from collections import Counter, defaultdict

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


def build_greedy_minimizer(elements, targets_dict, s=None):
    """Greedy minimizer: iteratively pick element adding fewest new targets."""
    if s is None:
        s = len(elements)
    s = min(s, len(elements))

    t2p = {}
    for k in elements:
        for h in targets_dict.get(k, set()):
            if h not in t2p:
                t2p[h] = []
            t2p[h].append(k)

    T = []
    NH = set()
    rem = set(elements)
    new_count = {k: len(targets_dict.get(k, set())) for k in elements}

    for step in range(s):
        best_k, best_new = None, float('inf')
        for k in rem:
            nc = new_count[k]
            if nc < best_new or (nc == best_new and (best_k is None or k > best_k)):
                best_new, best_k = nc, k
        if best_k is None:
            break
        T.append(best_k)
        rem.discard(best_k)
        newly_covered = targets_dict.get(best_k, set()) - NH
        NH |= newly_covered
        for h in newly_covered:
            for k2 in t2p.get(h, []):
                if k2 in rem:
                    new_count[k2] -= 1

    return T, NH


def setup_n(n, sf=0.5):
    L = target_L(n)
    if L <= n:
        return None
    M = L - n
    N = n // 2
    if N < 10:
        return None
    delta = 2 * M / n - 1
    nL = n + L
    sqrt_nL = nL ** 0.5
    s = int(sf * N)
    if s < 10:
        return None
    alpha = M / (s + 1)
    pool = sorted(range(int(alpha) + 1, N + 1))
    lpf_cache = {}
    for k in range(1, N + 1):
        lpf_cache[k] = largest_prime_factor(k)
    targets = {}
    for k in pool:
        targets[k] = compute_targets(k, n, L)
    pool_smooth = [k for k in pool if lpf_cache[k] <= sqrt_nL]
    intervals = get_dyadic_intervals(sqrt_nL, N)
    return {
        'n': n, 'L': L, 'M': M, 'N': N, 'delta': delta, 'nL': nL,
        'sqrt_nL': sqrt_nL, 's': s, 'alpha': alpha, 'pool': pool,
        'lpf_cache': lpf_cache, 'targets': targets,
        'pool_smooth': pool_smooth, 'intervals': intervals
    }


def get_interval_data(ctx, jj, ivl_lo, ivl_hi):
    I_j = sorted([k for k in ctx['pool_smooth']
                  if ivl_lo <= k <= ivl_hi and k > int(ctx['sqrt_nL'])])
    ivl_targets = {}
    for k in I_j:
        ivl_targets[k] = ctx['targets'].get(k, set())
    return I_j, ivl_targets


def build_codegree_matrix(T_list, targets_dict):
    """Build codegree matrix C where C[i,j] = |NH(k_i) intersect NH(k_j)|."""
    t = len(T_list)
    C = np.zeros((t, t), dtype=np.float64)

    h_to_indices = defaultdict(list)
    for i, k in enumerate(T_list):
        for h in targets_dict.get(k, set()):
            h_to_indices[h].append(i)

    for h, indices in h_to_indices.items():
        for a in range(len(indices)):
            for b in range(len(indices)):
                C[indices[a], indices[b]] += 1

    return C


def compute_cs_with_weight(T_list, targets_dict, f_vec, deg_vec):
    """
    Correct CS bound: |NH(T)| >= (d^T f)^2 / (f^T C f)
    = (sum_k f(k)*deg(k))^2 / sum_h (sum_{k|h} f(k))^2
    """
    num = np.dot(deg_vec, f_vec) ** 2

    # Denominator = f^T C f = sum_h (sum_{k in T, k|h} f(k))^2
    h_wt = defaultdict(float)
    for i, k in enumerate(T_list):
        for h in targets_dict.get(k, set()):
            h_wt[h] += f_vec[i]

    den = sum(w * w for w in h_wt.values())
    if den == 0:
        return float('inf')
    return num / den


def count_divisors(x):
    """Number of divisors of x."""
    if x <= 1:
        return 1
    cnt = 1
    p = 2
    while p * p <= x:
        e = 0
        while x % p == 0:
            e += 1
            x //= p
        cnt *= (e + 1)
        p += 1
    if x > 1:
        cnt *= 2
    return cnt


def find_adversarial_T(pool, tgts):
    """Find adversarial T via greedy minimizer + worst prefix."""
    T_list, _ = build_greedy_minimizer(pool, tgts)

    NH_prefix = set()
    worst_ratio = float('inf')
    worst_idx = 0
    for i, k in enumerate(T_list):
        NH_prefix |= tgts.get(k, set())
        ratio = len(NH_prefix) / (i + 1)
        if ratio < worst_ratio and i >= 9:
            worst_ratio = ratio
            worst_idx = i

    T_adv = T_list[:worst_idx + 1]

    tgts_adv = {}
    NH_adv = set()
    for k in T_adv:
        tgts_adv[k] = tgts.get(k, set())
        NH_adv |= tgts_adv[k]

    return T_adv, tgts_adv, NH_adv


# ===============================================================
# EXPERIMENT Z43a: TEST VARIOUS WEIGHT FUNCTIONS
# ===============================================================

def experiment_Z43a(n_values, sf=0.5):
    """
    For each n, find the adversarial T via greedy minimizer (GLOBAL, cross-interval).
    Test 7 weight functions.
    CORRECT CS: |NH(T)| >= (d^T f)^2 / (f^T C f).
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z43a: WEIGHTED CS ON ADVERSARIAL SUBSETS")
    print("=" * 78)
    print("  CS bound: |NH(T)| >= (d^T f)^2 / (f^T C f)")
    print("  For Hall: CS bound >= |T|")
    print("  Weight functions:")
    print("  1. f=1 (standard)     2. f=deg(k)    3. f=1/tau(k)")
    print("  4. f=1/sqrt(tau(k))   5. f=deg/tau   6. f=1/sqrt(CC(k))")
    print("  7. f=1/mu_bar(k)")
    print()

    results = []

    for n in n_values:
        t0 = time.time()
        ctx = setup_n(n, sf)
        if ctx is None:
            print(f"  n={n}: setup failed")
            continue

        pool = ctx['pool_smooth']
        tgts = ctx['targets']

        T_adv, tgts_adv, NH_adv = find_adversarial_T(pool, tgts)
        t_size = len(T_adv)
        nh_size = len(NH_adv)

        # Compute per-element properties
        deg = np.array([len(tgts_adv.get(k, set())) for k in T_adv], dtype=np.float64)
        tau = np.array([count_divisors(k) for k in T_adv], dtype=np.float64)

        # Per-target multiplicity
        mu_map = Counter()
        for k in T_adv:
            for h in tgts_adv.get(k, set()):
                mu_map[h] += 1

        # mu_bar(k) = average mu(h) over targets of k
        mu_bar = np.zeros(t_size, dtype=np.float64)
        for i, k in enumerate(T_adv):
            targets_k = tgts_adv.get(k, set())
            if len(targets_k) > 0:
                mu_bar[i] = sum(mu_map[h] for h in targets_k) / len(targets_k)
            else:
                mu_bar[i] = 1.0

        # Cross-codegree CC(k) = sum_h (mu(h)-1) for h in NH(k)
        CC = np.zeros(t_size, dtype=np.float64)
        for i, k in enumerate(T_adv):
            for h in tgts_adv.get(k, set()):
                CC[i] += mu_map[h] - 1

        # Weight functions
        weights = {}
        weights['f=1'] = np.ones(t_size)
        weights['f=deg'] = deg.copy()
        weights['f=1/tau'] = 1.0 / np.maximum(tau, 1)
        weights['f=1/sqrtau'] = 1.0 / np.sqrt(np.maximum(tau, 1))
        weights['f=deg/tau'] = deg / np.maximum(tau, 1)
        weights['f=1/sqrCC'] = 1.0 / np.sqrt(np.maximum(CC, 1))
        weights['f=1/mubar'] = 1.0 / np.maximum(mu_bar, 0.01)

        # Compute CS for each
        cs_vals = {}
        for name, f_vec in weights.items():
            cs_vals[name] = compute_cs_with_weight(T_adv, tgts_adv, f_vec, deg)

        # Standard CS (f=1): should give E1^2/E2
        E1 = np.sum(deg)
        E2_check = sum(v * v for v in mu_map.values())
        std_cs_check = E1 * E1 / E2_check if E2_check > 0 else 0

        elapsed = time.time() - t0

        row = {
            'n': n, '|T|': t_size, '|NH|': nh_size,
            'ratio': nh_size / t_size if t_size > 0 else 0,
            'delta': ctx['delta'],
            'cs': cs_vals, 'time': elapsed,
            'E1': E1, 'std_cs_check': std_cs_check
        }
        results.append(row)

        # Print
        print(f"  n={n:>6}: |T|={t_size}, |NH|={nh_size}, "
              f"|NH|/|T|={nh_size/t_size:.4f}, delta={ctx['delta']:.2f}  ({elapsed:.1f}s)")
        print(f"    E1={E1:.0f}, E1^2/E2={std_cs_check:.2f} (verify f=1: {cs_vals['f=1']:.2f})")
        for name in ['f=1', 'f=deg', 'f=1/tau', 'f=1/sqrtau',
                      'f=deg/tau', 'f=1/sqrCC', 'f=1/mubar']:
            v = cs_vals[name]
            tag = "PASS" if v >= t_size else f"need {v/t_size:.4f}x more"
            print(f"    {name:>14}: CS_bound = {v:.2f}, need >= {t_size}, [{tag}]")

    print()
    return results


# ===============================================================
# EXPERIMENT Z43b: OPTIMAL WEIGHTS VIA C^{-1}
# ===============================================================

def experiment_Z43b(n_values, sf=0.5):
    """
    Build codegree matrix C for adversarial T.
    Correct optimal CS bound: d^T C^{-1} d where d = degree vector.
    For Hall: need d^T C^{-1} d >= |T|.
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z43b: OPTIMAL WEIGHTED CS (ADVERSARIAL T)")
    print("=" * 78)
    print("  Optimal CS bound = d^T C^{-1} d (Rayleigh quotient max)")
    print("  For Hall: need d^T C^{-1} d >= |T|")
    print()

    results = []

    for n in n_values:
        t0 = time.time()
        ctx = setup_n(n, sf)
        if ctx is None:
            print(f"  n={n}: setup failed")
            continue

        pool = ctx['pool_smooth']
        tgts = ctx['targets']

        T_adv, tgts_adv, NH_adv = find_adversarial_T(pool, tgts)
        t_size = len(T_adv)
        nh_size = len(NH_adv)

        if t_size > 3000:
            print(f"  n={n}: |T|={t_size} too large for dense matrix, skipping")
            continue

        deg = np.array([len(tgts_adv.get(k, set())) for k in T_adv], dtype=np.float64)

        # Build codegree matrix
        C = build_codegree_matrix(T_adv, tgts_adv)

        try:
            eigenvalues = np.linalg.eigvalsh(C)
            lam_min = eigenvalues[0]
            lam_max = eigenvalues[-1]
            cond = lam_max / max(lam_min, 1e-15)

            if lam_min < 1e-10:
                C_inv_d = np.linalg.lstsq(C, deg, rcond=None)[0]
            else:
                C_inv_d = np.linalg.solve(C, deg)

            optimal_cs = np.dot(deg, C_inv_d)

            # Verify against actual |NH|
            # CS bound should be <= |NH|
            bound_vs_nh = optimal_cs / nh_size

            # Also compute 1^T C^{-1} 1 for comparison
            if lam_min < 1e-10:
                C_inv_1 = np.linalg.lstsq(C, np.ones(t_size), rcond=None)[0]
            else:
                C_inv_1 = np.linalg.solve(C, np.ones(t_size))
            alt_bound = np.dot(np.ones(t_size), C_inv_1)

            # Standard CS (f=1): E1^2/E2
            E1 = np.sum(deg)
            mu_map = Counter()
            for k in T_adv:
                for h in tgts_adv.get(k, set()):
                    mu_map[h] += 1
            E2 = sum(v * v for v in mu_map.values())
            std_cs = E1 * E1 / E2 if E2 > 0 else 0

            # Optimal f
            f_opt = C_inv_d

            # Correlations
            tau_arr = np.array([count_divisors(k) for k in T_adv], dtype=float)
            corr_deg = np.corrcoef(f_opt, deg)[0, 1] if np.std(f_opt) > 0 else 0
            corr_inv_tau = np.corrcoef(f_opt, 1.0/tau_arr)[0, 1] if np.std(f_opt) > 0 else 0
            corr_inv_k = np.corrcoef(f_opt, 1.0/np.array(T_adv, dtype=float))[0, 1] if np.std(f_opt) > 0 else 0

            elapsed = time.time() - t0

            row = {
                'n': n, '|T|': t_size, '|NH|': nh_size,
                'ratio': nh_size / t_size,
                'optimal_cs': optimal_cs,
                'std_cs': std_cs,
                'alt_bound': alt_bound,
                'hall_need': t_size,
                'pass': optimal_cs >= t_size,
                'cs_per_elt': optimal_cs / t_size,
                'bound_vs_nh': bound_vs_nh,
                'lam_min': lam_min, 'lam_max': lam_max, 'cond': cond,
                'corr_deg': corr_deg, 'corr_inv_tau': corr_inv_tau,
                'corr_inv_k': corr_inv_k,
                'time': elapsed
            }
            results.append(row)

            tag = "PASS" if optimal_cs >= t_size else "FAIL"
            print(f"  n={n:>6}: |T|={t_size}, |NH|={nh_size}")
            print(f"    d^T C^{{-1}} d = {optimal_cs:.2f}, need {t_size}, "
                  f"ratio={optimal_cs/t_size:.4f} [{tag}]")
            print(f"    std_CS(f=1) = {std_cs:.2f}, "
                  f"improvement = {optimal_cs/std_cs:.2f}x")
            print(f"    bound/|NH| = {bound_vs_nh:.4f} "
                  f"(should be <= 1)")
            print(f"    lam: [{lam_min:.4f}, {lam_max:.1f}], cond={cond:.1f}")
            print(f"    corr(f,deg)={corr_deg:.3f}, "
                  f"corr(f,1/tau)={corr_inv_tau:.3f}, "
                  f"corr(f,1/k)={corr_inv_k:.3f}")
            print(f"    ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  n={n}: ERROR: {e} ({elapsed:.1f}s)")

    print()
    return results


# ===============================================================
# EXPERIMENT Z43c: OPTIMAL WEIGHTS ON FULL S+
# ===============================================================

def experiment_Z43c(n_values_small, sf=0.5):
    """
    For small n (|S+| manageable), build codegree matrix for FULL S+.
    Compute d^T C^{-1} d and compare to |S+|.
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z43c: OPTIMAL WEIGHTED CS (FULL S+)")
    print("=" * 78)
    print("  Optimal CS bound = d^T C^{-1} d, need >= |S+|")
    print()

    results = []

    for n in n_values_small:
        t0 = time.time()
        ctx = setup_n(n, sf)
        if ctx is None:
            print(f"  n={n}: setup failed")
            continue

        pool = ctx['pool_smooth']
        tgts = ctx['targets']

        S_plus = pool
        t_size = len(S_plus)

        if t_size > 2500:
            print(f"  n={n}: |S+|={t_size} too large, skipping")
            continue
        if t_size == 0:
            print(f"  n={n}: |S+|=0, skipping")
            continue

        NH_all = set()
        for k in S_plus:
            NH_all |= tgts.get(k, set())
        nh_size = len(NH_all)

        deg = np.array([len(tgts.get(k, set())) for k in S_plus], dtype=np.float64)

        # Build codegree matrix
        C = build_codegree_matrix(S_plus, tgts)

        try:
            eigenvalues = np.linalg.eigvalsh(C)
            lam_min = eigenvalues[0]
            lam_max = eigenvalues[-1]

            if lam_min < 1e-10:
                C_inv_d = np.linalg.lstsq(C, deg, rcond=None)[0]
            else:
                C_inv_d = np.linalg.solve(C, deg)

            optimal_cs = np.dot(deg, C_inv_d)

            # Standard CS
            E1 = np.sum(deg)
            mu_map = Counter()
            for k in S_plus:
                for h in tgts.get(k, set()):
                    mu_map[h] += 1
            E2 = sum(v * v for v in mu_map.values())
            std_cs = E1 * E1 / E2 if E2 > 0 else 0

            elapsed = time.time() - t0

            row = {
                'n': n, '|S+|': t_size, '|NH|': nh_size,
                'ratio': nh_size / t_size,
                'optimal_cs': optimal_cs,
                'std_cs': std_cs,
                'hall_need': t_size,
                'pass': optimal_cs >= t_size,
                'cs_per_elt': optimal_cs / t_size,
                'lam_min': lam_min, 'lam_max': lam_max,
                'time': elapsed
            }
            results.append(row)

            tag = "PASS" if optimal_cs >= t_size else "FAIL"
            print(f"  n={n:>6}: |S+|={t_size}, |NH|={nh_size}")
            print(f"    d^T C^{{-1}} d = {optimal_cs:.2f}, need {t_size}, "
                  f"ratio={optimal_cs/t_size:.4f} [{tag}]")
            print(f"    std_CS(f=1) = {std_cs:.2f}, "
                  f"improvement = {optimal_cs/std_cs:.2f}x")
            print(f"    lam: [{lam_min:.4f}, {lam_max:.1f}]")
            print(f"    ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  n={n}: ERROR: {e} ({elapsed:.1f}s)")

    print()
    return results


# ===============================================================
# EXPERIMENT Z43d: STRUCTURE OF OPTIMAL WEIGHTS
# ===============================================================

def experiment_Z43d(n_values_detail, sf=0.5):
    """
    For a few n values, analyze the optimal f = C^{-1} d in detail.
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z43d: STRUCTURE OF OPTIMAL WEIGHTS f = C^{-1} d")
    print("=" * 78)
    print()

    for n in n_values_detail:
        t0 = time.time()
        ctx = setup_n(n, sf)
        if ctx is None:
            print(f"  n={n}: setup failed")
            continue

        pool = ctx['pool_smooth']
        tgts = ctx['targets']

        T_adv, tgts_adv, NH_adv = find_adversarial_T(pool, tgts)
        t_size = len(T_adv)

        if t_size > 2000:
            print(f"  n={n}: |T|={t_size} too large, skipping")
            continue

        deg = np.array([len(tgts_adv.get(k, set())) for k in T_adv], dtype=np.float64)
        C = build_codegree_matrix(T_adv, tgts_adv)

        try:
            eigenvalues = np.linalg.eigvalsh(C)
            lam_min = eigenvalues[0]

            if lam_min < 1e-10:
                f_opt = np.linalg.lstsq(C, deg, rcond=None)[0]
            else:
                f_opt = np.linalg.solve(C, deg)

            # Element properties
            tau_arr = np.array([count_divisors(k) for k in T_adv], dtype=float)
            k_arr = np.array(T_adv, dtype=float)
            log_k = np.log(k_arr)
            diag_C = np.diag(C)  # = deg

            # Identify interval of each k
            intervals = ctx['intervals']
            interval_of = {}
            for jj, ivl_lo, ivl_hi in intervals:
                for k in T_adv:
                    if ivl_lo <= k <= ivl_hi:
                        interval_of[k] = jj
            j_arr = np.array([interval_of.get(k, -1) for k in T_adv], dtype=float)

            # Correlations
            props = {
                'deg': deg,
                '1/tau': 1.0 / np.maximum(tau_arr, 1),
                '1/k': 1.0 / k_arr,
                'log(k)': log_k,
                'deg/tau': deg / np.maximum(tau_arr, 1),
                'interval_j': j_arr,
            }

            print(f"  n={n}: |T|={t_size}")
            print(f"    f_opt stats: min={np.min(f_opt):.4f}, "
                  f"median={np.median(f_opt):.4f}, max={np.max(f_opt):.4f}")
            neg_frac = np.sum(f_opt < 0) / t_size
            print(f"    Fraction f<0: {neg_frac:.4f}")
            print(f"    Correlations with f_opt = C^{{-1}} d:")
            for name, prop in props.items():
                if np.std(prop) > 1e-15 and np.std(f_opt) > 1e-15:
                    c = np.corrcoef(f_opt, prop)[0, 1]
                    print(f"      corr(f, {name:>10}) = {c:+.4f}")

            # Bin by interval
            unique_j = sorted(set(interval_of.values()))
            print(f"    Per-interval f_opt averages:")
            for jj in unique_j:
                mask = np.array([interval_of.get(k, -1) == jj for k in T_adv])
                if np.sum(mask) > 0:
                    avg_f = np.mean(f_opt[mask])
                    avg_deg = np.mean(deg[mask])
                    cnt = np.sum(mask)
                    print(f"      j={jj}: count={cnt:>4}, "
                          f"avg_f={avg_f:.4f}, avg_deg={avg_deg:.1f}")

            # Check: is f_opt approximately proportional to some simple function?
            # Try f_opt ~ a * deg + b
            A = np.column_stack([deg, np.ones(t_size)])
            coefs, res, _, _ = np.linalg.lstsq(A, f_opt, rcond=None)
            fitted = A @ coefs
            r2 = 1 - np.sum((f_opt - fitted)**2) / np.sum((f_opt - np.mean(f_opt))**2)
            print(f"    Linear fit f ~ {coefs[0]:.4f}*deg + {coefs[1]:.4f}: R^2={r2:.4f}")

            # Try f_opt ~ a * deg + b * log(k) + c
            A2 = np.column_stack([deg, log_k, np.ones(t_size)])
            coefs2, _, _, _ = np.linalg.lstsq(A2, f_opt, rcond=None)
            fitted2 = A2 @ coefs2
            r2_2 = 1 - np.sum((f_opt - fitted2)**2) / np.sum((f_opt - np.mean(f_opt))**2)
            print(f"    Linear fit f ~ {coefs2[0]:.4f}*deg + {coefs2[1]:.4f}*log(k) + {coefs2[2]:.4f}: R^2={r2_2:.4f}")

            elapsed = time.time() - t0
            print(f"    ({elapsed:.1f}s)\n")

        except Exception as e:
            print(f"  n={n}: ERROR: {e}\n")


# ===============================================================
# EXPERIMENT Z43e: PER-INTERVAL WEIGHTED CS
# ===============================================================

def experiment_Z43e(n_values, sf=0.5):
    """
    For each dyadic interval I_j:
    1. Within-interval codegree matrix C_within
    2. Compute d^T C^{-1} d and compare to |I_j|
    3. Also build global-weighted codegree matrix and repeat
    """
    print("\n" + "=" * 78)
    print("  EXPERIMENT Z43e: PER-INTERVAL WEIGHTED CS")
    print("=" * 78)
    print("  d^T C^{-1} d vs |I_j| (within-interval and global-weighted)")
    print()

    results = []

    for n in n_values:
        t0 = time.time()
        ctx = setup_n(n, sf)
        if ctx is None:
            print(f"  n={n}: setup failed")
            continue

        pool_smooth = ctx['pool_smooth']
        tgts = ctx['targets']
        intervals = ctx['intervals']

        print(f"  n={n}, delta={ctx['delta']:.2f}")

        for jj, ivl_lo, ivl_hi in intervals:
            I_j, ivl_targets_dict = get_interval_data(ctx, jj, ivl_lo, ivl_hi)

            if len(I_j) < 5:
                continue
            t_size = len(I_j)

            if t_size > 2000:
                print(f"    j={jj}: |I_j|={t_size} too large, skipping")
                continue

            # Targets reachable from I_j
            tgts_ij = {}
            NH_ij = set()
            for k in I_j:
                t_k = tgts.get(k, set())
                tgts_ij[k] = t_k
                NH_ij |= t_k

            deg = np.array([len(tgts_ij.get(k, set())) for k in I_j], dtype=np.float64)

            # Within-interval codegree matrix
            C_within = build_codegree_matrix(I_j, tgts_ij)

            # Global-weighted codegree: weight each target h by mu_global(h)/mu_within(h)
            mu_within = Counter()
            for k in I_j:
                for h in tgts_ij.get(k, set()):
                    mu_within[h] += 1

            mu_global = Counter()
            for h in NH_ij:
                for k in pool_smooth:
                    if h in tgts.get(k, set()):
                        mu_global[h] += 1

            C_weighted = np.zeros((t_size, t_size), dtype=np.float64)
            h_to_idx = defaultdict(list)
            for i, k in enumerate(I_j):
                for h in tgts_ij.get(k, set()):
                    h_to_idx[h].append(i)

            for h, indices in h_to_idx.items():
                w = mu_global[h] / max(mu_within[h], 1)
                for a in range(len(indices)):
                    for b in range(len(indices)):
                        C_weighted[indices[a], indices[b]] += w

            try:
                # Within-interval optimal
                eig_w = np.linalg.eigvalsh(C_within)
                lam_min_w = eig_w[0]
                if lam_min_w < 1e-10:
                    f_w = np.linalg.lstsq(C_within, deg, rcond=None)[0]
                else:
                    f_w = np.linalg.solve(C_within, deg)
                opt_within = np.dot(deg, f_w)

                # Global-weighted optimal
                eig_g = np.linalg.eigvalsh(C_weighted)
                lam_min_g = eig_g[0]
                if lam_min_g < 1e-10:
                    f_g = np.linalg.lstsq(C_weighted, deg, rcond=None)[0]
                else:
                    f_g = np.linalg.solve(C_weighted, deg)
                opt_global = np.dot(deg, f_g)

                # Standard CS (f=1)
                E1 = np.sum(deg)
                E2 = sum(v * v for v in mu_within.values())
                std_cs = E1 * E1 / (t_size * E2) if E2 > 0 else 0
                # Note: std_cs = E1^2/(t*E2), this is the per-element ratio
                # Actually the CS bound with f=1 gives E1^2/E2 (total bound)
                std_cs_total = E1 * E1 / E2 if E2 > 0 else 0

                avg_ratio = np.mean([mu_global[h]/mu_within[h] for h in NH_ij if mu_within[h] > 0])

                tag_w = "PASS" if opt_within >= t_size else "FAIL"
                tag_g = "PASS" if opt_global >= t_size else "FAIL"

                row = {
                    'n': n, 'j': jj, '|I_j|': t_size, '|NH|': len(NH_ij),
                    'std_cs': std_cs_total,
                    'opt_within': opt_within,
                    'opt_global': opt_global,
                    'ratio_w': opt_within / t_size,
                    'ratio_g': opt_global / t_size,
                    'avg_mu_ratio': avg_ratio
                }
                results.append(row)

                print(f"    j={jj}: |I_j|={t_size:>4}, |NH|={len(NH_ij):>5}, "
                      f"std(f=1)={std_cs_total:.1f}, "
                      f"opt_w={opt_within:.1f}/{t_size} ({opt_within/t_size:.3f}) [{tag_w}], "
                      f"opt_g={opt_global:.1f}/{t_size} ({opt_global/t_size:.3f}) [{tag_g}], "
                      f"mu_g/mu_w={avg_ratio:.2f}")

            except Exception as e:
                print(f"    j={jj}: ERROR: {e}")

        elapsed = time.time() - t0
        print(f"    Total: {elapsed:.1f}s\n")

    return results


# ===============================================================
# MAIN
# ===============================================================

def main():
    parser = argparse.ArgumentParser(description="Z43: Weighted CS experiments")
    parser.add_argument('--parts', default='abcde', help='Which parts to run')
    parser.add_argument('--sf', type=float, default=0.5, help='Size fraction')
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 78)
    print("  ERDOS 710 -- Z43: WEIGHTED CAUCHY-SCHWARZ FOR SUBSET HALL")
    print("  CS bound: |NH(T)| >= (d^T f)^2 / (f^T C f)")
    print("  Optimal: d^T C^{-1} d.  For Hall: need >= |T|.")
    print("=" * 78)

    n_main = [1000, 2000, 5000, 10000, 20000, 50000]
    n_small = [1000, 2000, 5000]
    n_detail = [2000, 5000, 10000]
    n_interval = [1000, 2000, 5000, 10000]

    all_results = {}

    if 'a' in args.parts:
        all_results['Z43a'] = experiment_Z43a(n_main, args.sf)

    if 'b' in args.parts:
        all_results['Z43b'] = experiment_Z43b(n_main, args.sf)

    if 'c' in args.parts:
        all_results['Z43c'] = experiment_Z43c(n_small, args.sf)

    if 'd' in args.parts:
        experiment_Z43d(n_detail, args.sf)

    if 'e' in args.parts:
        all_results['Z43e'] = experiment_Z43e(n_interval, args.sf)

    # ====== SUMMARY ======
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)

    if 'Z43a' in all_results:
        print("\n  Z43a (Weight Functions on Adversarial T):")
        print(f"  {'n':>6} {'|T|':>5} {'|NH|':>5} {'f=1':>8} {'f=deg':>8} "
              f"{'f=1/tau':>8} {'f=1/sqrCC':>10} {'f=1/mubar':>10}")
        for row in all_results['Z43a']:
            cs = row['cs']
            print(f"  {row['n']:>6} {row['|T|']:>5} {row['|NH|']:>5} "
                  f"{cs['f=1']:>8.2f} {cs['f=deg']:>8.2f} "
                  f"{cs['f=1/tau']:>8.2f} {cs['f=1/sqrCC']:>10.2f} "
                  f"{cs['f=1/mubar']:>10.2f}")

    if 'Z43b' in all_results:
        print("\n  Z43b (Optimal CS = d^T C^{-1} d on Adversarial T):")
        for row in all_results['Z43b']:
            tag = "PASS" if row['pass'] else "FAIL"
            print(f"    n={row['n']:>6}: opt={row['optimal_cs']:.2f}, "
                  f"std={row['std_cs']:.2f}, "
                  f"need {row['hall_need']}, "
                  f"ratio={row['cs_per_elt']:.4f} [{tag}], "
                  f"improv={row['optimal_cs']/row['std_cs']:.2f}x")

    if 'Z43c' in all_results:
        print("\n  Z43c (Optimal CS on Full S+):")
        for row in all_results['Z43c']:
            tag = "PASS" if row['pass'] else "FAIL"
            print(f"    n={row['n']:>6}: opt={row['optimal_cs']:.2f}, "
                  f"std={row['std_cs']:.2f}, "
                  f"need {row['hall_need']}, "
                  f"ratio={row['cs_per_elt']:.4f} [{tag}]")

    if 'Z43e' in all_results:
        print("\n  Z43e (Per-Interval Weighted CS):")
        pass_w = sum(1 for r in all_results['Z43e'] if r['ratio_w'] >= 1)
        fail_w = sum(1 for r in all_results['Z43e'] if r['ratio_w'] < 1)
        pass_g = sum(1 for r in all_results['Z43e'] if r['ratio_g'] >= 1)
        fail_g = sum(1 for r in all_results['Z43e'] if r['ratio_g'] < 1)
        total = len(all_results['Z43e'])
        print(f"    Within-interval: {pass_w}/{total} PASS, {fail_w}/{total} FAIL")
        print(f"    Global-weighted: {pass_g}/{total} PASS, {fail_g}/{total} FAIL")
        if fail_w > 0:
            worst = min(all_results['Z43e'], key=lambda r: r['ratio_w'])
            print(f"    Worst within: n={worst['n']}, j={worst['j']}, "
                  f"ratio={worst['ratio_w']:.4f}")
        if fail_g > 0:
            worst = min(all_results['Z43e'], key=lambda r: r['ratio_g'])
            print(f"    Worst global: n={worst['n']}, j={worst['j']}, "
                  f"ratio={worst['ratio_g']:.4f}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 78)


if __name__ == '__main__':
    main()
