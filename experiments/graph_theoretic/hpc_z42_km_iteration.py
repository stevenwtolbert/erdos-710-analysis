#!/usr/bin/env python3
"""
Z42: Koukoulopoulos-Maynard Iterative Splitting
Parts A-E: Sequential splitting, simultaneous splitting, deficiency pigeonhole,
density increment analysis, and iterated quality tracking.
"""

import numpy as np
from math import gcd, isqrt, log, sqrt, ceil, floor
from collections import defaultdict
import time
import sys

# ─── Graph construction ───

def compute_graph(n):
    C_TARGET = 2 / np.sqrt(np.e)
    EPS = 0.05
    L = (C_TARGET + EPS) * n * sqrt(log(n) / log(log(n)))
    M = n + L
    B = isqrt(int(M))
    M_int = int(M)

    # Sieve for B-smooth: a number is B-smooth iff all prime factors <= B
    # We'll use a smallest-prime-factor sieve, then check smoothness
    spf = list(range(M_int + 1))  # smallest prime factor
    for p in range(2, isqrt(M_int) + 1):
        if spf[p] == p:  # p is prime
            for multiple in range(p * p, M_int + 1, p):
                if spf[multiple] == multiple:
                    spf[multiple] = p

    # is_smooth[k] = True iff k is B-smooth
    is_smooth = np.zeros(M_int + 1, dtype=bool)
    is_smooth[1] = True
    for k in range(2, M_int + 1):
        if spf[k] > B:
            is_smooth[k] = False
        else:
            # k's smallest prime factor is <= B; check k // spf[k]
            is_smooth[k] = is_smooth[k // spf[k]]
            # But we need to fully factor — actually the above recursion works:
            # if spf[k] <= B and k/spf[k] is B-smooth, then k is B-smooth
            pass

    # S_+ = B-smooth k with B < k <= n//2
    S_plus = [k for k in range(B + 1, n // 2 + 1) if is_smooth[k]]

    # Target range (n, n+L]
    target_lo = n + 1
    target_hi = M_int

    # Adjacency: k -> list of B-smooth targets h in (n, n+L] with k|h
    adj = {}
    H_set = set()

    for k in S_plus:
        targets = []
        m_lo = ceil(target_lo / k)  # smallest m with m*k >= target_lo
        m_hi = floor(target_hi / k)  # largest m with m*k <= target_hi
        for m in range(max(m_lo, 2), m_hi + 1):
            h = m * k
            if target_lo <= h <= target_hi and is_smooth[h]:
                targets.append(h)
                H_set.add(h)
        adj[k] = targets

    H_smooth = sorted(H_set)
    return S_plus, H_smooth, adj, B, L, M, is_smooth


def neighborhood(T, adj):
    """Return NH(T) = union of adj[k] for k in T."""
    result = set()
    for k in T:
        result.update(adj.get(k, []))
    return result


def hall_ratio(T, adj):
    """Return |NH(T)|/|T|."""
    if len(T) == 0:
        return float('inf')
    nh = neighborhood(T, adj)
    return len(nh) / len(T)


# ─── Greedy Hall ratio minimizer ───

def greedy_minimize(S_plus, adj, max_size=None):
    """
    Greedy minimizer: iteratively add elements that minimize |NH(T∪{k})|/|T∪{k}|.
    Then try removing elements to improve.
    """
    if max_size is None:
        max_size = min(len(S_plus), 500)

    T = []
    T_set = set()
    nh_set = set()
    best_ratio = float('inf')

    # Phase 1: greedy addition
    candidates = list(S_plus)
    for iteration in range(min(max_size, len(candidates))):
        best_k = None
        best_new_ratio = float('inf')
        best_new_nh = None

        for k in candidates:
            if k in T_set:
                continue
            new_nh = nh_set | set(adj.get(k, []))
            new_ratio = len(new_nh) / (len(T) + 1)
            if new_ratio < best_new_ratio:
                best_new_ratio = new_ratio
                best_k = k
                best_new_nh = new_nh

        if best_k is None:
            break

        # Only add if ratio doesn't increase too much (or T is small)
        if len(T) >= 3 and best_new_ratio > best_ratio * 1.5:
            break

        T.append(best_k)
        T_set.add(best_k)
        nh_set = best_new_nh
        current_ratio = len(nh_set) / len(T)
        if current_ratio < best_ratio:
            best_ratio = current_ratio

    # Phase 2: try removing elements to improve ratio
    improved = True
    while improved and len(T) > 2:
        improved = False
        current_nh = neighborhood(T, adj)
        current_ratio = len(current_nh) / len(T)

        for i in range(len(T)):
            T_minus = T[:i] + T[i+1:]
            nh_minus = neighborhood(T_minus, adj)
            new_ratio = len(nh_minus) / len(T_minus)
            if new_ratio < current_ratio - 1e-10:
                T = T_minus
                current_ratio = new_ratio
                improved = True
                break

    return T


# ─── Valuation ───

def v_p(k, p):
    """p-adic valuation of k."""
    if k == 0:
        return 0
    v = 0
    while k % p == 0:
        k //= p
        v += 1
    return v


# ─── Part A: Sequential Prime Splitting ───

def part_a(n, S_plus, adj):
    print(f"\n{'='*70}")
    print(f"PART A: Sequential Prime Splitting (n={n})")
    print(f"{'='*70}")

    T = greedy_minimize(S_plus, adj)
    if len(T) <= 1:
        print(f"  Adversarial T has size {len(T)}, skipping.")
        return None

    initial_ratio = hall_ratio(T, adj)
    print(f"  Initial adversarial T: |T|={len(T)}, |NH(T)|={len(neighborhood(T, adj))}, ratio={initial_ratio:.4f}")

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    results = []
    current_T = list(T)

    for p in primes:
        if len(current_T) <= 1:
            print(f"  |T| <= 1 after splitting, stopping.")
            break

        # Split by v_p
        classes = defaultdict(list)
        for k in current_T:
            a = v_p(k, p)
            classes[a].append(k)

        class_data = []
        worst_ratio = float('inf')
        worst_class = None
        best_ratio = 0

        for a in sorted(classes.keys()):
            T_a = classes[a]
            if len(T_a) == 0:
                continue
            r = hall_ratio(T_a, adj)
            class_data.append((a, len(T_a), r))
            if r < worst_ratio:
                worst_ratio = r
                worst_class = T_a
            if r > best_ratio:
                best_ratio = r

        current_ratio = hall_ratio(current_T, adj)

        print(f"  Split by p={p}: |T|={len(current_T)}, T_ratio={current_ratio:.4f}")
        for a, sz, r in class_data:
            marker = " *** WORST" if abs(r - worst_ratio) < 1e-10 else ""
            print(f"    v_{p}={a}: size={sz}, ratio={r:.4f}{marker}")

        step_result = {
            'prime': p,
            'T_size': len(current_T),
            'T_ratio': current_ratio,
            'worst_ratio': worst_ratio,
            'best_ratio': best_ratio,
            'classes': class_data,
            'ratio_increased': worst_ratio > current_ratio
        }
        results.append(step_result)

        # Descend into worst class
        if worst_class is not None and len(worst_class) > 1:
            current_T = worst_class
        else:
            print(f"  Worst class has size <= 1, stopping.")
            break

    # Check: does worst_ratio always increase?
    ratios = [(r['prime'], r['worst_ratio']) for r in results]
    print(f"\n  Ratio progression (worst class at each step):")
    always_increasing = True
    for i, (p, r) in enumerate(ratios):
        inc = ""
        if i > 0 and r < ratios[i-1][1] - 1e-10:
            inc = " DECREASED!"
            always_increasing = False
        elif i > 0:
            inc = " (increased)"
        print(f"    p={p}: worst_ratio={r:.4f}{inc}")
    print(f"  Always increasing: {always_increasing}")

    return results


# ─── Part B: Simultaneous Multi-Prime Splitting ───

def part_b(n, S_plus, adj):
    print(f"\n{'='*70}")
    print(f"PART B: Simultaneous Multi-Prime Splitting (n={n})")
    print(f"{'='*70}")

    T = greedy_minimize(S_plus, adj)
    if len(T) <= 1:
        print(f"  Adversarial T has size {len(T)}, skipping.")
        return None

    initial_ratio = hall_ratio(T, adj)
    print(f"  Initial: |T|={len(T)}, ratio={initial_ratio:.4f}")

    results = {}

    for prime_set_name, prime_set in [("(2,3,5)", [2, 3, 5]), ("(2,3,5,7,11)", [2, 3, 5, 7, 11])]:
        atoms = defaultdict(list)
        for k in T:
            key = tuple(v_p(k, p) for p in prime_set)
            atoms[key].append(k)

        nontrivial = 0
        worst_atom_ratio = float('inf')
        worst_atom_key = None
        atom_sizes = []
        atom_ratios = []

        for key in sorted(atoms.keys()):
            T_atom = atoms[key]
            sz = len(T_atom)
            atom_sizes.append(sz)
            if sz >= 2:
                nontrivial += 1
                r = hall_ratio(T_atom, adj)
                atom_ratios.append(r)
                if r < worst_atom_ratio:
                    worst_atom_ratio = r
                    worst_atom_key = key
            elif sz == 1:
                atom_ratios.append(float('inf'))

        print(f"\n  Split by {prime_set_name}:")
        print(f"    Total atoms: {len(atoms)}, non-trivial (size>=2): {nontrivial}")
        print(f"    Atom sizes: min={min(atom_sizes)}, max={max(atom_sizes)}, median={sorted(atom_sizes)[len(atom_sizes)//2]}")
        if nontrivial > 0:
            print(f"    Worst atom ratio: {worst_atom_ratio:.4f} at key={worst_atom_key}")
            finite_ratios = [r for r in atom_ratios if r < float('inf')]
            if finite_ratios:
                print(f"    Atom ratio range: [{min(finite_ratios):.4f}, {max(finite_ratios):.4f}]")
        else:
            print(f"    All atoms are singletons!")

        results[prime_set_name] = {
            'total_atoms': len(atoms),
            'nontrivial': nontrivial,
            'worst_ratio': worst_atom_ratio if nontrivial > 0 else float('inf'),
            'worst_key': worst_atom_key,
            'atom_sizes': atom_sizes,
        }

    return results


# ─── Part C: Deficiency Pigeonhole Test ───

def part_c(n, S_plus, adj):
    print(f"\n{'='*70}")
    print(f"PART C: Deficiency Pigeonhole Test (n={n})")
    print(f"{'='*70}")

    T = greedy_minimize(S_plus, adj)
    if len(T) <= 2:
        print(f"  Adversarial T has size {len(T)}, skipping.")
        return None

    nh_T = neighborhood(T, adj)
    initial_ratio = len(nh_T) / len(T)
    print(f"  Initial: |T|={len(T)}, |NH(T)|={len(nh_T)}, ratio={initial_ratio:.4f}")

    # Count target multiplicities (how many elements of T hit each target)
    target_mult = defaultdict(int)
    for k in T:
        for h in adj.get(k, []):
            if h in nh_T:
                target_mult[h] += 1

    # Remove target with highest multiplicity (to simulate violation)
    sorted_targets = sorted(nh_T, key=lambda h: -target_mult.get(h, 0))
    removed_target = sorted_targets[0]
    W = nh_T - {removed_target}

    print(f"  Removed target h={removed_target} (mult={target_mult[removed_target]})")
    print(f"  |W| = |NH(T)|-1 = {len(W)}, need |W| < |T|: {len(W)} < {len(T)} -> {len(W) < len(T)}")

    # Now split T by v_2 and check restricted neighborhoods
    restricted_adj = {}
    for k in T:
        restricted_adj[k] = [h for h in adj.get(k, []) if h in W]

    # Split by p=2
    classes = defaultdict(list)
    for k in T:
        a = v_p(k, 2)
        classes[a].append(k)

    violating_classes = 0
    print(f"\n  Splitting by p=2 with restricted targets W:")
    for a in sorted(classes.keys()):
        T_a = classes[a]
        nh_restricted = set()
        for k in T_a:
            nh_restricted.update(restricted_adj[k])
        ratio = len(nh_restricted) / len(T_a) if len(T_a) > 0 else float('inf')
        violates = len(nh_restricted) < len(T_a)
        if violates:
            violating_classes += 1
        print(f"    v_2={a}: |T_a|={len(T_a)}, |NH_W(T_a)|={len(nh_restricted)}, ratio={ratio:.4f}{' VIOLATES!' if violates else ''}")

    # Also test: is there necessarily a violator by pigeonhole?
    # If |W| < |T|, and ∪_a NH_W(T_a) ⊆ W, then Σ|NH_W(T_a)| ...
    # Actually, pigeonhole on the partition: Σ|T_a| = |T| > |W| ≥ |∪ NH_W(T_a)|
    # But NH_W(T_a) can overlap, so we can't directly conclude.
    # The KEY is: if NH_W(T_a) are DISJOINT, then ∃a with |NH_W(T_a)| < |T_a|

    # Check disjointness of restricted neighborhoods
    all_nhs = {}
    for a in sorted(classes.keys()):
        T_a = classes[a]
        nh_restricted = set()
        for k in T_a:
            nh_restricted.update(restricted_adj[k])
        all_nhs[a] = nh_restricted

    # Count pairwise overlaps
    keys = sorted(all_nhs.keys())
    total_overlap = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap = len(all_nhs[keys[i]] & all_nhs[keys[j]])
            total_overlap += overlap

    union_size = len(set().union(*all_nhs.values())) if all_nhs else 0
    sum_sizes = sum(len(s) for s in all_nhs.values())

    print(f"\n  Restricted neighborhood overlaps:")
    print(f"    Σ|NH_W(T_a)| = {sum_sizes}, |∪NH_W(T_a)| = {union_size}")
    print(f"    Total pairwise overlap = {total_overlap}")
    print(f"    Pigeonhole requires disjointness or near-disjointness")
    print(f"    Violating classes found: {violating_classes}")

    # If we remove more targets to force bigger deficiency
    results_by_removal = []
    for num_remove in [1, 2, 3, 5]:
        if num_remove >= len(nh_T):
            break
        W2 = nh_T - set(sorted_targets[:num_remove])
        deficiency = len(T) - len(W2)

        # Split by p=2 with W2
        n_violating = 0
        for a in sorted(classes.keys()):
            T_a = classes[a]
            nh_r = set()
            for k in T_a:
                nh_r.update(h for h in adj.get(k, []) if h in W2)
            if len(nh_r) < len(T_a):
                n_violating += 1

        results_by_removal.append((num_remove, deficiency, n_violating))
        print(f"    Remove {num_remove} targets: deficiency={deficiency}, violating classes={n_violating}")

    return {
        'initial_ratio': initial_ratio,
        'violating_classes': violating_classes,
        'total_overlap': total_overlap,
        'results_by_removal': results_by_removal,
    }


# ─── Part D: Density Increment Analysis ───

def part_d(n, S_plus, adj):
    print(f"\n{'='*70}")
    print(f"PART D: Density Increment Analysis (n={n})")
    print(f"{'='*70}")

    T = greedy_minimize(S_plus, adj)
    if len(T) <= 1:
        print(f"  Adversarial T has size {len(T)}, skipping.")
        return None

    nh_T = neighborhood(T, adj)
    T_ratio = hall_ratio(T, adj)
    print(f"  Initial: |T|={len(T)}, |NH(T)|={len(nh_T)}, ratio={T_ratio:.4f}")

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    results = {}

    for p in primes:
        classes = defaultdict(list)
        for k in T:
            a = v_p(k, p)
            classes[a].append(k)

        if len(classes) <= 1:
            # All elements have same v_p, splitting is trivial
            continue

        # Full neighborhoods
        class_ratios = {}
        class_restricted_sizes = {}

        for a in sorted(classes.keys()):
            T_a = classes[a]
            if len(T_a) == 0:
                continue

            # Full neighborhood
            nh_a = neighborhood(T_a, adj)
            full_ratio = len(nh_a) / len(T_a)

            # Restricted neighborhood (within NH(T))
            nh_a_restricted = nh_a & nh_T
            restricted_ratio = len(nh_a_restricted) / len(T_a)

            class_ratios[a] = (full_ratio, restricted_ratio, len(T_a))
            class_restricted_sizes[a] = len(nh_a_restricted)

        if not class_ratios:
            continue

        worst_full = min(r[0] for r in class_ratios.values())
        worst_restricted = min(r[1] for r in class_ratios.values())
        increment_full = worst_full / T_ratio if T_ratio > 0 else float('inf')
        increment_restricted = worst_restricted / T_ratio if T_ratio > 0 else float('inf')

        # Verify: Σ|T_a| = |T|
        sum_sizes = sum(r[2] for r in class_ratios.values())
        # Verify: ∪NH(T_a) = NH(T) — may not hold if we use full neighborhoods
        union_nh = set()
        for a in classes:
            union_nh.update(neighborhood(classes[a], adj))
        union_eq = (union_nh & nh_T) == nh_T

        # Pigeonhole test: if |NH(T)| < |T|, must ∃a with |NH(T_a)∩NH(T)| < |T_a|?
        # Check: Σ|T_a| = |T|, and |∪(NH(T_a)∩NH(T))| ≤ |NH(T)| < |T|
        # But overlaps prevent direct pigeonhole. Check overlap degree.
        sum_restricted = sum(class_restricted_sizes.get(a, 0) for a in classes)

        print(f"\n  Prime p={p}: {len(class_ratios)} classes")
        for a in sorted(class_ratios.keys()):
            full_r, restr_r, sz = class_ratios[a]
            print(f"    v_{p}={a}: size={sz}, full_ratio={full_r:.4f}, restricted_ratio={restr_r:.4f}")
        print(f"    Increment (full): {increment_full:.4f}, (restricted): {increment_restricted:.4f}")
        print(f"    Σ|T_a|={sum_sizes} = |T|={len(T)}: {sum_sizes == len(T)}")
        print(f"    ∪NH(T_a)∩NH(T) = NH(T): {union_eq}")
        print(f"    Σ|NH_restr(T_a)| = {sum_restricted} vs |NH(T)| = {len(nh_T)} (overlap ratio: {sum_restricted/len(nh_T):.3f})")

        results[p] = {
            'classes': dict(class_ratios),
            'increment_full': increment_full,
            'increment_restricted': increment_restricted,
            'sum_restricted': sum_restricted,
            'nh_T_size': len(nh_T),
        }

    return results


# ─── Part E: Iterated Quality ───

def part_e(n, S_plus, adj):
    print(f"\n{'='*70}")
    print(f"PART E: Iterated Quality (n={n})")
    print(f"{'='*70}")

    T = greedy_minimize(S_plus, adj)
    if len(T) <= 1:
        print(f"  Adversarial T has size {len(T)}, skipping.")
        return None

    def quality(T_sub):
        if len(T_sub) == 0:
            return float('inf'), float('inf'), 0, 0
        nh = neighborhood(T_sub, adj)
        if len(nh) == 0:
            return 0, 0, 0, 0
        sum_inv_k = sum(1.0 / k for k in T_sub)
        sum_inv_h = sum(1.0 / h for h in nh)
        Q = sum_inv_k * len(nh) / len(T_sub)
        ratio_inv = sum_inv_k / sum_inv_h if sum_inv_h > 0 else float('inf')
        return Q, ratio_inv, sum_inv_k, sum_inv_h

    initial_Q, initial_inv_ratio, _, _ = quality(T)
    initial_hall = hall_ratio(T, adj)
    print(f"  Initial: |T|={len(T)}, Hall ratio={initial_hall:.4f}, Q={initial_Q:.4f}, inv_ratio={initial_inv_ratio:.4f}")

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    current_T = list(T)
    quality_trace = [(0, len(current_T), initial_hall, initial_Q, initial_inv_ratio)]

    for step, p in enumerate(primes):
        if len(current_T) <= 1:
            break

        classes = defaultdict(list)
        for k in current_T:
            a = v_p(k, p)
            classes[a].append(k)

        # Find worst class by Hall ratio
        worst_ratio = float('inf')
        worst_class = None
        worst_Q = None
        worst_inv = None

        for a in sorted(classes.keys()):
            T_a = classes[a]
            if len(T_a) == 0:
                continue
            r = hall_ratio(T_a, adj)
            Q_a, inv_a, _, _ = quality(T_a)
            if r < worst_ratio:
                worst_ratio = r
                worst_class = T_a
                worst_Q = Q_a
                worst_inv = inv_a

        if worst_class is None or len(worst_class) <= 0:
            break

        current_T = worst_class
        quality_trace.append((step + 1, len(current_T), worst_ratio, worst_Q, worst_inv))

    print(f"\n  Quality trace through iterations:")
    print(f"  {'Step':>4} {'|T|':>6} {'Hall':>8} {'Q':>10} {'inv_ratio':>10}")
    q_monotone = True
    for i, (step, sz, hr, Q, inv_r) in enumerate(quality_trace):
        q_str = f"{Q:.4f}" if Q < float('inf') else "inf"
        inv_str = f"{inv_r:.4f}" if inv_r < float('inf') else "inf"
        flag = ""
        if i > 0 and Q < quality_trace[i-1][3] - 1e-10:
            flag = " Q_DECREASED"
            q_monotone = False
        print(f"  {step:>4} {sz:>6} {hr:>8.4f} {q_str:>10} {inv_str:>10}{flag}")

    print(f"  Q monotonically increasing: {q_monotone}")

    return quality_trace


# ─── Main ───

def main():
    N_VALUES = [1000, 2000, 5000, 10000, 20000, 50000]

    all_results = {}

    for n in N_VALUES:
        print(f"\n{'#'*70}")
        print(f"# n = {n}")
        print(f"{'#'*70}")

        t0 = time.time()
        S_plus, H_smooth, adj, B, L, M, is_smooth = compute_graph(n)
        t_graph = time.time() - t0
        print(f"  Graph: |S+|={len(S_plus)}, |H_smooth|={len(H_smooth)}, B={B}, L={L:.1f}, time={t_graph:.2f}s")

        # Compute average degree
        if S_plus:
            avg_deg = np.mean([len(adj.get(k, [])) for k in S_plus])
            print(f"  Avg degree: {avg_deg:.2f}")

        results_n = {}

        t0 = time.time()
        results_n['A'] = part_a(n, S_plus, adj)
        print(f"  Part A time: {time.time()-t0:.2f}s")

        t0 = time.time()
        results_n['B'] = part_b(n, S_plus, adj)
        print(f"  Part B time: {time.time()-t0:.2f}s")

        t0 = time.time()
        results_n['C'] = part_c(n, S_plus, adj)
        print(f"  Part C time: {time.time()-t0:.2f}s")

        t0 = time.time()
        results_n['D'] = part_d(n, S_plus, adj)
        print(f"  Part D time: {time.time()-t0:.2f}s")

        t0 = time.time()
        results_n['E'] = part_e(n, S_plus, adj)
        print(f"  Part E time: {time.time()-t0:.2f}s")

        all_results[n] = results_n

    # ─── Write state file ───
    write_state_file(all_results)
    print("\n\nDone. State file written.")


def write_state_file(all_results):
    lines = []
    lines.append("# State 54: Z42 Koukoulopoulos-Maynard Iterative Splitting")
    lines.append("")
    lines.append("## Overview")
    lines.append("Testing the K-M density increment strategy: split adversarial T by p-adic valuations,")
    lines.append("track Hall ratio through iterations, test pigeonhole and quality invariants.")
    lines.append("")

    # Summarize Part A
    lines.append("## Part A: Sequential Prime Splitting")
    lines.append("")
    lines.append("| n | |T_init| | init_ratio | primes_used | final_ratio | always_increasing |")
    lines.append("|---|---------|------------|-------------|-------------|-------------------|")
    for n, res in sorted(all_results.items()):
        A = res.get('A')
        if A is None or len(A) == 0:
            lines.append(f"| {n} | - | - | - | - | - |")
            continue
        init_ratio = A[0]['T_ratio']
        final_ratio = A[-1]['worst_ratio']
        primes_used = len(A)
        always_inc = all(not r.get('ratio_increased', False) for r in A)
        # Actually check ratio progression
        ratios = [r['worst_ratio'] for r in A]
        always_inc_real = all(ratios[i] >= ratios[i-1] - 1e-10 for i in range(1, len(ratios)))
        init_size = A[0]['T_size']
        lines.append(f"| {n} | {init_size} | {init_ratio:.4f} | {primes_used} | {final_ratio:.4f} | {always_inc_real} |")

    lines.append("")

    # Summarize Part B
    lines.append("## Part B: Simultaneous Multi-Prime Splitting")
    lines.append("")
    lines.append("| n | primes | total_atoms | nontrivial | worst_ratio |")
    lines.append("|---|--------|-------------|------------|-------------|")
    for n, res in sorted(all_results.items()):
        B = res.get('B')
        if B is None:
            lines.append(f"| {n} | - | - | - | - |")
            continue
        for pset, data in sorted(B.items()):
            wr = f"{data['worst_ratio']:.4f}" if data['worst_ratio'] < float('inf') else "inf (all singletons)"
            lines.append(f"| {n} | {pset} | {data['total_atoms']} | {data['nontrivial']} | {wr} |")

    lines.append("")

    # Summarize Part C
    lines.append("## Part C: Deficiency Pigeonhole Test")
    lines.append("")
    lines.append("| n | init_ratio | violating_classes (remove 1) | total_overlap |")
    lines.append("|---|------------|------------------------------|---------------|")
    for n, res in sorted(all_results.items()):
        C = res.get('C')
        if C is None:
            lines.append(f"| {n} | - | - | - |")
            continue
        lines.append(f"| {n} | {C['initial_ratio']:.4f} | {C['violating_classes']} | {C['total_overlap']} |")

    lines.append("")

    # Summarize Part D
    lines.append("## Part D: Density Increment Analysis")
    lines.append("")
    lines.append("| n | prime | increment_full | increment_restricted | overlap_ratio |")
    lines.append("|---|-------|----------------|----------------------|---------------|")
    for n, res in sorted(all_results.items()):
        D = res.get('D')
        if D is None:
            continue
        for p in sorted(D.keys()):
            data = D[p]
            olap = data['sum_restricted'] / data['nh_T_size'] if data['nh_T_size'] > 0 else 0
            lines.append(f"| {n} | {p} | {data['increment_full']:.4f} | {data['increment_restricted']:.4f} | {olap:.3f} |")

    lines.append("")

    # Summarize Part E
    lines.append("## Part E: Iterated Quality")
    lines.append("")
    lines.append("| n | steps | init_Q | final_Q | Q_monotone |")
    lines.append("|---|-------|--------|---------|------------|")
    for n, res in sorted(all_results.items()):
        E = res.get('E')
        if E is None:
            lines.append(f"| {n} | - | - | - | - |")
            continue
        steps = len(E)
        init_Q = E[0][3]
        final_Q = E[-1][3]
        q_mono = all(E[i][3] >= E[i-1][3] - 1e-10 for i in range(1, len(E)))
        init_str = f"{init_Q:.4f}" if init_Q < float('inf') else "inf"
        final_str = f"{final_Q:.4f}" if final_Q < float('inf') else "inf"
        lines.append(f"| {n} | {steps} | {init_str} | {final_str} | {q_mono} |")

    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    lines.append("(To be filled after experiment runs)")
    lines.append("")

    state_path = "/home/ashbringer/projects/e710_new_H/states/state_54_z42_km_iteration.md"
    with open(state_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nState file written to {state_path}")


if __name__ == '__main__':
    main()
