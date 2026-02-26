#!/usr/bin/env python3
"""
ERDOS 710 -- EXPERIMENT Z41: DESCENT CHAIN STRUCTURE

The insight: If T is "self-covering" (every k in T has NH(k) subset of NH(T minus k)),
then for each target h = mk of k, there must exist k' in T with k'|h and k' != k.
The simplest such k' is k/p for some prime p dividing k, which lies in a LOWER interval.
This creates "descent chains" from high intervals to low intervals, where expansion grows.

Parts:
  A: Find near-tight subsets T of S_+ using multiple strategies
  B: Descent chain analysis for elements of adversarial T
  C: Self-covering constraint analysis
  D: Interval descent depth
  E: "Forced expansion" calculation
  F: Can a self-covering T violate Hall?
"""

import time
import random
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


def prime_divisors(x):
    """Return sorted list of distinct prime factors of x."""
    primes = []
    p = 2
    while p * p <= x:
        if x % p == 0:
            primes.append(p)
            while x % p == 0:
                x //= p
        p += 1
    if x > 1:
        primes.append(x)
    return primes


def compute_targets(k, n, L):
    j0 = (2 * n) // k + 1
    j1 = (n + L) // k
    return set(k * j for j in range(j0, j1 + 1))


def get_dyadic_intervals(B, N):
    lo = B + 1
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


def setup_n(n):
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
    B = int(sqrt_nL)

    pool_smooth = []
    for k in range(B + 1, N + 1):
        lpf = largest_prime_factor(k)
        if lpf <= B:
            pool_smooth.append(k)

    targets = {}
    H_all = set()
    for k in pool_smooth:
        tgts = compute_targets(k, n, L)
        targets[k] = tgts
        H_all |= tgts

    intervals = get_dyadic_intervals(B, N)

    k_to_interval = {}
    for jj, ivl_lo, ivl_hi in intervals:
        for k in pool_smooth:
            if ivl_lo <= k <= ivl_hi:
                k_to_interval[k] = jj

    pool_set = set(pool_smooth)

    h_to_divisors = defaultdict(list)
    for k in pool_smooth:
        for h in targets[k]:
            h_to_divisors[h].append(k)
    # Sort each list for efficient smallest-other lookups
    for h in h_to_divisors:
        h_to_divisors[h].sort()

    # Build t2p (same as h_to_divisors but as defaultdict(list))
    t2p = defaultdict(list)
    for k in pool_smooth:
        for h in targets[k]:
            t2p[h].append(k)

    return {
        'n': n, 'L': L, 'M': M, 'N': N, 'delta': delta, 'nL': nL,
        'sqrt_nL': sqrt_nL, 'B': B,
        'pool_smooth': pool_smooth, 'pool_set': pool_set,
        'targets': targets, 'H_all': H_all,
        'intervals': intervals, 'k_to_interval': k_to_interval,
        'h_to_divisors': h_to_divisors, 't2p': t2p
    }


# ================================================================
#  EFFICIENT PRIVATE TARGET COMPUTATION
# ================================================================

def compute_private_counts(T, targets):
    """
    Efficiently compute private target count for each k in T.
    A target h is private to k if mu_T(h) == 1, where mu_T(h) = |{k' in T : h in targets(k')}|.
    Returns list of (k, private_count).
    """
    T_set = set(T)
    # Count mu_T(h) for each target h
    mu_T = Counter()
    for k in T:
        for h in targets.get(k, set()):
            mu_T[h] += 1

    # For each k, count targets with mu_T(h) == 1
    result = []
    for k in T:
        private = sum(1 for h in targets.get(k, set()) if mu_T[h] == 1)
        result.append((k, private))
    return result


# ================================================================
#  GREEDY RATIO MINIMIZER
# ================================================================

def greedy_minimize_ratio(pool, targets, max_size, t2p):
    """
    Greedy: build T adding elements that minimize |NH(T)|/|T|.
    Try multiple seeds, track minimum ratio snapshot.
    """
    best_T = None
    best_ratio = float('inf')
    best_NH = None

    pool_set = set(pool)

    degs = [(len(targets.get(k, set())), k) for k in pool]
    degs.sort()
    seeds = [k for _, k in degs[:30]]

    random.seed(42)
    if len(pool) > 100:
        seeds += random.sample(pool, min(70, len(pool)))
    else:
        seeds += list(pool)
    seeds = list(dict.fromkeys(seeds))

    for seed in seeds:
        T_curr = [seed]
        NH_curr = set(targets.get(seed, set()))
        rem = pool_set - {seed}
        new_count = {}
        for k in rem:
            new_count[k] = len(targets.get(k, set()) - NH_curr)

        ratio_here = len(NH_curr) / 1.0
        if ratio_here < best_ratio:
            best_ratio = ratio_here
            best_T = list(T_curr)
            best_NH = set(NH_curr)

        for step in range(1, max_size):
            if not rem:
                break
            best_k = None
            best_new = float('inf')
            for k in rem:
                nc = new_count.get(k, 0)
                if nc < best_new or (nc == best_new and (best_k is None or k > best_k)):
                    best_new = nc
                    best_k = k
            if best_k is None:
                break

            T_curr.append(best_k)
            rem.discard(best_k)
            newly_covered = targets.get(best_k, set()) - NH_curr
            NH_curr |= newly_covered

            for h in newly_covered:
                for k2 in t2p[h]:
                    if k2 in rem:
                        new_count[k2] = new_count.get(k2, 0) - 1

            ratio_here = len(NH_curr) / len(T_curr)
            if ratio_here < best_ratio:
                best_ratio = ratio_here
                best_T = list(T_curr)
                best_NH = set(NH_curr)

    return best_T, best_NH, best_ratio


# ================================================================
#  PART A: FIND NEAR-TIGHT SUBSETS
# ================================================================

def part_A(ctx):
    print("\n" + "=" * 72)
    print("  PART A: FIND NEAR-TIGHT SUBSETS T of S_+")
    print("=" * 72)

    n = ctx['n']
    pool = ctx['pool_smooth']
    targets = ctx['targets']
    h_to_div = ctx['h_to_divisors']
    k_to_ivl = ctx['k_to_interval']
    intervals = ctx['intervals']
    t2p = ctx['t2p']
    pool_set = ctx['pool_set']
    B = ctx['B']

    # --- Strategy 1: Greedy ratio minimizer ---
    max_sz = min(500, len(pool) // 2)
    if max_sz < 5:
        max_sz = len(pool)
    t0 = time.time()
    T1, NH1, ratio1 = greedy_minimize_ratio(pool, targets, max_sz, t2p)
    T1_set = set(T1)
    el1 = time.time() - t0
    print(f"\n  Strategy 1 (greedy minimizer): {el1:.1f}s")
    print(f"    |T| = {len(T1)}, |NH(T)| = {len(NH1)}, ratio = {ratio1:.4f}")

    # --- Strategy 2: Min-degree seeded, grow by shared targets ---
    degs = [(len(targets.get(k, set())), k) for k in pool]
    degs.sort()
    seed2 = degs[0][1]
    T2 = [seed2]
    T2_set = {seed2}
    NH2 = set(targets.get(seed2, set()))
    for step in range(min(499, len(pool) - 1)):
        best_k, best_shared = None, -1
        for k in pool:
            if k in T2_set:
                continue
            shared = len(targets.get(k, set()) & NH2)
            if shared > best_shared or (shared == best_shared and (best_k is None or k > best_k)):
                best_shared = shared
                best_k = k
        if best_k is None:
            break
        T2.append(best_k)
        T2_set.add(best_k)
        NH2 |= targets.get(best_k, set())
        if len(NH2) / len(T2) <= ratio1 * 1.1:
            pass
    # Trim to best ratio prefix
    NH2_curr = set()
    best_T2, best_NH2, best_ratio2 = None, None, float('inf')
    for i, k in enumerate(T2):
        NH2_curr |= targets.get(k, set())
        r = len(NH2_curr) / (i + 1)
        if r < best_ratio2:
            best_ratio2 = r
            best_T2 = T2[:i + 1]
            best_NH2 = set(NH2_curr)
    T2, NH2, ratio2 = best_T2, best_NH2, best_ratio2
    T2_set = set(T2)
    print(f"\n  Strategy 2 (min-deg + shared targets):")
    print(f"    |T| = {len(T2)}, |NH(T)| = {len(NH2)}, ratio = {ratio2:.4f}")

    # --- Strategy 3: Chain construction ---
    best_T3, best_NH3, best_ratio3 = None, None, float('inf')

    candidates = sorted([k for k in pool if k > ctx['N'] * 0.6], reverse=True)[:200]
    if not candidates:
        candidates = pool[-20:]

    for start_k in candidates:
        chain = [start_k]
        chain_set = {start_k}
        queue = [start_k]
        while queue and len(chain) < 500:
            curr = queue.pop(0)
            for p in prime_divisors(curr):
                child = curr // p
                if child in pool_set and child not in chain_set:
                    chain.append(child)
                    chain_set.add(child)
                    queue.append(child)
        NH3_curr = set()
        best_prefix, best_nh_prefix, best_r_prefix = None, None, float('inf')
        for i, k in enumerate(chain):
            NH3_curr |= targets.get(k, set())
            r = len(NH3_curr) / (i + 1)
            if r < best_r_prefix:
                best_r_prefix = r
                best_prefix = chain[:i + 1]
                best_nh_prefix = set(NH3_curr)
        if best_r_prefix < best_ratio3:
            best_ratio3 = best_r_prefix
            best_T3 = best_prefix
            best_NH3 = best_nh_prefix

    T3, NH3, ratio3 = best_T3, best_NH3, best_ratio3
    print(f"\n  Strategy 3 (descent chain construction):")
    print(f"    |T| = {len(T3)}, |NH(T)| = {len(NH3)}, ratio = {ratio3:.4f}")

    # Pick the best T overall
    all_strats = [(ratio1, T1, NH1, "greedy"), (ratio2, T2, NH2, "shared"),
                  (ratio3, T3, NH3, "chain")]
    all_strats.sort(key=lambda x: x[0])
    best_ratio, best_T, best_NH, best_name = all_strats[0]
    best_T_set = set(best_T)

    print(f"\n  >>> Best strategy: {best_name}")
    print(f"  >>> |T| = {len(best_T)}, |NH(T)| = {len(best_NH)}, ratio = {best_ratio:.4f}")
    print(f"  >>> Hall margin = {len(best_NH) - len(best_T)}")

    # Private target analysis (EFFICIENT: uses mu_T counting)
    private_counts = compute_private_counts(best_T, targets)

    zero_private = sum(1 for _, c in private_counts if c == 0)
    print(f"\n  Private target analysis:")
    print(f"    Elements with 0 private targets: {zero_private}/{len(best_T)} "
          f"({zero_private / len(best_T) * 100:.1f}%)")
    if private_counts:
        pvals = [c for _, c in private_counts]
        print(f"    Private count stats: min={min(pvals)}, max={max(pvals)}, "
              f"avg={sum(pvals)/len(pvals):.2f}")

    # Interval distribution
    ivl_counts = Counter()
    for k in best_T:
        j = k_to_ivl.get(k, -1)
        ivl_counts[j] += 1
    print(f"\n  Interval distribution of best T:")
    for jj, ivl_lo, ivl_hi in intervals:
        cnt = ivl_counts.get(jj, 0)
        total_in_ivl = sum(1 for k in pool if k_to_ivl.get(k) == jj)
        if cnt > 0 or total_in_ivl > 0:
            print(f"    I_{jj} [{ivl_lo}, {ivl_hi}]: {cnt}/{total_in_ivl} "
                  f"({cnt / max(len(best_T), 1) * 100:.1f}% of T)")

    results = {
        'best_T': best_T, 'best_NH': best_NH, 'best_ratio': best_ratio,
        'best_name': best_name, 'best_T_set': best_T_set,
        'private_counts': private_counts, 'zero_private': zero_private,
        'ratios': {'greedy': ratio1, 'shared': ratio2, 'chain': ratio3},
    }
    return results


# ================================================================
#  PART B: DESCENT CHAIN ANALYSIS
# ================================================================

def part_B(ctx, partA):
    print("\n" + "=" * 72)
    print("  PART B: DESCENT CHAIN ANALYSIS")
    print("=" * 72)

    n = ctx['n']
    T = partA['best_T']
    T_set = partA['best_T_set']
    targets = ctx['targets']
    pool_set = ctx['pool_set']
    h_to_div = ctx['h_to_divisors']
    k_to_ivl = ctx['k_to_interval']

    descent_in_T = 0
    descent_in_pool = 0
    total_prime_slots = 0

    T_sample = T if len(T) <= 200 else T[:200]

    for k in T_sample:
        primes_k = prime_divisors(k)
        for p in primes_k:
            total_prime_slots += 1
            child = k // p
            if child in T_set:
                descent_in_T += 1
            if child in pool_set:
                descent_in_pool += 1

    print(f"\n  Descent link statistics (sample of {len(T_sample)} elements):")
    print(f"    Total prime-descent slots: {total_prime_slots}")
    print(f"    k/p in T: {descent_in_T} ({descent_in_T / max(total_prime_slots, 1) * 100:.1f}%)")
    print(f"    k/p in S_+: {descent_in_pool} ({descent_in_pool / max(total_prime_slots, 1) * 100:.1f}%)")

    # For each target h of k: smallest k' in S_+ (besides k) dividing h, is k' in T?
    targets_covered_by_T = 0
    targets_covered_by_pool = 0
    total_target_slots = 0
    smallest_divisor_in_lower = 0
    smallest_divisor_in_same = 0

    for k in T_sample:
        k_ivl = k_to_ivl.get(k, -1)
        for h in targets.get(k, set()):
            total_target_slots += 1
            divs = h_to_div[h]  # already sorted
            smallest_other = None
            for d in divs:
                if d != k:
                    smallest_other = d
                    break
            if smallest_other is not None:
                targets_covered_by_pool += 1
                if smallest_other in T_set:
                    targets_covered_by_T += 1
                other_ivl = k_to_ivl.get(smallest_other, -1)
                if other_ivl < k_ivl:
                    smallest_divisor_in_lower += 1
                elif other_ivl == k_ivl:
                    smallest_divisor_in_same += 1

    print(f"\n  Target covering analysis (sample):")
    print(f"    Total target slots: {total_target_slots}")
    print(f"    Targets with another divisor in S_+: {targets_covered_by_pool} "
          f"({targets_covered_by_pool / max(total_target_slots, 1) * 100:.1f}%)")
    print(f"    Targets with another divisor in T: {targets_covered_by_T} "
          f"({targets_covered_by_T / max(total_target_slots, 1) * 100:.1f}%)")
    print(f"    Smallest other divisor in LOWER interval: {smallest_divisor_in_lower} "
          f"({smallest_divisor_in_lower / max(total_target_slots, 1) * 100:.1f}%)")
    print(f"    Smallest other divisor in SAME interval: {smallest_divisor_in_same} "
          f"({smallest_divisor_in_same / max(total_target_slots, 1) * 100:.1f}%)")

    # Required support set for top-degree elements
    print(f"\n  Required support set analysis (top 10 elements by degree):")
    degs_in_T = [(len(targets.get(k, set())), k) for k in T_sample]
    degs_in_T.sort(reverse=True)
    for deg, k in degs_in_T[:10]:
        k_tgts = targets.get(k, set())
        k_ivl = k_to_ivl.get(k, -1)
        support = set()
        for h in k_tgts:
            for d in h_to_div[h]:
                if d != k and d in pool_set:
                    support.add(d)
                    break
        support_in_T = support & T_set
        support_lower = sum(1 for s in support if k_to_ivl.get(s, -1) < k_ivl)
        print(f"    k={k} (deg={deg}, ivl={k_ivl}): "
              f"|support|={len(support)}, in_T={len(support_in_T)}, "
              f"in_lower_ivl={support_lower}")

    return {
        'descent_in_T': descent_in_T, 'descent_in_pool': descent_in_pool,
        'total_prime_slots': total_prime_slots,
        'targets_covered_by_T': targets_covered_by_T,
        'targets_covered_by_pool': targets_covered_by_pool,
        'total_target_slots': total_target_slots,
        'smallest_in_lower': smallest_divisor_in_lower,
        'smallest_in_same': smallest_divisor_in_same
    }


# ================================================================
#  PART C: SELF-COVERING CONSTRAINT (EFFICIENT)
# ================================================================

def part_C(ctx, partA):
    print("\n" + "=" * 72)
    print("  PART C: SELF-COVERING CONSTRAINT ANALYSIS")
    print("=" * 72)

    n = ctx['n']
    T = partA['best_T']
    T_set = partA['best_T_set']
    targets = ctx['targets']
    pool_set = ctx['pool_set']
    h_to_div = ctx['h_to_divisors']
    k_to_ivl = ctx['k_to_interval']

    # Efficient: compute mu_T(h) = number of T elements hitting h
    mu_T = Counter()
    for k in T:
        for h in targets.get(k, set()):
            mu_T[h] += 1

    # k has zero private targets iff all its targets h have mu_T(h) >= 2
    private_counts = partA['private_counts']
    priv_dict = dict(private_counts)
    T_covered = [k for k in T if priv_dict.get(k, 1) == 0]

    print(f"\n  |T| = {len(T)}, |T_covered| (zero private targets) = {len(T_covered)} "
          f"({len(T_covered) / max(len(T), 1) * 100:.1f}%)")

    if not T_covered:
        print("  No self-covering elements found. Using elements with lowest private count.")
        sorted_priv = sorted(private_counts, key=lambda x: x[1])
        T_covered = [k for k, _ in sorted_priv[:min(20, len(sorted_priv))]]
        print(f"  Using {len(T_covered)} elements with lowest private counts instead.")

    # For each k in T_covered: find covering elements
    # A covering element for target h of k is k' in T, k' != k, with h in targets(k')
    covering_in_T_count = 0
    covering_outside_T_count = 0
    covering_in_lower_count = 0
    covering_in_same_count = 0

    T_covered_sample = T_covered[:min(100, len(T_covered))]

    for k in T_covered_sample:
        k_tgts = targets.get(k, set())
        k_ivl = k_to_ivl.get(k, -1)

        S_k = set()
        for h in k_tgts:
            # Find some k' in T (k' != k) covering h
            covered_by = None
            for d in h_to_div[h]:
                if d != k and d in T_set:
                    covered_by = d
                    break
            if covered_by is not None:
                S_k.add(covered_by)
            else:
                # Look in pool
                for d in h_to_div[h]:
                    if d != k:
                        covered_by = d
                        break
                if covered_by is not None:
                    S_k.add(covered_by)

        sk_in_T = S_k & T_set
        sk_outside_T = S_k - T_set
        sk_lower = sum(1 for s in S_k if k_to_ivl.get(s, -1) < k_ivl)
        sk_same = sum(1 for s in S_k if k_to_ivl.get(s, -1) == k_ivl)

        covering_in_T_count += len(sk_in_T)
        covering_outside_T_count += len(sk_outside_T)
        covering_in_lower_count += sk_lower
        covering_in_same_count += sk_same

    total_covering = covering_in_T_count + covering_outside_T_count
    print(f"\n  Covering element analysis (sample of {len(T_covered_sample)}):")
    print(f"    Total covering elements found: {total_covering}")
    print(f"    Covering elements IN T: {covering_in_T_count} "
          f"({covering_in_T_count / max(total_covering, 1) * 100:.1f}%)")
    print(f"    Covering elements OUTSIDE T: {covering_outside_T_count} "
          f"({covering_outside_T_count / max(total_covering, 1) * 100:.1f}%)")
    print(f"    Covering elements in LOWER intervals: {covering_in_lower_count} "
          f"({covering_in_lower_count / max(total_covering, 1) * 100:.1f}%)")
    print(f"    Covering elements in SAME interval: {covering_in_same_count} "
          f"({covering_in_same_count / max(total_covering, 1) * 100:.1f}%)")

    # Self-covering consistency check
    print(f"\n  Self-covering consistency check:")
    inconsistent = 0
    for k in T_covered_sample:
        if priv_dict.get(k, 1) == 0:
            for h in targets.get(k, set()):
                if mu_T[h] < 2:
                    # h is hit only by k in T => k has a private target => contradiction
                    inconsistent += 1
                    break
    if inconsistent > 0:
        print(f"    WARNING: {inconsistent} elements have 0 private but mu_T(h)=1 targets!")
    else:
        print(f"    All zero-private elements have targets fully covered by T. Consistent.")

    return {
        'n_covered': len(T_covered),
        'covering_in_T': covering_in_T_count,
        'covering_outside_T': covering_outside_T_count,
        'covering_in_lower': covering_in_lower_count,
        'covering_in_same': covering_in_same_count
    }


# ================================================================
#  PART D: INTERVAL DESCENT DEPTH
# ================================================================

def part_D(ctx, partA):
    print("\n" + "=" * 72)
    print("  PART D: INTERVAL DESCENT DEPTH")
    print("=" * 72)

    n = ctx['n']
    T = partA['best_T']
    T_set = partA['best_T_set']
    targets = ctx['targets']
    pool_set = ctx['pool_set']
    k_to_ivl = ctx['k_to_interval']
    intervals = ctx['intervals']

    # Build descent graph in T: edges k -> k/p if k/p in T
    edges = defaultdict(list)
    for k in T:
        for p in prime_divisors(k):
            child = k // p
            if child in T_set:
                edges[k].append(child)

    # Longest path via DFS with memoization
    memo = {}

    def longest_path(v, visited=None):
        if visited is None:
            visited = set()
        if v in memo:
            return memo[v]
        visited.add(v)
        best = 0
        best_path = [v]
        for child in edges.get(v, []):
            if child not in visited:
                clen, cpath = longest_path(child, visited)
                if clen + 1 > best:
                    best = clen + 1
                    best_path = [v] + cpath
        visited.discard(v)
        memo[v] = (best, best_path)
        return best, best_path

    max_depth = 0
    max_path = []
    for k in T:
        d, p = longest_path(k)
        if d > max_depth:
            max_depth = d
            max_path = p

    n_edges_T = sum(len(v) for v in edges.values())
    print(f"\n  Descent graph in T: {n_edges_T} edges among {len(T)} nodes")
    print(f"  Longest descent path: depth = {max_depth}")
    if max_path:
        path_str = " -> ".join([f"{k}(I_{k_to_ivl.get(k, '?')})" for k in max_path])
        if len(path_str) > 200:
            path_str = path_str[:200] + "..."
        print(f"    Path: {path_str}")

        path_ivls = [k_to_ivl.get(k, -1) for k in max_path]
        print(f"    Intervals along path: {path_ivls}")
        top_ivl = max(path_ivls)
        bottom_ivl = min(path_ivls)
        print(f"    Top interval: I_{top_ivl}, Bottom interval: I_{bottom_ivl}")

        for jj, ivl_lo, ivl_hi in intervals:
            if jj == bottom_ivl:
                I_j = [k for k in ctx['pool_smooth'] if ivl_lo <= k <= ivl_hi]
                if I_j:
                    degs = [len(targets.get(k, set())) for k in I_j]
                    alpha_j = sum(degs) / len(I_j)
                    print(f"    Bottom interval I_{jj}: |I_j|={len(I_j)}, alpha_j={alpha_j:.2f}")
            if jj == top_ivl:
                I_j_top = [k for k in ctx['pool_smooth'] if ivl_lo <= k <= ivl_hi]
                if I_j_top:
                    degs_top = [len(targets.get(k, set())) for k in I_j_top]
                    alpha_top = sum(degs_top) / len(I_j_top)
                    print(f"    Top interval I_{jj}: |I_j|={len(I_j_top)}, alpha_j={alpha_top:.2f}")

    # Broader descent in S_+
    print(f"\n  Broader descent (in S_+, not just T):")
    edges_pool = defaultdict(list)
    for k in ctx['pool_smooth']:
        for p in prime_divisors(k):
            child = k // p
            if child in pool_set:
                edges_pool[k].append(child)

    memo2 = {}

    def longest_path_pool(v, depth_limit=50, visited=None):
        if visited is None:
            visited = set()
        if v in memo2:
            return memo2[v]
        if depth_limit == 0:
            memo2[v] = (0, [v])
            return 0, [v]
        visited.add(v)
        best = 0
        best_path = [v]
        for child in edges_pool.get(v, []):
            if child not in visited:
                clen, cpath = longest_path_pool(child, depth_limit - 1, visited)
                if clen + 1 > best:
                    best = clen + 1
                    best_path = [v] + cpath
        visited.discard(v)
        memo2[v] = (best, best_path)
        return best, best_path

    max_depth_pool = 0
    max_path_pool = []
    for k in ctx['pool_smooth']:
        d, p = longest_path_pool(k)
        if d > max_depth_pool:
            max_depth_pool = d
            max_path_pool = p

    print(f"  Longest descent path in S_+: depth = {max_depth_pool}")
    if max_path_pool:
        path_ivls = [k_to_ivl.get(k, -1) for k in max_path_pool]
        path_str = " -> ".join([f"{k}(I_{k_to_ivl.get(k, '?')})" for k in max_path_pool])
        if len(path_str) > 200:
            path_str = path_str[:200] + "..."
        print(f"    Path: {path_str}")
        print(f"    Top I_{max(path_ivls)}, Bottom I_{min(path_ivls)}")

    return {
        'max_depth_T': max_depth,
        'max_path_T': max_path,
        'max_depth_pool': max_depth_pool,
        'max_path_pool': max_path_pool
    }


# ================================================================
#  PART E: "FORCED EXPANSION" CALCULATION
# ================================================================

def part_E(ctx, partA, partD):
    print("\n" + "=" * 72)
    print("  PART E: FORCED EXPANSION CALCULATION")
    print("=" * 72)

    n = ctx['n']
    T = partA['best_T']
    T_set = partA['best_T_set']
    targets = ctx['targets']
    k_to_ivl = ctx['k_to_interval']
    intervals = ctx['intervals']

    ivl_data = {}
    for jj, ivl_lo, ivl_hi in intervals:
        T_in_j = [k for k in T if k_to_ivl.get(k) == jj]
        I_j = [k for k in ctx['pool_smooth'] if ivl_lo <= k <= ivl_hi]
        if not T_in_j:
            continue
        degs_j = [len(targets.get(k, set())) for k in I_j]
        alpha_j = sum(degs_j) / len(I_j) if I_j else 0

        NH_j = set()
        for k in T_in_j:
            NH_j |= targets.get(k, set())

        ivl_data[jj] = {
            'T_in_j': len(T_in_j),
            'I_j_size': len(I_j),
            'alpha_j': alpha_j,
            'NH_j': len(NH_j),
            'actual_ratio_j': len(NH_j) / len(T_in_j) if T_in_j else 0
        }

    print(f"\n  Per-interval forced expansion:")
    print(f"  {'Ivl':>5} {'|T nI_j|':>8} {'|I_j|':>7} {'alpha_j':>8} {'|NH_j|':>8} "
          f"{'ratio_j':>8} {'alpha*|TnI|':>12}")

    total_forced = 0
    for jj, ivl_lo, ivl_hi in intervals:
        if jj not in ivl_data:
            continue
        d = ivl_data[jj]
        forced = d['alpha_j'] * d['T_in_j']
        total_forced += forced
        print(f"  I_{jj:>3}: {d['T_in_j']:>7} {d['I_j_size']:>7} {d['alpha_j']:>8.2f} "
              f"{d['NH_j']:>8} {d['actual_ratio_j']:>8.2f} {forced:>12.1f}")

    print(f"\n  Sum alpha_j * |T n I_j| = {total_forced:.1f}")
    print(f"  |T| = {len(T)}, |NH(T)| = {len(partA['best_NH'])}")
    print(f"  Forced expansion lower bound: {total_forced:.1f} vs |T| = {len(T)}")
    print(f"  Forced ratio: {total_forced / max(len(T), 1):.4f}")

    if ivl_data:
        j_min = min(ivl_data.keys())
        j_max = max(ivl_data.keys())
        d_min = ivl_data[j_min]
        d_max = ivl_data[j_max]
        print(f"\n  Span: I_{j_min} to I_{j_max}")
        print(f"  Bottom interval I_{j_min}: alpha={d_min['alpha_j']:.2f}, "
              f"|T n I|={d_min['T_in_j']}")
        print(f"  Top interval I_{j_max}: alpha={d_max['alpha_j']:.2f}, "
              f"|T n I|={d_max['T_in_j']}")
        print(f"  Expansion ratio at bottom: {d_min['actual_ratio_j']:.2f}")
        print(f"  Expansion ratio at top: {d_max['actual_ratio_j']:.2f}")

    return {'ivl_data': ivl_data, 'total_forced': total_forced}


# ================================================================
#  PART F: CAN A SELF-COVERING T VIOLATE HALL?
# ================================================================

def part_F(ctx):
    print("\n" + "=" * 72)
    print("  PART F: SELF-COVERING T CONSTRUCTION")
    print("=" * 72)

    n = ctx['n']
    pool = ctx['pool_smooth']
    pool_set = ctx['pool_set']
    targets = ctx['targets']
    h_to_div = ctx['h_to_divisors']
    k_to_ivl = ctx['k_to_interval']
    intervals = ctx['intervals']

    degs = [(len(targets.get(k, set())), k) for k in pool]
    degs.sort()

    best_T_sc = None
    best_ratio_sc = float('inf')
    best_NH_sc = None
    best_sc_frac = 0

    max_T_size = min(2000, len(pool))

    seeds = [k for _, k in degs[:20]]
    random.seed(123)
    if len(pool) > 50:
        seeds += random.sample(pool, min(30, len(pool)))
    seeds = list(dict.fromkeys(seeds))

    t0 = time.time()

    for seed_idx, seed in enumerate(seeds):
        if time.time() - t0 > 120:  # Time limit: 2 minutes per n
            print(f"  [Time limit reached after {seed_idx} seeds]")
            break

        T_sc = [seed]
        T_sc_set = {seed}
        NH_sc = set(targets.get(seed, set()))

        need_covering = {seed}

        iteration = 0
        while need_covering and len(T_sc) < max_T_size:
            iteration += 1
            if iteration > max_T_size * 2:
                break

            k = next(iter(need_covering))
            k_tgts = targets.get(k, set())

            # Find targets of k with mu_T(h) == 1 (private targets)
            # Efficient: check each target, count how many T elements hit it
            uncovered = []
            for h in k_tgts:
                cnt = 0
                for d in h_to_div[h]:
                    if d in T_sc_set:
                        cnt += 1
                        if cnt >= 2:
                            break
                if cnt < 2:
                    uncovered.append(h)

            if not uncovered:
                need_covering.discard(k)
                continue

            # Find element covering most uncovered targets
            cover_count = Counter()
            for h in uncovered:
                for d in h_to_div[h]:
                    if d != k and d not in T_sc_set:
                        cover_count[d] += 1

            if not cover_count:
                need_covering.discard(k)
                continue

            best_cover = cover_count.most_common(1)[0][0]
            T_sc.append(best_cover)
            T_sc_set.add(best_cover)
            NH_sc |= targets.get(best_cover, set())
            need_covering.add(best_cover)

            # Re-check k: are all its targets now covered by 2+ T elements?
            still_uncov = False
            for h in k_tgts:
                cnt = 0
                for d in h_to_div[h]:
                    if d in T_sc_set:
                        cnt += 1
                        if cnt >= 2:
                            break
                if cnt < 2:
                    still_uncov = True
                    break
            if not still_uncov:
                need_covering.discard(k)

        # Efficient self-covering check via mu_T
        mu_T_sc = Counter()
        for kk in T_sc:
            for h in targets.get(kk, set()):
                mu_T_sc[h] += 1

        zero_priv = 0
        for kk in T_sc:
            is_covered = True
            for h in targets.get(kk, set()):
                if mu_T_sc[h] < 2:
                    is_covered = False
                    break
            if is_covered:
                zero_priv += 1

        sc_frac = zero_priv / len(T_sc) if T_sc else 0
        ratio_sc = len(NH_sc) / len(T_sc) if T_sc else float('inf')

        if ratio_sc < best_ratio_sc:
            best_ratio_sc = ratio_sc
            best_T_sc = list(T_sc)
            best_NH_sc = set(NH_sc)
            best_sc_frac = sc_frac

    if best_T_sc is None:
        print("  No self-covering T could be constructed.")
        return {'T_sc': [], 'NH_sc': set(), 'ratio_sc': float('inf'),
                'sc_frac': 0, 'descent_links': 0}

    print(f"\n  Best self-covering T found:")
    print(f"    |T| = {len(best_T_sc)}, |NH(T)| = {len(best_NH_sc)}, "
          f"ratio = {best_ratio_sc:.4f}")
    print(f"    Self-covering fraction: {best_sc_frac:.4f} "
          f"({best_sc_frac * 100:.1f}%)")
    print(f"    Hall margin: {len(best_NH_sc) - len(best_T_sc)}")
    print(f"    Hall satisfied: {len(best_NH_sc) >= len(best_T_sc)}")

    # Interval distribution
    ivl_counts = Counter()
    for k in best_T_sc:
        j = k_to_ivl.get(k, -1)
        ivl_counts[j] += 1
    print(f"\n  Interval distribution:")
    for jj, ivl_lo, ivl_hi in intervals:
        cnt = ivl_counts.get(jj, 0)
        if cnt > 0:
            print(f"    I_{jj} [{ivl_lo}, {ivl_hi}]: {cnt} "
                  f"({cnt / len(best_T_sc) * 100:.1f}%)")

    # Descent links
    T_sc_set = set(best_T_sc)
    descent_links = 0
    for k in best_T_sc:
        for p in prime_divisors(k):
            if k // p in T_sc_set:
                descent_links += 1

    print(f"\n  Descent links in self-covering T: {descent_links}")
    print(f"  Descent link density: {descent_links / max(len(best_T_sc), 1):.3f} per element")

    return {
        'T_sc': best_T_sc, 'NH_sc': best_NH_sc,
        'ratio_sc': best_ratio_sc, 'sc_frac': best_sc_frac,
        'descent_links': descent_links
    }


# ================================================================
#  MAIN
# ================================================================

def run_one_n(n):
    print("\n" + "#" * 72)
    print(f"  Z41 DESCENT CHAINS: n = {n}")
    print("#" * 72)

    t0 = time.time()
    ctx = setup_n(n)
    if ctx is None:
        print(f"  Setup failed for n={n}")
        return None

    L = ctx['L']
    B = ctx['B']
    delta = ctx['delta']
    pool = ctx['pool_smooth']

    print(f"  n={n}, L={L}, B={B}, delta={delta:.4f}")
    print(f"  |S_+| = {len(pool)}, |H_smooth| = {len(ctx['H_all'])}")

    rA = part_A(ctx)
    rB = part_B(ctx, rA)
    rC = part_C(ctx, rA)
    rD = part_D(ctx, rA)
    rE = part_E(ctx, rA, rD)
    rF = part_F(ctx)

    elapsed = time.time() - t0
    print(f"\n  Total time for n={n}: {elapsed:.1f}s")

    return {
        'n': n, 'L': L, 'B': B, 'delta': delta,
        'pool_size': len(pool), 'H_size': len(ctx['H_all']),
        'A': rA, 'B': rB, 'C': rC, 'D': rD, 'E': rE, 'F': rF,
        'elapsed': elapsed
    }


def main():
    print("=" * 72)
    print("  ERDOS 710 -- Z41: DESCENT CHAIN STRUCTURE")
    print("  Testing descent chains in tight/adversarial subsets")
    print("=" * 72)

    N_VALUES = [1000, 5000, 10000, 50000, 100000]
    all_results = {}

    for n in N_VALUES:
        res = run_one_n(n)
        if res is not None:
            all_results[n] = res

    # ================================================================
    #  SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 72)
    print("  Z41 SUMMARY TABLE")
    print("=" * 72)

    print(f"\n  {'n':>7} {'|S+|':>6} {'delta':>6} {'|T|':>5} {'ratio':>7} "
          f"{'0-priv%':>8} {'desc_T%':>8} {'desc_S+%':>8} "
          f"{'cov_lower%':>10} "
          f"{'depth_T':>8} {'depth_S+':>8} "
          f"{'sc_ratio':>8} {'sc_frac':>8}")

    for n in N_VALUES:
        if n not in all_results:
            continue
        r = all_results[n]
        rA = r['A']
        rB_data = r['B']
        rC_data = r['C']
        rD_data = r['D']
        rF_data = r['F']

        zero_priv_pct = rA['zero_private'] / max(len(rA['best_T']), 1) * 100
        desc_T_pct = (rB_data['descent_in_T'] / max(rB_data['total_prime_slots'], 1) * 100)
        desc_pool_pct = (rB_data['descent_in_pool'] / max(rB_data['total_prime_slots'], 1) * 100)
        cov_total = rC_data['covering_in_T'] + rC_data['covering_outside_T']
        cov_lower_pct = rC_data['covering_in_lower'] / max(cov_total, 1) * 100

        print(f"  {n:>7} {r['pool_size']:>6} {r['delta']:>6.2f} "
              f"{len(rA['best_T']):>5} {rA['best_ratio']:>7.3f} "
              f"{zero_priv_pct:>7.1f}% "
              f"{desc_T_pct:>7.1f}% {desc_pool_pct:>7.1f}% "
              f"{cov_lower_pct:>9.1f}% "
              f"{rD_data['max_depth_T']:>8} {rD_data['max_depth_pool']:>8} "
              f"{rF_data['ratio_sc']:>8.3f} {rF_data['sc_frac']:>8.3f}")

    # Key observations
    print("\n" + "=" * 72)
    print("  KEY OBSERVATIONS")
    print("=" * 72)

    print("""
  1. DESCENT LINKS: What fraction of k/p links stay in T vs S_+?
     If most descent links go OUTSIDE T, tight sets avoid chains.

  2. SELF-COVERING FRACTION: What % of T has zero private targets?
     If low, most elements contribute unique targets => Hall holds easily.

  3. DESCENT DEPTH: How deep do chains go?
     Deeper chains reach intervals with higher alpha => forced expansion.

  4. FORCED EXPANSION: Does the bottom interval's alpha force NH(T) >> |T|?
     If alpha_{j_min} >> 1 and descent forces elements into I_{j_min},
     then self-covering T cannot violate Hall.

  5. SELF-COVERING T vs HALL: Can we build a self-covering T that
     comes close to violating Hall? If not, this is a proof strategy.
""")

    print(f"\n  Total runtime: {sum(r['elapsed'] for r in all_results.values()):.1f}s")


if __name__ == "__main__":
    main()
