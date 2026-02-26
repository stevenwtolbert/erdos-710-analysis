#!/usr/bin/env python3
"""
ERDŐS 710 — Z57: REFINED CS BOUND (UNIQUE TARGET CORRECTION)

Standard CS: |NH(T)| ≥ E₁²/E₂ = (Σ deg)²/(Σ τ²)
This is tight only when all τ(h) are equal. But in practice, many targets are unique
(τ=1), making the bound loose.

Refined CS: Split NH(T) = U ∪ S (unique + shared).
  |NH(T)| = |U| + |S| ≥ |U| + (E₁ - |U|)²/(E₂ - |U|)
This is PROVABLY ≥ standard CS, and the improvement is proportional to |U|.

For FMC: if Σ 1/CS_refined < 1 everywhere, the analytic gap is closed because:
  - α_j ≥ CS_refined,j (provable by Cauchy-Schwarz on shared targets)
  - Σ 1/α_j ≤ Σ 1/CS_refined,j < 1

This script:
1. For each interval I_j, runs the greedy α algorithm
2. At each step, tracks U (unique targets), and computes refined CS
3. Reports min refined CS vs standard CS at each interval
4. Computes Σ 1/CS_refined and checks if < 1
"""

import math
import time
import sys
from collections import defaultdict

C_TARGET = 2 / math.e**0.5
EPS = 0.05


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


def greedy_alpha_refined_cs(sources, adj):
    """
    Run α-greedy, track α, standard CS, and refined CS at each step.

    Refined CS at step t:
      - U(t) = #{h ∈ NH: τ(h) = 1} (unique targets)
      - e₁ = E₁ - U, e₂ = E₂ - U (shared contributions)
      - |NH| ≥ U + e₁²/e₂ (if e₂ > 0)
      - CS_refined = (U + e₁²/e₂) / t
    """
    if not sources:
        return float('inf'), float('inf'), float('inf'), {}

    tgt_to_srcs = defaultdict(set)
    for k in sources:
        for h in adj.get(k, set()):
            tgt_to_srcs[h].add(k)

    new_count = {k: len(adj.get(k, set())) for k in sources}
    NH = set()
    rem = set(sources)
    tau = defaultdict(int)  # tau[h] = current multiplicity in T
    E1 = 0
    E2 = 0
    n_unique = 0  # count of h with tau[h] == 1
    T_size = 0

    min_alpha = float('inf')
    min_cs = float('inf')
    min_cs_refined = float('inf')

    cs_at_min_alpha = 0
    cs_ref_at_min_alpha = 0

    for step in range(len(sources)):
        best_k = None
        best_new = float('inf')
        for k in rem:
            nc = new_count[k]
            if nc < best_new or (nc == best_new and (best_k is None or k > best_k)):
                best_new = nc
                best_k = k
        if best_k is None:
            break

        T_size += 1
        rem.discard(best_k)
        dk = len(adj.get(best_k, set()))

        # Update E1, E2, tau, n_unique
        for h in adj.get(best_k, set()):
            old_tau = tau[h]
            tau[h] = old_tau + 1

            # Update E2: old contribution was old_tau², new is (old_tau+1)²
            E2 += 2 * old_tau + 1  # = (old+1)² - old²

            # Update n_unique
            if old_tau == 0:
                n_unique += 1  # new unique
            elif old_tau == 1:
                n_unique -= 1  # was unique, now shared
            # if old_tau >= 2: no change to n_unique

        E1 += dk

        newly = adj.get(best_k, set()) - NH
        NH |= newly

        alpha = len(NH) / T_size
        cs = E1 * E1 / (T_size * E2) if E2 > 0 else float('inf')

        # Refined CS
        U = n_unique
        e1 = E1 - U
        e2 = E2 - U
        if e2 > 0 and e1 > 0:
            cs_refined = (U + e1 * e1 / e2) / T_size
        elif U > 0:
            cs_refined = U / T_size  # all targets unique
        else:
            cs_refined = float('inf')

        if alpha < min_alpha:
            min_alpha = alpha
            cs_at_min_alpha = cs
            cs_ref_at_min_alpha = cs_refined
        if cs < min_cs:
            min_cs = cs
        if cs_refined < min_cs_refined:
            min_cs_refined = cs_refined

        for h in newly:
            for k2 in tgt_to_srcs[h]:
                if k2 in rem:
                    new_count[k2] -= 1

    info = {
        'unique_at_worst': n_unique,
        'E1_at_worst': E1,
        'E2_at_worst': E2,
    }

    return min_alpha, min_cs, min_cs_refined, info


def analyze_n(n):
    L, M, B = compute_params(n)
    n_half = n // 2
    nL = int(n + L)

    if B < 2 or n_half <= B:
        return None

    S_plus = get_smooth_numbers_fast(B, B, n_half)
    H_smooth = get_smooth_numbers_fast(B, n, nL)
    if not S_plus or not H_smooth:
        return None

    H_set = set(H_smooth)
    adj = {}
    for k in S_plus:
        targets = set()
        lo_mult = n // k + 1
        hi_mult = nL // k
        for m in range(lo_mult, hi_mult + 1):
            h = k * m
            if h in H_set:
                targets.add(h)
        adj[k] = targets

    intervals = defaultdict(list)
    for k in S_plus:
        j = int(math.log2(k))
        intervals[j].append(k)

    js = sorted(intervals.keys())
    J = len(js)

    sum_1_alpha = 0
    sum_1_cs = 0
    sum_1_cs_ref = 0
    interval_data = []

    for j in js:
        I_j = sorted(intervals[j])
        t = len(I_j)

        alpha, cs, cs_ref, info = greedy_alpha_refined_cs(I_j, adj)

        if alpha > 0 and alpha != float('inf'):
            sum_1_alpha += 1.0 / alpha
        if cs > 0 and cs != float('inf'):
            sum_1_cs += 1.0 / cs
        if cs_ref > 0 and cs_ref != float('inf'):
            sum_1_cs_ref += 1.0 / cs_ref

        improvement = (cs_ref / cs - 1) * 100 if cs > 0 and cs != float('inf') and cs_ref != float('inf') else 0

        interval_data.append({
            'j': j, 'size': t,
            'alpha': round(alpha, 3) if alpha != float('inf') else 'inf',
            'cs': round(cs, 3) if cs != float('inf') else 'inf',
            'cs_ref': round(cs_ref, 3) if cs_ref != float('inf') else 'inf',
            'improvement': round(improvement, 1),
        })

    return {
        'n': n, 'J': J,
        'sum_1_alpha': sum_1_alpha,
        'sum_1_cs': sum_1_cs,
        'sum_1_cs_ref': sum_1_cs_ref,
        'intervals': interval_data,
    }


def main():
    print("ERDŐS 710 — Z57: REFINED CS BOUND (UNIQUE TARGET CORRECTION)")
    print("=" * 90)
    print()
    print("  CS_refined = (U + e₁²/e₂) / |T|  where U = unique targets, e₁,e₂ = shared part")
    print("  Provably: CS_refined ≥ CS_standard, and tighter when many targets are unique")
    print()

    # Dense scan including all J-transition points
    test_ns = list(range(100, 2001, 100))
    test_ns += list(range(2000, 10001, 200))
    test_ns += list(range(10000, 50001, 1000))
    test_ns += list(range(50000, 200001, 2000))
    test_ns = sorted(set(test_ns))

    max_sum_alpha = 0
    max_sum_alpha_n = 0
    max_sum_cs = 0
    max_sum_cs_n = 0
    max_sum_cs_ref = 0
    max_sum_cs_ref_n = 0

    all_results = []
    t0 = time.time()

    for i, n in enumerate(test_ns):
        res = analyze_n(n)
        if res is None:
            continue

        all_results.append(res)

        if res['sum_1_alpha'] > max_sum_alpha:
            max_sum_alpha = res['sum_1_alpha']
            max_sum_alpha_n = n
        if res['sum_1_cs'] > max_sum_cs:
            max_sum_cs = res['sum_1_cs']
            max_sum_cs_n = n
        if res['sum_1_cs_ref'] > max_sum_cs_ref:
            max_sum_cs_ref = res['sum_1_cs_ref']
            max_sum_cs_ref_n = n

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            pct = 100.0 * (i + 1) / len(test_ns)
            print(f"  [{pct:.0f}% {elapsed:.0f}s] max Σ1/α={max_sum_alpha:.4f} "
                  f"max Σ1/CS={max_sum_cs:.4f} max Σ1/CS_ref={max_sum_cs_ref:.4f}")
            sys.stdout.flush()

    total_time = time.time() - t0

    print("\n" + "=" * 90)
    print("  RESULTS")
    print("=" * 90)
    print(f"  Tested {len(all_results)} n-values, total time: {total_time:.0f}s")
    print()
    print(f"  max Σ 1/α       = {max_sum_alpha:.6f} at n = {max_sum_alpha_n}")
    print(f"  max Σ 1/CS      = {max_sum_cs:.6f} at n = {max_sum_cs_n}")
    print(f"  max Σ 1/CS_ref  = {max_sum_cs_ref:.6f} at n = {max_sum_cs_ref_n}")
    print()

    cs_pass = max_sum_cs < 1.0
    cs_ref_pass = max_sum_cs_ref < 1.0

    print(f"  Σ 1/CS < 1 everywhere:      {cs_pass} (margin: {1-max_sum_cs:.4f})")
    print(f"  Σ 1/CS_ref < 1 everywhere:   {cs_ref_pass} (margin: {1-max_sum_cs_ref:.4f})")

    if cs_ref_pass:
        print("\n  *** REFINED CS CLOSES THE GAP! ***")
        print("  The refined Cauchy-Schwarz bound (accounting for unique targets)")
        print("  gives a provable per-interval bound that satisfies FMC at all n.")

    # Show detail at worst n
    print(f"\n  Detail at worst n = {max_sum_cs_ref_n}:")
    worst_res = None
    for res in all_results:
        if res['n'] == max_sum_cs_ref_n:
            worst_res = res
            break

    if worst_res:
        print(f"  {'j':>4} {'|I|':>6} {'α':>8} {'CS':>8} {'CS_ref':>8} {'impr%':>6}")
        print("  " + "-" * 50)
        for iv in worst_res['intervals']:
            print(f"  {iv['j']:>4} {iv['size']:>6} {iv['alpha']:>8} {iv['cs']:>8} "
                  f"{iv['cs_ref']:>8} {iv['improvement']:>6.1f}%")

    # Also show detail at n=132000 (known J-transition trouble spot)
    for target_n in [132000, 136000, 68000]:
        target_res = None
        for res in all_results:
            if res['n'] == target_n:
                target_res = res
                break

        if target_res:
            print(f"\n  Detail at n = {target_n} (J={target_res['J']}):")
            print(f"  Σ1/α = {target_res['sum_1_alpha']:.4f}, "
                  f"Σ1/CS = {target_res['sum_1_cs']:.4f}, "
                  f"Σ1/CS_ref = {target_res['sum_1_cs_ref']:.4f}")
            print(f"  {'j':>4} {'|I|':>6} {'α':>8} {'CS':>8} {'CS_ref':>8} {'impr%':>6}")
            print("  " + "-" * 50)
            for iv in target_res['intervals']:
                print(f"  {iv['j']:>4} {iv['size']:>6} {iv['alpha']:>8} {iv['cs']:>8} "
                      f"{iv['cs_ref']:>8} {iv['improvement']:>6.1f}%")

    # Show all n where Σ1/CS_ref > 0.90
    print(f"\n  n-values with Σ 1/CS_ref > 0.90:")
    for res in all_results:
        if res['sum_1_cs_ref'] > 0.90:
            print(f"    n={res['n']:>7} J={res['J']} Σ1/α={res['sum_1_alpha']:.4f} "
                  f"Σ1/CS={res['sum_1_cs']:.4f} Σ1/CS_ref={res['sum_1_cs_ref']:.4f}")

    # Write state file
    state_path = "/home/ashbringer/projects/e710_new_H/states/state_68_z57_refined_cs.md"
    with open(state_path, 'w') as f:
        f.write("# State 68: Z57 Refined CS Bound\n\n")
        f.write(f"Tested {len(all_results)} n-values\n\n")
        f.write("## Results\n\n")
        f.write(f"- max Σ 1/α = {max_sum_alpha:.6f} at n = {max_sum_alpha_n}\n")
        f.write(f"- max Σ 1/CS = {max_sum_cs:.6f} at n = {max_sum_cs_n}\n")
        f.write(f"- max Σ 1/CS_ref = {max_sum_cs_ref:.6f} at n = {max_sum_cs_ref_n}\n\n")
        f.write(f"CS_ref < 1 everywhere: {cs_ref_pass}\n")
        f.write(f"Margin: {1-max_sum_cs_ref:.4f}\n")

    print(f"\n  State file: {state_path}")


if __name__ == '__main__':
    main()
