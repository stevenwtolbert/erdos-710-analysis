#!/usr/bin/env python3
"""
ERDŐS 710 — Z65: SAWTOOTH PLOT DATA

Generate dense data for the sawtooth plot of Σ 1/CS_ref vs n.

The sawtooth has:
- Within each J-regime: slow decrease (δ growing, CS improving)
- At J-transitions: sudden jumps (new interval shockwave)
- Peak envelope: rises to global max at j=15, then decreases

This produces:
1. Dense plot data (every 50-500 integers for n ≤ 100,000; every 1000-5000 for larger)
2. Clear J-transition markers
3. The peak envelope
4. The Σ 1/α curve alongside Σ 1/CS_ref
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


def compute_sums(n):
    """Compute J, Σ 1/α, Σ 1/CS_ref, δ for a given n."""
    L, M, B = compute_params(n)
    n_half = n // 2
    nL = int(n + L)
    delta = 2 * M / n - 1

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
    sum_1_csref = 0

    for j in js:
        I_j = sorted(intervals[j])

        tgt_to_srcs = defaultdict(set)
        for k in I_j:
            for h in adj.get(k, set()):
                tgt_to_srcs[h].add(k)

        new_count = {k: len(adj.get(k, set())) for k in I_j}
        NH = set()
        rem = set(I_j)
        tau = defaultdict(int)
        E1 = 0
        E2 = 0
        n_unique = 0
        T_size = 0
        min_alpha = float('inf')
        min_cs_ref = float('inf')

        for step in range(len(I_j)):
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

            for h in adj.get(best_k, set()):
                old_tau = tau[h]
                tau[h] = old_tau + 1
                E2 += 2 * old_tau + 1
                if old_tau == 0:
                    n_unique += 1
                elif old_tau == 1:
                    n_unique -= 1
            E1 += len(adj.get(best_k, set()))

            newly = adj.get(best_k, set()) - NH
            NH |= newly

            alpha = len(NH) / T_size
            U = n_unique
            e1 = E1 - U
            e2 = E2 - U
            if e2 > 0 and e1 > 0:
                cs_ref = (U + e1 * e1 / e2) / T_size
            elif U > 0:
                cs_ref = U / T_size
            else:
                cs_ref = float('inf')

            if alpha < min_alpha:
                min_alpha = alpha
            if cs_ref < min_cs_ref:
                min_cs_ref = cs_ref

            for h in newly:
                for k2 in tgt_to_srcs[h]:
                    if k2 in rem:
                        new_count[k2] -= 1

        if min_alpha > 0 and min_alpha != float('inf'):
            sum_1_alpha += 1.0 / min_alpha
        if min_cs_ref > 0 and min_cs_ref != float('inf'):
            sum_1_csref += 1.0 / min_cs_ref

    return (J, sum_1_alpha, sum_1_csref, delta)


def main():
    print("ERDŐS 710 — Z65: SAWTOOTH PLOT DATA")
    print("=" * 100)
    t0 = time.time()

    # Build a dense grid with adaptive step size
    # Dense near transitions, sparser elsewhere
    test_ns = []

    # n = 100 to 1000: step 20
    test_ns.extend(range(100, 1001, 20))

    # n = 1000 to 5000: step 50
    test_ns.extend(range(1000, 5001, 50))

    # n = 5000 to 20000: step 200
    test_ns.extend(range(5000, 20001, 200))

    # n = 20000 to 100000: step 500
    test_ns.extend(range(20000, 100001, 500))

    # n = 100000 to 300000: step 2000
    test_ns.extend(range(100000, 300001, 2000))

    # Remove duplicates and sort
    test_ns = sorted(set(test_ns))

    print(f"\n  Computing Σ 1/CS_ref at {len(test_ns)} n-values from {test_ns[0]} to {test_ns[-1]}")
    print()

    # Output format: n, J, Σ1/α, Σ1/CSr, δ
    data = []
    prev_J = None
    transitions = []
    peaks_per_J = {}
    global_max_csref = 0
    global_max_n = 0

    batch_size = 50
    for batch_start in range(0, len(test_ns), batch_size):
        batch = test_ns[batch_start:batch_start + batch_size]
        for n in batch:
            res = compute_sums(n)
            if res is None:
                continue

            J, s_alpha, s_csref, delta = res
            data.append((n, J, s_alpha, s_csref, delta))

            if prev_J is not None and J != prev_J:
                transitions.append((n, prev_J, J))

            if J not in peaks_per_J or s_csref > peaks_per_J[J][1]:
                peaks_per_J[J] = (n, s_csref)

            if s_csref > global_max_csref:
                global_max_csref = s_csref
                global_max_n = n

            prev_J = J

        elapsed = time.time() - t0
        last_n = batch[-1]
        print(f"\r  n up to {last_n:>7}: {len(data)} points, "
              f"max Σ1/CSr = {global_max_csref:.6f} at n={global_max_n}, "
              f"{elapsed:.0f}s", end="")
        sys.stdout.flush()

        if elapsed > 7200:  # 2 hour timeout
            print("\n  TIMEOUT")
            break

    print()

    total_time = time.time() - t0

    # Write plot data to CSV
    csv_path = "/home/ashbringer/projects/e710_new_H/data_z65_sawtooth.csv"
    with open(csv_path, 'w') as f:
        f.write("n,J,sum_1_alpha,sum_1_csref,delta\n")
        for row in data:
            f.write(f"{row[0]},{row[1]},{row[2]:.8f},{row[3]:.8f},{row[4]:.6f}\n")

    print(f"\n  Plot data: {csv_path} ({len(data)} points)")

    # Summary
    print(f"\n{'='*100}")
    print(f"  SAWTOOTH SUMMARY")
    print(f"{'='*100}")
    print()

    print(f"  {'J':>3} {'peak_n':>8} {'Σ1/CSr_peak':>12}")
    print("  " + "-" * 30)
    for J in sorted(peaks_per_J.keys()):
        n, s = peaks_per_J[J]
        print(f"  {J:>3} {n:>8} {s:>12.6f}")

    print(f"\n  Global max: Σ1/CSr = {global_max_csref:.6f} at n = {global_max_n}")
    print(f"  Margin: {1.0 - global_max_csref:.6f} ({(1.0 - global_max_csref)*100:.2f}%)")
    print(f"  J-transitions found: {len(transitions)}")
    for n, J_old, J_new in transitions:
        direction = "↑" if J_new > J_old else "↓"
        print(f"    n={n:>7}: J {J_old} → {J_new} {direction}")

    print(f"\n  Total: {len(data)} data points in {total_time:.0f}s")

    # Write state file
    state_path = "/home/ashbringer/projects/e710_new_H/states/state_76_z65_sawtooth.md"
    with open(state_path, 'w') as f:
        f.write("# State 76: Z65 Sawtooth Plot Data\n\n")
        f.write(f"## Data: {csv_path}\n\n")
        f.write(f"{len(data)} data points from n={data[0][0]} to n={data[-1][0]}\n\n")
        f.write("## Peak Envelope\n\n")
        f.write("| J | peak_n | Σ1/CSr |\n")
        f.write("|---|--------|--------|\n")
        for J in sorted(peaks_per_J.keys()):
            n, s = peaks_per_J[J]
            f.write(f"| {J} | {n} | {s:.6f} |\n")
        f.write(f"\n## Global Max\n\n")
        f.write(f"Σ1/CSr = {global_max_csref:.6f} at n = {global_max_n}\n")
        f.write(f"Margin to 1.0: {1.0 - global_max_csref:.6f} ({(1.0 - global_max_csref)*100:.2f}%)\n")
        f.write(f"\n## Transitions\n\n")
        for n, J_old, J_new in transitions:
            f.write(f"- n={n}: J {J_old} → {J_new}\n")

    print(f"  State file: {state_path}")


if __name__ == '__main__':
    main()
