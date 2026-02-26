#!/usr/bin/env python3
"""
Z114b: Codegree-by-layer analysis for Erdos 710 bipartite graph.

Bipartite graph: V = (B, N], H = (2n, n+L], edge k-h iff k | h.
Split V at cutpoint N/c. V_hard = (N/c, N].
For k in V_hard and h in H, multiplier j = h/k is restricted to a small range,
so tau_hard(h) = |{k in V_hard : k | h}| should be bounded.

Hypothesis: max_tau_hard = O(1) while d_min -> infinity,
so d_min / max_tau_hard -> infinity and Hall holds trivially for V_hard.
"""

import math
from collections import defaultdict

def run_analysis(n, c):
    N = n // 2
    B = int(math.exp(math.sqrt(math.log(n) * math.log(math.log(n)))))
    C_val = 2.0 / math.sqrt(math.e) + 0.05
    L = math.ceil(C_val * n * math.sqrt(math.log(n) / math.log(math.log(n))))
    M = L - n

    # V_hard = (N/c, N]
    v_lo = N // c  # exclusive lower bound
    v_hi = N       # inclusive upper bound

    # H = (2n, n+L] = (2n, 2n + M]
    h_lo = 2 * n       # exclusive
    h_hi = n + L        # inclusive (= 2n + M)

    # Multiplier range for k in V_hard, h in H:
    #   j = h/k, so j_min ~ h_lo / v_hi, j_max ~ h_hi / v_lo
    # More precisely:
    j_min_possible = math.ceil((h_lo + 1) / v_hi)    # smallest j s.t. j*k can be in H for some k <= v_hi
    j_max_possible = (h_hi) // (v_lo + 1)             # largest j s.t. j*k can be in H for some k > v_lo

    # Build V_hard list
    V_hard = list(range(v_lo + 1, v_hi + 1))

    # Compute degree of each k in V_hard: d(k) = |{h in H : k | h}|
    # h = j*k for j in [ceil((h_lo+1)/k), floor(h_hi/k)]
    degrees = {}
    for k in V_hard:
        if k < B:
            continue  # skip if below smoothness bound
        j_lo_k = math.ceil((h_lo + 1) / k)
        j_hi_k = h_hi // k
        deg = max(0, j_hi_k - j_lo_k + 1)
        degrees[k] = deg

    if not degrees:
        return None

    d_min = min(degrees.values())
    d_max = max(degrees.values())
    d_mean = sum(degrees.values()) / len(degrees)

    # Compute tau_hard(h) for each h in H
    # tau_hard(h) = |{k in V_hard : k | h}|
    # Only count k >= B (in the actual vertex set)
    tau = defaultdict(int)
    for k in V_hard:
        if k < B:
            continue
        j_lo_k = math.ceil((h_lo + 1) / k)
        j_hi_k = h_hi // k
        for j in range(j_lo_k, j_hi_k + 1):
            h = j * k
            if h_lo < h <= h_hi:
                tau[h] += 1

    if not tau:
        return None

    tau_values = list(tau.values())
    max_tau = max(tau_values)
    mean_tau = sum(tau_values) / len(tau_values)

    ratio = d_min / max_tau if max_tau > 0 else float('inf')

    return {
        'n': n,
        'c': c,
        'V_hard_size': len([k for k in V_hard if k >= B]),
        'M': M,
        'j_range': (j_min_possible, j_max_possible),
        'd_min': d_min,
        'd_max': d_max,
        'd_mean': d_mean,
        'max_tau': max_tau,
        'mean_tau': mean_tau,
        'ratio': ratio,
    }


def main():
    ns = [5000, 10000, 20000, 50000, 100000]
    cs = [2, 3, 4]

    print("=" * 120)
    print("Z114b: Codegree-by-layer analysis — V_hard = (N/c, N]")
    print("Hypothesis: max tau_hard = O(1), d_min -> inf, so d_min/max_tau -> inf => Hall trivial for V_hard")
    print("=" * 120)
    print()

    header = f"{'n':>8} {'c':>3} {'|V_hard|':>8} {'j_range':>12} {'d_min':>7} {'d_max':>7} {'d_mean':>8} {'max_tau':>8} {'mean_tau':>9} {'d_min/max_tau':>13}"
    print(header)
    print("-" * len(header))

    for n in ns:
        for c in cs:
            res = run_analysis(n, c)
            if res is None:
                print(f"{n:>8} {c:>3}  (no valid vertices)")
                continue
            jr = f"[{res['j_range'][0]},{res['j_range'][1]}]"
            print(f"{res['n']:>8} {res['c']:>3} {res['V_hard_size']:>8} {jr:>12} "
                  f"{res['d_min']:>7} {res['d_max']:>7} {res['d_mean']:>8.1f} "
                  f"{res['max_tau']:>8} {res['mean_tau']:>9.3f} {res['ratio']:>13.2f}")
        print()

    # Summary analysis
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print()
    print("For each c, track how max_tau and d_min/max_tau scale with n:")
    print()
    for c in cs:
        print(f"  c = {c}:")
        for n in ns:
            res = run_analysis(n, c)
            if res is None:
                continue
            print(f"    n={n:>6}: max_tau={res['max_tau']:>3}, d_min={res['d_min']:>5}, "
                  f"ratio={res['ratio']:>8.2f}, j_range=[{res['j_range'][0]},{res['j_range'][1]}]")
        print()


if __name__ == "__main__":
    main()
