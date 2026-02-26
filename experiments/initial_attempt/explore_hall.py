#!/usr/bin/env python3
"""
Deep exploration of Hall's condition for V -> H.
Focus: why does the ratio |N_H(S)|/|S| improve for larger S?
"""
import sys, math, random
from math import gcd, log, sqrt, exp, ceil, floor
from collections import defaultdict
from itertools import combinations

random.seed(42)
C_TARGET = 2 / sqrt(exp(1))

def targets_H(k, n, L):
    """Multiples of k in H = (2n, n+L]."""
    j0 = (2*n) // k + 1
    j1 = (n+L) // k
    return set(k*j for j in range(j0, j1+1))

def target_L(n, eps=0.05):
    return int((C_TARGET + eps) * n * sqrt(log(n) / log(log(n))))

def explore_ratio_by_size(n, eps=0.05, num_samples=2000):
    """For each subset size s, find the worst ratio |N_H(S)|/s."""
    L = target_L(n, eps)
    M = L - n
    V = list(range(2, n//2 + 1))  # exclude 1 (trivially covers all)

    adj = {}
    for k in V:
        adj[k] = targets_H(k, n, L)

    print(f"\nn={n}, L={L}, M={M}, |V|={len(V)}, |H|={M}")
    print(f"  M/|V| = {M/len(V):.3f}")
    print()

    # Worst ratio by size
    print(f"  {'s':>4} {'worst_ratio':>11} {'worst_S_type':>15} {'avg_ratio':>10}")

    for s in [1, 2, 3, 5, 10, 20, 50, min(100, len(V)//2)]:
        if s > len(V): break
        worst_ratio = float('inf')
        worst_S = None
        ratios = []

        # Try random subsets
        for _ in range(num_samples):
            S = random.sample(V, s)
            NS = set()
            for k in S:
                NS |= adj[k]
            r = len(NS) / s
            ratios.append(r)
            if r < worst_ratio:
                worst_ratio = r
                worst_S = sorted(S)

        # Also try consecutive subsets near n/2 (known worst case)
        for a in range(max(2, n//2 - s - 10), n//2 - s + 2):
            S = list(range(a, a+s))
            S = [k for k in S if k in set(V)]
            if len(S) != s: continue
            NS = set()
            for k in S:
                NS |= adj[k]
            r = len(NS) / s
            ratios.append(r)
            if r < worst_ratio:
                worst_ratio = r
                worst_S = sorted(S)

        stype = "consecutive" if worst_S and worst_S == list(range(worst_S[0], worst_S[0]+s)) else "scattered"
        avg_r = sum(ratios) / len(ratios)
        print(f"  {s:>4} {worst_ratio:>11.4f} {stype:>15} {avg_r:>10.4f}")

def explore_interval_telescoping(n, eps=0.05):
    """
    For consecutive S = {a,...,a+s-1}, the complement product telescopes:
    prod_{k=a}^{a+s-1} (1 - 1/k) = (a-1)/(a+s-1)

    This gives |complement| ~ M * (a-1)/(a+s-1), so
    |N_H(S)| ~ M * s/(a+s-1)

    Hall needs: M * s/(a+s-1) >= s, i.e., M >= a+s-1.
    Since a <= n/2 and s <= n/2-a+1: a+s-1 <= n/2 < M. Done for intervals!

    But does this extend to non-consecutive S?
    """
    L = target_L(n, eps)
    M = L - n

    print(f"\n=== INTERVAL TELESCOPING TEST (n={n}) ===")
    print(f"M={M}, n/2={n//2}")

    # Verify telescoping formula for various intervals
    print(f"\n  {'a':>5} {'s':>4} {'actual_N':>9} {'predicted':>10} {'ratio':>7} {'M_bound':>8}")

    for a in [n//4, n//3, n//2-20, n//2-5, n//2-2]:
        for s in [2, 5, 10, 20]:
            if a + s - 1 > n//2: continue
            S = list(range(a, a+s))
            NS = set()
            for k in S:
                NS |= targets_H(k, n, L)
            actual = len(NS)
            # Predicted: M * s/(a+s-1) (from telescoping, approximate for divisibility)
            predicted = M * s / (a+s-1)
            print(f"  {a:>5} {s:>4} {actual:>9} {predicted:>10.1f} {actual/predicted:>7.3f} {'OK' if M >= a+s-1 else 'FAIL':>8}")

def explore_general_complement(n, eps=0.05):
    """
    For general S = {k_1,...,k_s}, the complement in H is:
    C_H(S) = {m in H : k_i does not divide m, for all i}

    By inclusion-exclusion with Mobius:
    |C_H(S)| = sum_{d | lcm_subset} mu_S(d) * floor(|H|/d)

    But for general S, we need: |N_H(S)| = M - |C_H(S)| >= s.

    Key insight: for S of CONSECUTIVE integers, the product telescopes.
    For general S, we can use the bound:

    prod_{k in S} (1 - 1/k) <= prod_{k=a}^{a+s-1} (1-1/k) = (a-1)/(a+s-1)

    where a = min(S). This is because 1-1/k is INCREASING in k, so
    replacing S with consecutive {a,...,a+s-1} makes each factor SMALLER
    (since those elements are <= the scattered ones).

    WAIT - this is WRONG. The product formula only gives complement
    for pairwise coprime moduli. For general moduli, inclusion-exclusion
    is needed.

    Let me think more carefully...
    """
    L = target_L(n, eps)
    M = L - n

    print(f"\n=== GENERAL COMPLEMENT ANALYSIS (n={n}) ===")

    V = list(range(2, n//2 + 1))
    adj = {}
    for k in V:
        adj[k] = targets_H(k, n, L)

    # For a set S, compute:
    # 1. Actual |N_H(S)|
    # 2. Product bound M * (1 - prod(1-1/k))
    # 3. Inclusion-exclusion (exact for small S)

    print(f"\n  S (elements) -> actual |N| vs product bound vs IE")

    test_sets = []
    # Consecutive near n/2
    a = n//2 - 5
    test_sets.append(("consec_near_n/2", list(range(a, a+5))))
    # Scattered near n/2
    test_sets.append(("scattered_n/2", [n//2-10, n//2-7, n//2-3, n//2-1, n//2]))
    # Coprime set near n/2
    # Find coprimes
    coprimes = []
    for k in range(n//2, n//4, -1):
        if all(gcd(k, c) == 1 for c in coprimes):
            coprimes.append(k)
            if len(coprimes) == 5: break
    test_sets.append(("coprime_near_n/2", sorted(coprimes)))
    # Set with shared factor 2
    evens = [k for k in range(n//2-10, n//2+1) if k % 2 == 0][:5]
    test_sets.append(("evens_near_n/2", evens))

    for name, S in test_sets:
        S = [k for k in S if 2 <= k <= n//2]
        if not S: continue
        s = len(S)
        NS = set()
        for k in S:
            NS |= adj[k]
        actual = len(NS)

        # Product bound (only valid for coprime!)
        prod_comp = 1.0
        for k in S:
            prod_comp *= (1 - 1/k)
        product_bound_N = M * (1 - prod_comp)

        # Actual complement
        comp = M - actual
        predicted_comp = M * prod_comp

        print(f"\n  {name}: S={S}")
        print(f"    |N|={actual}, s={s}, ratio={actual/s:.3f}")
        print(f"    Product bound (if coprime): N >= {product_bound_N:.1f}")
        print(f"    Actual complement: {comp}, predicted (coprime): {predicted_comp:.1f}")

        # Check pairwise coprimality
        coprime_pairs = sum(1 for i in range(s) for j in range(i+1,s) if gcd(S[i],S[j])==1)
        total_pairs = s*(s-1)//2
        print(f"    Coprime pairs: {coprime_pairs}/{total_pairs}")

def explore_lcm_bound(n, eps=0.05):
    """
    Key idea: |N_H(S)| >= sum_{k in S} M/k - sum_{i<j} M/lcm(k_i,k_j) + ...

    For the LOWER bound, we can use:
    |N_H(S)| >= sum M/k - sum M/lcm(k_i,k_j)   (Bonferroni truncation)

    This gives: |N_H(S)| >= M * (sum 1/k - sum 1/lcm(k_i,k_j))

    For S near n/2: sum 1/k ~ s * 2/n, sum 1/lcm ~ s(s-1)/2 * ??

    The lcm of two elements near n/2: if gcd(k_i,k_j)=g, then
    lcm = k_i*k_j/g ~ (n/2)^2 / g.

    So sum 1/lcm ~ (s choose 2) * 4g/(n^2) where g is the average gcd.

    For random elements near n/2: average gcd ~ 6/pi^2 * n/4 ??? No.
    Average gcd of two random numbers up to N is about N * zeta(2)^{-1} ??? No.

    Actually E[gcd(a,b)] = sum_{d=1}^{N} d * P[d | gcd] = sum d * (1/d)^2 * ...
    For a,b ~ n/2: E[gcd] ~ sum_{d=1}^{n/2} phi(d)/d * ... this is tricky.

    Let me just compute it.
    """
    L = target_L(n, eps)
    M = L - n
    V = list(range(2, n//2 + 1))

    adj = {}
    for k in V:
        adj[k] = targets_H(k, n, L)

    print(f"\n=== BONFERRONI / LCM ANALYSIS (n={n}) ===")

    # For consecutive S near n/2
    for s in [2, 3, 5, 10, 20]:
        a = n//2 - s + 1
        S = list(range(a, a + s))
        S = [k for k in S if 2 <= k <= n//2]
        if len(S) != s: continue

        NS = set()
        for k in S:
            NS |= adj[k]
        actual = len(NS)

        # Bonferroni lower bound (first two terms)
        E1 = sum(len(adj[k]) for k in S)
        E2 = 0
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                E2 += len(adj[S[i]] & adj[S[j]])

        bonf = E1 - E2

        # Also compute: sum 1/k, sum 1/lcm
        sum_inv_k = sum(1/k for k in S)
        sum_inv_lcm = sum(1/(S[i]*S[j]//gcd(S[i],S[j])) for i in range(len(S)) for j in range(i+1,len(S)))

        print(f"  s={s}, a={a}: |N|={actual}, s={s}, ratio={actual/s:.3f}")
        print(f"    E1={E1}, E2={E2}, Bonf={bonf}, need>={s}")
        print(f"    sum(1/k)={sum_inv_k:.5f}, sum(1/lcm)={sum_inv_lcm:.8f}")
        print(f"    M*sum(1/k)={M*sum_inv_k:.1f}, M*sum(1/lcm)={M*sum_inv_lcm:.3f}")
        print(f"    Bonf_analytic = M*(sum1/k - sum1/lcm) = {M*(sum_inv_k - sum_inv_lcm):.1f}")

if __name__ == '__main__':
    for n in [500, 1000]:
        explore_ratio_by_size(n, num_samples=1000)
        explore_interval_telescoping(n)
        explore_general_complement(n)
        explore_lcm_bound(n)
