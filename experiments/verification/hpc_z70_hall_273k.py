#!/usr/bin/env python3
"""
ERDŐS 710 — Z70: Global Hall verification at n=273582

At n=273582, Z69 found α_exact(I₁₅) = 0.812 < 1 (per-interval Hall FAILS).
This script checks whether GLOBAL Hall still holds via Hopcroft-Karp.
"""

import math, time, sys
from collections import deque

C_TARGET = 2 / math.e**0.5
EPS = 0.05

def log(msg):
    print(msg); sys.stdout.flush()

def compute_params(n):
    ln_n = math.log(n)
    ln_ln_n = math.log(ln_n) if ln_n > 1 else 0.1
    L = (C_TARGET + EPS) * n * math.sqrt(ln_n / ln_ln_n)
    M = n + L
    B = int(math.sqrt(M))
    return L, M, B

def sieve_primes(limit):
    if limit < 2: return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            for mult in range(p*p, limit + 1, p):
                sieve[mult] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def get_smooth_numbers_fast(B, lo, hi):
    if hi <= lo: return []
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

def hopcroft_karp(n_left, n_right, adj):
    match_left = [-1] * n_left
    match_right = [-1] * n_right
    dist = [0] * n_left

    def bfs():
        queue = deque()
        for u in range(n_left):
            if match_left[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = float('inf')
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                w = match_right[v]
                if w == -1:
                    found = True
                elif dist[w] == float('inf'):
                    dist[w] = dist[u] + 1
                    queue.append(w)
        return found

    def dfs(u):
        for v in adj[u]:
            w = match_right[v]
            if w == -1 or (dist[w] == dist[u] + 1 and dfs(w)):
                match_left[u] = v
                match_right[v] = u
                return True
        dist[u] = float('inf')
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if match_left[u] == -1:
                if dfs(u):
                    matching += 1
    return matching

def main():
    test_ns = [273582, 274582, 275582]

    for n in test_ns:
        log(f"\n{'='*60}")
        log(f"  n = {n}")
        log(f"{'='*60}")

        L, M, B = compute_params(n)
        n_half = n // 2
        nL = int(n + L)
        delta = 2 * M / n - 1

        log(f"  δ = {delta:.3f}, B = {B}, L = {int(L)}")

        t1 = time.time()
        S_plus = get_smooth_numbers_fast(B, B, n_half)
        H_smooth = get_smooth_numbers_fast(B, n, nL)
        log(f"  |S₊| = {len(S_plus)}, |H_smooth| = {len(H_smooth)}")
        log(f"  Sieve time: {time.time()-t1:.1f}s")

        H_set = set(H_smooth)
        s_idx = {k: i for i, k in enumerate(S_plus)}
        h_idx = {h: i for i, h in enumerate(H_smooth)}
        n_left = len(S_plus)
        n_right = len(H_smooth)

        t1 = time.time()
        adj = [[] for _ in range(n_left)]
        total_edges = 0
        for k in S_plus:
            u = s_idx[k]
            lo_mult = n // k + 1
            hi_mult = nL // k
            for m in range(lo_mult, hi_mult + 1):
                h = k * m
                if h in H_set:
                    adj[u].append(h_idx[h])
                    total_edges += 1
        log(f"  |E| = {total_edges}, graph time: {time.time()-t1:.1f}s")

        t1 = time.time()
        matching = hopcroft_karp(n_left, n_right, adj)
        dt_hk = time.time() - t1

        if matching >= n_left:
            log(f"  GLOBAL HALL: ✓ PASS — matching = {matching} = |S₊|")
        else:
            log(f"  GLOBAL HALL: ✗ FAIL — matching = {matching} < |S₊| = {n_left}")
            log(f"  Deficiency = {n_left - matching}")
        log(f"  HK time: {dt_hk:.1f}s")

    log(f"\nDone.")

if __name__ == '__main__':
    main()
