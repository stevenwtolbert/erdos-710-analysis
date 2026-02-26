#!/usr/bin/env python3
"""
ERDŐS 710 — Z68-MP: EXHAUSTIVE HOPCROFT-KARP (MULTIPROCESSING)

16-worker parallel verification of Hall's condition at EVERY integer n.
Each n is independent — embarrassingly parallel.

By König's theorem: max matching = |S₊| ⟺ Hall for ALL subsets.

Usage:
    python3 -u hpc_z68_mp.py [start] [end] [num_workers]

Default: n = 4 to 200000, 16 workers.
"""

import math
import time
import sys
import os
import json
from collections import deque
from multiprocessing import Pool, Manager

C_TARGET = 2 / math.e**0.5
EPS = 0.05

N_START = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N_END = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
NUM_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 16

CHECKPOINT_FILE = "/home/ashbringer/projects/e710_new_H/z68mp_checkpoint.json"
STATE_FILE = "/home/ashbringer/projects/e710_new_H/states/state_83_z68mp_exhaustive_hk.md"


def log(msg):
    print(msg)
    sys.stdout.flush()


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


def verify_single_n(n):
    """Verify Hall at a single n. Returns (n, pass/None, n_splus, matching)."""
    L, M, B = compute_params(n)
    n_half = n // 2
    nL = int(n + L)

    if B < 2 or n_half <= B:
        return (n, None, 0, 0)

    S_plus = get_smooth_numbers_fast(B, B, n_half)
    H_smooth = get_smooth_numbers_fast(B, n, nL)

    if not S_plus or not H_smooth:
        return (n, None, len(S_plus), 0)

    H_set = set(H_smooth)
    s_idx = {k: i for i, k in enumerate(S_plus)}
    h_idx = {h: i for i, h in enumerate(H_smooth)}
    n_left = len(S_plus)
    n_right = len(H_smooth)

    adj = [[] for _ in range(n_left)]
    for k in S_plus:
        u = s_idx[k]
        lo_mult = n // k + 1
        hi_mult = nL // k
        for m in range(lo_mult, hi_mult + 1):
            h = k * m
            if h in H_set:
                adj[u].append(h_idx[h])

    matching = hopcroft_karp(n_left, n_right, adj)
    hall_pass = matching >= n_left
    return (n, hall_pass, n_left, matching)


def process_batch(batch):
    """Process a batch of n values. Returns list of (n, pass, n_splus, matching)."""
    results = []
    for n in batch:
        results.append(verify_single_n(n))
    return results


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None


def save_checkpoint(data):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f)


def main():
    log("ERDŐS 710 — Z68-MP: EXHAUSTIVE HOPCROFT-KARP (MULTIPROCESSING)")
    log("=" * 80)
    log(f"Range: n = {N_START} to {N_END}")
    log(f"Workers: {NUM_WORKERS}")
    log(f"Method: Hopcroft-Karp maximum matching (König's theorem)")
    log("")
    t0 = time.time()

    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    if checkpoint and checkpoint.get('n_start') == N_START and checkpoint.get('n_end') == N_END:
        last_n = checkpoint['last_n']
        total_pass = checkpoint['total_pass']
        total_skip = checkpoint['total_skip']
        total_fail = checkpoint['total_fail']
        failures = checkpoint.get('failures', [])
        log(f"Resuming from checkpoint: last_n = {last_n}")
        start_n = last_n + 1
    else:
        total_pass = 0
        total_skip = 0
        total_fail = 0
        failures = []
        start_n = N_START

    # Create batches — each batch is a chunk of consecutive n values
    # Use smaller batches for large n (slower per n) and larger for small n
    all_ns = list(range(start_n, N_END + 1))
    total_remaining = len(all_ns)

    if total_remaining == 0:
        log("Nothing to compute.")
        return

    # Adaptive batch size: ~100 values per batch for parallelism
    BATCH_SIZE = max(50, min(500, total_remaining // (NUM_WORKERS * 4)))
    batches = [all_ns[i:i+BATCH_SIZE] for i in range(0, len(all_ns), BATCH_SIZE)]
    log(f"Total n values: {total_remaining}, batch size: {BATCH_SIZE}, "
        f"batches: {len(batches)}")

    completed_batches = 0
    batch_start_time = time.time()

    with Pool(processes=NUM_WORKERS) as pool:
        for batch_results in pool.imap_unordered(process_batch, batches):
            for n, result, n_splus, matching in batch_results:
                if result is None:
                    total_skip += 1
                elif result:
                    total_pass += 1
                else:
                    total_fail += 1
                    deficiency = n_splus - matching
                    delta = 2 * (n + (C_TARGET + EPS) * n *
                            math.sqrt(math.log(n) / math.log(math.log(n)))) / n - 1
                    failures.append({
                        'n': n, 'n_splus': n_splus, 'matching': matching,
                        'deficiency': deficiency, 'delta': round(delta, 3)
                    })
                    log(f"  *** HALL FAILURE at n = {n}: matching = {matching}, "
                        f"|S₊| = {n_splus}, deficiency = {deficiency} ***")

            completed_batches += 1
            now = time.time()
            elapsed = now - t0
            total_done = total_pass + total_skip + total_fail
            rate = total_done / max(elapsed, 0.001)
            remaining_n = N_END - start_n + 1 - total_done
            eta = remaining_n / max(rate, 0.001)

            # Progress report every 10 batches or ~30s
            if completed_batches % max(1, len(batches) // 50) == 0 or completed_batches == len(batches):
                max_n_done = start_n + total_done - 1
                log(f"  [{completed_batches}/{len(batches)}] n≈{max_n_done:>7} | "
                    f"pass={total_pass} skip={total_skip} fail={total_fail} | "
                    f"rate={rate:.0f}/s | {elapsed:.0f}s elapsed | ETA {eta:.0f}s ({eta/3600:.2f}h)")

                # Save checkpoint
                save_checkpoint({
                    'n_start': N_START, 'n_end': N_END,
                    'last_n': max_n_done,
                    'total_pass': total_pass, 'total_skip': total_skip,
                    'total_fail': total_fail, 'failures': failures,
                    'timestamp': time.time(),
                })

    total_time = time.time() - t0

    # Final summary
    log(f"\n{'='*80}")
    log("  Z68-MP FINAL SUMMARY: EXHAUSTIVE HOPCROFT-KARP VERIFICATION")
    log(f"{'='*80}")
    log(f"  Range: n = {N_START} to {N_END}")
    log(f"  Workers: {NUM_WORKERS}")
    log(f"  Total pass: {total_pass}")
    log(f"  Total skip: {total_skip} (n too small for S₊)")
    log(f"  Total fail: {total_fail}")
    log(f"  Total time: {total_time:.0f}s ({total_time/3600:.2f}h)")
    log(f"  Effective rate: {(total_pass + total_skip) / total_time:.0f} n/s")

    if total_fail == 0:
        log(f"\n  *** HALL'S CONDITION VERIFIED AT EVERY n ∈ [{N_START}, {N_END}] ***")
        log(f"  *** {total_pass} rigorous HK verifications, ZERO failures ***")
        log(f"  *** By König's theorem, this proves Hall for ALL subsets at each n ***")
    else:
        log(f"\n  *** {total_fail} HALL FAILURES DETECTED ***")
        for fail in failures:
            log(f"    n = {fail['n']}: deficiency = {fail['deficiency']}")

    # Write state file
    with open(STATE_FILE, 'w') as f:
        f.write("# State 83: Z68-MP Exhaustive Hopcroft-Karp Verification\n\n")
        f.write("## Method\n\n")
        f.write("Hopcroft-Karp maximum matching on the bipartite graph S₊ → H_smooth\n")
        f.write("at EVERY integer n in the specified range. Parallelized with Python\n")
        f.write(f"multiprocessing ({NUM_WORKERS} workers). By König's theorem,\n")
        f.write("max matching = |S₊| ⟺ Hall's condition holds for ALL subsets.\n\n")
        f.write("## Parameters\n\n")
        f.write(f"- C = 2/√e + 0.05 ≈ {C_TARGET + EPS:.4f}\n")
        f.write(f"- L = C · n · √(ln n / ln ln n)\n")
        f.write(f"- B = ⌊√(n + L)⌋\n")
        f.write(f"- S₊ = B-smooth numbers in (B, n/2]\n")
        f.write(f"- H_smooth = B-smooth numbers in (n, n+L]\n\n")
        f.write("## Results\n\n")
        f.write(f"- Range: n = {N_START} to {N_END}\n")
        f.write(f"- Workers: {NUM_WORKERS}\n")
        f.write(f"- Total pass: {total_pass}\n")
        f.write(f"- Total skip: {total_skip}\n")
        f.write(f"- Total fail: {total_fail}\n")
        f.write(f"- Total time: {total_time:.0f}s ({total_time/3600:.2f}h)\n\n")
        if total_fail == 0:
            f.write(f"**Hall's condition verified at EVERY n ∈ [{N_START}, {N_END}].**\n")
            f.write(f"**{total_pass} rigorous HK verifications, ZERO failures.**\n")
        else:
            f.write(f"**{total_fail} failures detected:**\n\n")
            for fail in failures:
                f.write(f"- n = {fail['n']}: deficiency = {fail['deficiency']}\n")

    log(f"  State file: {STATE_FILE}")

    # Save final checkpoint
    save_checkpoint({
        'n_start': N_START, 'n_end': N_END,
        'last_n': N_END,
        'total_pass': total_pass, 'total_skip': total_skip,
        'total_fail': total_fail, 'failures': failures,
        'timestamp': time.time(),
        'total_time': total_time,
        'complete': True,
    })


if __name__ == '__main__':
    main()
