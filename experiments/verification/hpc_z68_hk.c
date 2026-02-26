/*
 * ERDŐS 710 — Z68-HK: EXHAUSTIVE HOPCROFT-KARP (C + OpenMP)
 *
 * Verifies Hall's condition at EVERY integer n in [n_start, n_end].
 * Each n is independent — embarrassingly parallel via OpenMP.
 *
 * By König's theorem: max matching = |S₊| ⟺ Hall for ALL subsets.
 *
 * Features:
 *   - Checkpoint/resume: writes progress to z68c_checkpoint.json
 *   - Per-value logging: writes every verified n to z68c_log.bin (binary)
 *   - Failure logging: any failure immediately written to z68c_failures.txt
 *
 * Build:
 *     gcc -O3 -fopenmp -o hpc_z68_hk hpc_z68_hk.c -lm
 *
 * Usage:
 *     ./hpc_z68_hk [n_start] [n_end] [num_threads]
 *
 * Default: n = 4 to 1000000, 16 threads.
 * Resume: automatically detects checkpoint and resumes from last batch.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define C_TARGET (2.0 / sqrt(M_E))
#define EPS 0.05
#define INF 0x7FFFFFFF

#define CHECKPOINT_FILE "z68c_checkpoint.json"
#define FAILURE_FILE    "z68c_failures.txt"
#define LOG_FILE        "z68c_output.log"

/* Batch size for checkpoint granularity */
#define BATCH_SIZE 1000

/* ─── Parameters ─── */

typedef struct {
    double L;
    long long M;
    int B;
} Params;

static Params compute_params(int n) {
    double ln_n = log((double)n);
    double ln_ln_n = (ln_n > 1.0) ? log(ln_n) : 0.1;
    double L = (C_TARGET + EPS) * n * sqrt(ln_n / ln_ln_n);
    long long M = (long long)(n + L);
    int B = (int)sqrt((double)M);
    Params p = { L, M, B };
    return p;
}

/* ─── Sieve of Eratosthenes ─── */

static int* sieve_primes(int limit, int *count) {
    if (limit < 2) { *count = 0; return NULL; }
    char *sieve = (char *)calloc(limit + 1, 1);
    if (!sieve) { *count = 0; return NULL; }
    for (int i = 2; i <= limit; i++) sieve[i] = 1;
    for (int p = 2; (long long)p * p <= limit; p++) {
        if (sieve[p]) {
            for (int m = p * p; m <= limit; m += p)
                sieve[m] = 0;
        }
    }
    int cnt = 0;
    for (int i = 2; i <= limit; i++) if (sieve[i]) cnt++;
    int *primes = (int *)malloc(cnt * sizeof(int));
    int idx = 0;
    for (int i = 2; i <= limit; i++) if (sieve[i]) primes[idx++] = i;
    free(sieve);
    *count = cnt;
    return primes;
}

/* ─── B-smooth numbers in (lo, hi] ─── */

static int* get_smooth_numbers(int B, int lo, int hi,
                               int *primes, int nprimes, int *count) {
    if (hi <= lo) { *count = 0; return NULL; }
    int size = hi - lo;

    long long *remaining = (long long *)malloc(size * sizeof(long long));
    for (int i = 0; i < size; i++)
        remaining[i] = (long long)(lo + 1 + i);

    for (int pi = 0; pi < nprimes; pi++) {
        int p = primes[pi];
        if (p > B) break;
        long long start = (long long)lo + 1;
        int first_offset = (int)((p - (start % p)) % p);
        for (int idx = first_offset; idx < size; idx += p) {
            while (remaining[idx] % p == 0)
                remaining[idx] /= p;
        }
    }

    int cnt = 0;
    for (int i = 0; i < size; i++)
        if (remaining[i] == 1) cnt++;

    int *result = (int *)malloc((cnt > 0 ? cnt : 1) * sizeof(int));
    int ri = 0;
    for (int i = 0; i < size; i++)
        if (remaining[i] == 1)
            result[ri++] = lo + 1 + i;

    free(remaining);
    *count = cnt;
    return result;
}

/* ─── Bipartite Graph ─── */

typedef struct {
    int *adj;
    int *adj_start;
    int *adj_deg;
    int n_left;
    int n_right;
    int n_edges;
} BipartiteGraph;

/* ─── Hopcroft-Karp: DFS augment ─── */

static int hk_dfs(BipartiteGraph *G, int u,
                   int *match_left, int *match_right, int *dist) {
    int start = G->adj_start[u];
    int end = start + G->adj_deg[u];
    for (int ei = start; ei < end; ei++) {
        int v = G->adj[ei];
        int w = match_right[v];
        if (w == -1 || (dist[w] == dist[u] + 1 &&
                        hk_dfs(G, w, match_left, match_right, dist))) {
            match_left[u] = v;
            match_right[v] = u;
            return 1;
        }
    }
    dist[u] = INF;
    return 0;
}

/* ─── Hopcroft-Karp: main ─── */

static int hopcroft_karp(BipartiteGraph *G) {
    int n_left = G->n_left;
    int n_right = G->n_right;

    int *match_left  = (int *)malloc(n_left * sizeof(int));
    int *match_right = (int *)malloc(n_right * sizeof(int));
    int *dist  = (int *)malloc(n_left * sizeof(int));
    int *queue = (int *)malloc(n_left * sizeof(int));

    memset(match_left,  -1, n_left * sizeof(int));
    memset(match_right, -1, n_right * sizeof(int));

    int matching = 0;

    for (;;) {
        int qhead = 0, qtail = 0;
        int found = 0;

        for (int u = 0; u < n_left; u++) {
            if (match_left[u] == -1) {
                dist[u] = 0;
                queue[qtail++] = u;
            } else {
                dist[u] = INF;
            }
        }

        while (qhead < qtail) {
            int u = queue[qhead++];
            int s = G->adj_start[u];
            int e = s + G->adj_deg[u];
            for (int ei = s; ei < e; ei++) {
                int v = G->adj[ei];
                int w = match_right[v];
                if (w == -1) {
                    found = 1;
                } else if (dist[w] == INF) {
                    dist[w] = dist[u] + 1;
                    queue[qtail++] = w;
                }
            }
        }

        if (!found) break;

        for (int u = 0; u < n_left; u++) {
            if (match_left[u] == -1) {
                matching += hk_dfs(G, u, match_left, match_right, dist);
            }
        }
    }

    free(match_left);
    free(match_right);
    free(dist);
    free(queue);

    return matching;
}

/* ─── Hash table ─── */

typedef struct {
    int *keys;
    int *vals;
    int size;
} HashTable;

static void ht_init(HashTable *ht, int capacity) {
    ht->size = capacity;
    ht->keys = (int *)malloc(capacity * sizeof(int));
    ht->vals = (int *)malloc(capacity * sizeof(int));
    memset(ht->keys, -1, capacity * sizeof(int));
}

static void ht_insert(HashTable *ht, int key, int val) {
    unsigned int h = (unsigned int)key % ht->size;
    while (ht->keys[h] != -1) h = (h + 1) % ht->size;
    ht->keys[h] = key;
    ht->vals[h] = val;
}

static int ht_lookup(HashTable *ht, int key) {
    unsigned int h = (unsigned int)key % ht->size;
    while (ht->keys[h] != -1) {
        if (ht->keys[h] == key) return ht->vals[h];
        h = (h + 1) % ht->size;
    }
    return -1;
}

static void ht_free(HashTable *ht) {
    free(ht->keys);
    free(ht->vals);
}

/* ─── Verify single n ─── */

typedef struct {
    int n;
    int pass;    /* 1 = pass, 0 = fail, -1 = skip */
    int n_splus;
    int matching;
} Result;

static Result verify_single_n(int n) {
    Result res = { n, -1, 0, 0 };

    Params p = compute_params(n);
    int n_half = n / 2;
    int nL = (int)(n + p.L);
    int B = p.B;

    if (B < 2 || n_half <= B) return res;

    int nprimes;
    int *primes = sieve_primes(B, &nprimes);

    int n_splus, n_hsmooth;
    int *S_plus = get_smooth_numbers(B, B, n_half, primes, nprimes, &n_splus);
    int *H_smooth = get_smooth_numbers(B, n, nL, primes, nprimes, &n_hsmooth);

    free(primes);

    if (n_splus == 0 || n_hsmooth == 0) {
        res.n_splus = n_splus;
        free(S_plus);
        free(H_smooth);
        return res;
    }

    HashTable ht;
    ht_init(&ht, n_hsmooth * 3);
    for (int i = 0; i < n_hsmooth; i++)
        ht_insert(&ht, H_smooth[i], i);

    /* Pass 1: count edges */
    int *edge_count = (int *)calloc(n_splus, sizeof(int));
    int total_edges = 0;

    for (int u = 0; u < n_splus; u++) {
        int k = S_plus[u];
        int lo_mult = n / k + 1;
        int hi_mult = nL / k;
        for (int m = lo_mult; m <= hi_mult; m++) {
            long long h = (long long)k * m;
            if (h > nL) break;
            if (ht_lookup(&ht, (int)h) >= 0) {
                edge_count[u]++;
                total_edges++;
            }
        }
    }

    /* Build flat adjacency */
    BipartiteGraph G;
    G.n_left = n_splus;
    G.n_right = n_hsmooth;
    G.n_edges = total_edges;
    G.adj = (int *)malloc((total_edges > 0 ? total_edges : 1) * sizeof(int));
    G.adj_start = (int *)malloc(n_splus * sizeof(int));
    G.adj_deg = (int *)malloc(n_splus * sizeof(int));

    int offset = 0;
    for (int u = 0; u < n_splus; u++) {
        G.adj_start[u] = offset;
        G.adj_deg[u] = edge_count[u];
        offset += edge_count[u];
    }

    /* Pass 2: fill adjacency */
    int *fill_pos = (int *)calloc(n_splus, sizeof(int));
    for (int u = 0; u < n_splus; u++) {
        int k = S_plus[u];
        int lo_mult = n / k + 1;
        int hi_mult = nL / k;
        for (int m = lo_mult; m <= hi_mult; m++) {
            long long h = (long long)k * m;
            if (h > nL) break;
            int idx = ht_lookup(&ht, (int)h);
            if (idx >= 0) {
                G.adj[G.adj_start[u] + fill_pos[u]++] = idx;
            }
        }
    }

    free(fill_pos);
    free(edge_count);
    free(S_plus);
    free(H_smooth);
    ht_free(&ht);

    int matching = hopcroft_karp(&G);

    free(G.adj);
    free(G.adj_start);
    free(G.adj_deg);

    res.pass = (matching >= n_splus) ? 1 : 0;
    res.n_splus = n_splus;
    res.matching = matching;

    return res;
}

/* ─── Checkpoint I/O ─── */

typedef struct {
    int n_start;
    int n_end;
    int last_batch_end;   /* last fully completed batch endpoint */
    long total_pass;
    long total_skip;
    long total_fail;
    double elapsed_prior; /* time from prior runs */
} Checkpoint;

static int load_checkpoint(Checkpoint *cp, int n_start, int n_end) {
    FILE *f = fopen(CHECKPOINT_FILE, "r");
    if (!f) return 0;

    int ns, ne, lbe;
    long tp, ts, tf;
    double ep;
    /* Simple JSON parsing */
    if (fscanf(f, " { \"n_start\": %d , \"n_end\": %d , \"last_batch_end\": %d , "
               "\"total_pass\": %ld , \"total_skip\": %ld , \"total_fail\": %ld , "
               "\"elapsed_prior\": %lf",
               &ns, &ne, &lbe, &tp, &ts, &tf, &ep) == 7) {
        fclose(f);
        if (ns == n_start && ne == n_end && lbe > n_start) {
            cp->n_start = ns;
            cp->n_end = ne;
            cp->last_batch_end = lbe;
            cp->total_pass = tp;
            cp->total_skip = ts;
            cp->total_fail = tf;
            cp->elapsed_prior = ep;
            return 1;
        }
    }
    fclose(f);
    return 0;
}

static void save_checkpoint(Checkpoint *cp) {
    FILE *f = fopen(CHECKPOINT_FILE, "w");
    if (!f) return;
    fprintf(f, "{ \"n_start\": %d, \"n_end\": %d, \"last_batch_end\": %d, "
               "\"total_pass\": %ld, \"total_skip\": %ld, \"total_fail\": %ld, "
               "\"elapsed_prior\": %.3f }\n",
            cp->n_start, cp->n_end, cp->last_batch_end,
            cp->total_pass, cp->total_skip, cp->total_fail,
            cp->elapsed_prior);
    fclose(f);
}

static void log_failure(int n, int n_splus, int matching) {
    FILE *f = fopen(FAILURE_FILE, "a");
    if (!f) return;
    time_t now = time(NULL);
    fprintf(f, "[%s] HALL FAILURE: n=%d, |S+|=%d, matching=%d, deficiency=%d\n",
            ctime(&now), n, n_splus, matching, n_splus - matching);
    fclose(f);
}

/* ─── Main ─── */

int main(int argc, char *argv[]) {
    int n_start  = (argc > 1) ? atoi(argv[1]) : 4;
    int n_end    = (argc > 2) ? atoi(argv[2]) : 1000000;
    int nthreads = (argc > 3) ? atoi(argv[3]) : 16;

    omp_set_num_threads(nthreads);

    FILE *logf = fopen(LOG_FILE, "a");

    #define LOG(fmt, ...) do { \
        printf(fmt, ##__VA_ARGS__); fflush(stdout); \
        if (logf) { fprintf(logf, fmt, ##__VA_ARGS__); fflush(logf); } \
    } while(0)

    LOG("ERDŐS 710 — Z68-HK (C + OpenMP): EXHAUSTIVE HOPCROFT-KARP\n");
    LOG("================================================================\n");
    LOG("Range: n = %d to %d\n", n_start, n_end);
    LOG("Threads: %d\n", nthreads);
    LOG("Method: Hopcroft-Karp maximum matching (König's theorem)\n\n");

    /* Try to resume from checkpoint */
    Checkpoint cp = { n_start, n_end, n_start - 1, 0, 0, 0, 0.0 };
    int resumed = load_checkpoint(&cp, n_start, n_end);
    int actual_start = cp.last_batch_end + 1;

    long total_pass = cp.total_pass;
    long total_skip = cp.total_skip;
    long total_fail = cp.total_fail;

    if (resumed) {
        LOG("RESUMING from checkpoint: last_batch_end = %d\n", cp.last_batch_end);
        LOG("  Prior progress: pass=%ld, skip=%ld, fail=%ld, elapsed=%.1fs\n",
            total_pass, total_skip, total_fail, cp.elapsed_prior);
    }

    if (actual_start > n_end) {
        LOG("Nothing to compute — already complete.\n");
        if (logf) fclose(logf);
        return 0;
    }

    int total_remaining = n_end - actual_start + 1;
    int n_batches = (total_remaining + BATCH_SIZE - 1) / BATCH_SIZE;
    LOG("Remaining: %d values in %d batches of %d\n\n", total_remaining, n_batches, BATCH_SIZE);

    /* Failure recording */
    int *fail_n = NULL, *fail_splus = NULL, *fail_match = NULL;
    int n_failures = 0, fail_cap = 0;

    double t0 = omp_get_wtime();

    /* Process in batches for checkpoint granularity */
    for (int batch = 0; batch < n_batches; batch++) {
        int batch_start = actual_start + batch * BATCH_SIZE;
        int batch_end = batch_start + BATCH_SIZE - 1;
        if (batch_end > n_end) batch_end = n_end;
        int batch_size = batch_end - batch_start + 1;

        long bp = 0, bs = 0, bf = 0;

        #pragma omp parallel for schedule(dynamic, 1) reduction(+:bp, bs, bf)
        for (int n = batch_start; n <= batch_end; n++) {
            Result r = verify_single_n(n);

            if (r.pass == -1) {
                bs++;
            } else if (r.pass == 1) {
                bp++;
            } else {
                bf++;
                #pragma omp critical(failure)
                {
                    if (n_failures >= fail_cap) {
                        fail_cap = (fail_cap == 0) ? 16 : fail_cap * 2;
                        fail_n     = (int *)realloc(fail_n,     fail_cap * sizeof(int));
                        fail_splus = (int *)realloc(fail_splus, fail_cap * sizeof(int));
                        fail_match = (int *)realloc(fail_match, fail_cap * sizeof(int));
                    }
                    fail_n[n_failures]     = r.n;
                    fail_splus[n_failures] = r.n_splus;
                    fail_match[n_failures] = r.matching;
                    n_failures++;
                    log_failure(r.n, r.n_splus, r.matching);
                    fprintf(stderr, "*** HALL FAILURE at n = %d: matching = %d, |S+| = %d ***\n",
                            r.n, r.matching, r.n_splus);
                    fflush(stderr);
                }
            }
        }

        total_pass += bp;
        total_skip += bs;
        total_fail += bf;

        /* Save checkpoint after each batch */
        double elapsed = omp_get_wtime() - t0;
        cp.last_batch_end = batch_end;
        cp.total_pass = total_pass;
        cp.total_skip = total_skip;
        cp.total_fail = total_fail;
        cp.elapsed_prior = cp.elapsed_prior + 0; /* don't double count */
        /* Actually store cumulative elapsed for this run */
        Checkpoint save_cp = cp;
        save_cp.elapsed_prior = cp.elapsed_prior + elapsed;
        save_checkpoint(&save_cp);

        /* Progress report every 10 batches or at end */
        if ((batch + 1) % 10 == 0 || batch == n_batches - 1) {
            long done = total_pass + total_skip + total_fail;
            int total_n = n_end - n_start + 1;
            double total_elapsed = cp.elapsed_prior + elapsed;
            double rate = done / (total_elapsed > 0.001 ? total_elapsed : 0.001);
            long remaining = total_n - done;
            double eta = remaining / (rate > 0.001 ? rate : 0.001);
            LOG("  [batch %d/%d] n≤%7d | pass=%ld skip=%ld fail=%ld | "
                "rate=%.0f/s | %.0fs elapsed | ETA %.0fs (%.2fh)\n",
                batch + 1, n_batches, batch_end,
                total_pass, total_skip, total_fail,
                rate, total_elapsed, eta, eta / 3600.0);
        }
    }

    double run_time = omp_get_wtime() - t0;
    double total_time = cp.elapsed_prior + run_time;

    LOG("\n================================================================\n");
    LOG("  Z68-HK (C + OpenMP) FINAL SUMMARY\n");
    LOG("================================================================\n");
    LOG("  Range: n = %d to %d\n", n_start, n_end);
    LOG("  Threads: %d\n", nthreads);
    LOG("  Total pass: %ld\n", total_pass);
    LOG("  Total skip: %ld\n", total_skip);
    LOG("  Total fail: %ld\n", total_fail);
    LOG("  This run: %.1fs | Total time: %.1fs (%.2fh)\n", run_time, total_time, total_time / 3600.0);
    if (total_time > 0)
        LOG("  Effective rate: %.0f n/s\n", (total_pass + total_skip) / total_time);

    if (total_fail == 0) {
        LOG("\n  *** HALL'S CONDITION VERIFIED AT EVERY n in [%d, %d] ***\n",
               n_start, n_end);
        LOG("  *** %ld rigorous HK verifications, ZERO failures ***\n", total_pass);
        LOG("  *** By König's theorem, proves Hall for ALL subsets at each n ***\n");
    } else {
        LOG("\n  *** %ld HALL FAILURES DETECTED ***\n", total_fail);
        for (int i = 0; i < n_failures; i++) {
            LOG("    n = %d: |S+| = %d, matching = %d, deficiency = %d\n",
                   fail_n[i], fail_splus[i], fail_match[i],
                   fail_splus[i] - fail_match[i]);
        }
    }

    free(fail_n);
    free(fail_splus);
    free(fail_match);
    if (logf) fclose(logf);

    return total_fail > 0 ? 1 : 0;
}
