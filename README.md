# Computational and Analytic Approaches to the Erdős–Pomerance Divisible Injection Problem

**Author:** Steven William Tolbert

This repository contains the paper, computational experiments, and datasets from a systematic investigation of [Erdős Problem #710](https://www.erdosproblems.com/710): determining the minimum value of $f(n)$, the smallest $L$ such that every integer in $\{1, \ldots, n\}$ can be injectively mapped to a multiple in $(n, n+L]$.

## The Problem

The best known bounds are:

$$\left(\tfrac{2}{\sqrt{e}} + o(1)\right) n \sqrt{\frac{\ln n}{\ln \ln n}} \;\leq\; f(n) \;\leq\; (1.7398\ldots + o(1))\, n \sqrt{\ln n}$$

The lower bound is due to Erdős and the upper bound to Erdős and Pomerance (1980). Closing the $\sqrt{\ln \ln n}$ gap between them remains open.

## What's Here

We recast the problem as finding a perfect matching in a bipartite divisibility graph $G_n = (V, H, E)$ and systematically attempted **43 distinct analytic approaches** to prove Hall's condition for all $n$. All failed for the global case. This repository documents every approach, why it fails, and what computational phenomena we discovered along the way.

### Exhaustive Verification

Hall's condition (and hence a valid injection) holds for **every** $n \in [4, 10^6]$, verified via Hopcroft–Karp. Zero failures.

### The Critical Gap

No analytic argument we found extends the finite verification to $n \to \infty$. The structural obstruction is that the worst-case subset spans 48–53% of all vertices across every dyadic interval, with expansion ratio barely exceeding 1 — leaving no room for any of the standard combinatorial, probabilistic, or analytic methods to gain leverage.

## Repository Structure

```
paper/                          # LaTeX source and compiled PDF
  erdos_pomerance.tex           # Main file (inputs section files)
  section_01_introduction.tex   # Sections 1–12
  ...
  section_12_conclusion.tex
  erdos_pomerance.pdf           # Compiled paper (~40 pages)

experiments/                    # 61 Python/C scripts organized by method
  verification/                 # Hopcroft–Karp exhaustive verification (6 scripts)
  cauchy_schwarz/               # CS variants: global, per-interval, weighted,
                                #   filtered, truncated, variable-constant (11 scripts)
  fractional_matching/          # FMC, uniform/nonuniform weights, Sinkhorn,
                                #   CLP, three-block (11 scripts)
  partition/                    # Dyadic, √2, fine partition, degree bands (9 scripts)
  graph_theoretic/              # Spectral, Turán, multiplicative energy,
                                #   descent chains (4 scripts)
  probabilistic/                # LLL, semi-random nibble (3 scripts)
  variable_constant/            # 2D C(n) sweep experiments (2 scripts)
  initial_attempt/              # First-pass Shearer bound and CS analysis (5 scripts)
  figures/                      # Figure-generating scripts + output (10 scripts)
  sieve/                        # (Sieve scripts in main project, not yet copied)

datasets/
  z65_sawtooth.csv              # 461-point sawtooth oscillation data (n ∈ [100, 300000])
```

## Key Figures

| Figure | Description |
|--------|-------------|
| `fig_sawtooth` | FMC sum $\sum 1/\alpha_j$ sawtooth pattern with $J$-transition shockwaves |
| `fig_degree_distribution` | Vertex degree $d(k) \approx M/k$ across several $n$ |
| `fig_global_cs_oscillation` | Cauchy–Schwarz ratio oscillating near 1.0, never converging |
| `fig_variable_constant` | 2D sweep of CS ratio vs $(n, C_{\text{mult}})$ showing stabilization |
| `fig_nibble_residual` | Semi-random nibble residual fraction vs $n$ |
| `fig_approach_summary` | Visual heatmap of all 43 approaches by category and outcome |

## Approaches Tried (Summary)

| Category | Count | Outcome |
|----------|-------|---------|
| Cauchy–Schwarz variants | 6 | Per-interval works; global ratio 0.988–1.03 |
| Matching / fractional | 8 | FMC theorem proved; $\sum 1/\alpha_j > 1$ for greedy partition |
| Probabilistic (LLL, nibble) | 5 | $P \cdot e(D+1) \approx 10^4$; residual unmatchable |
| Sieve / inclusion-exclusion | 4 | Signs diverge at all orders |
| Graph-theoretic / spectral | 5 | Spectral gap too small; overlap 87–93% |
| Partition strategies | 6+ | Cross-interval overlap defeats all decompositions |
| Variable-constant | 2 | CS stabilizes; no divergence |
| Other (GCD, Ford cap, etc.) | 7+ | Various structural obstructions |

See Section 12 of the paper for the full table with one-line failure reasons.

## Requirements

- **Paper:** Any LaTeX distribution with `amsmath`, `amssymb`, `hyperref`, `geometry`, `booktabs`
- **Experiments:** Python 3.8+, `numpy`, `matplotlib`. The C verifier (`hpc_z68_hk.c`) requires GCC with OpenMP.

## Building the Paper

```bash
cd paper
pdflatex erdos_pomerance.tex
pdflatex erdos_pomerance.tex   # twice for cross-references
```

## Regenerating Figures

```bash
cd experiments/figures
python3 fig_sawtooth.py
python3 fig_degree_distribution.py
python3 fig_global_cs_oscillation.py
# etc.
```

Figures are saved as both PNG (200 dpi) and PDF.

## Citation

If you find this work useful for your research on Erdős Problem #710 or related divisibility problems:

```
Steven Tolbert, "Computational and analytic approaches to the Erdős–Pomerance
divisible injection problem," 2026. Available: https://github.com/stevenwtolbert/erdos-710-analysis
```

## License

This work is provided for research purposes. See the repository for details.
