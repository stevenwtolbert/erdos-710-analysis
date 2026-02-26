#!/usr/bin/env python3
"""
Figure: FMC Sawtooth — Σ 1/α_j with J-transition shockwaves

The signature plot of the Erdős 710 investigation.  Shows:
- The sawtooth pattern of Σ 1/CS_ref,j vs n
- The companion Σ 1/α_j (greedy lower bound)
- J-transition shockwaves (vertical lines where #intervals changes)
- Peak envelope (red dashed) tracking the rising maxima
- Green margin band between the peak and the threshold 1.0
- Bottom panel: δ(n) and J(n)

Data from hpc_z65_sawtooth_plot.py (461 points, n ∈ [100, 300000]).
"""

import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_data(csv_path):
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'n': int(row['n']),
                'J': int(row['J']),
                'sum_1_alpha': float(row['sum_1_alpha']),
                'sum_1_csref': float(row['sum_1_csref']),
                'delta': float(row['delta']),
            })
    return data


def main():
    csv_path = '../../data_z65_sawtooth.csv'
    data = load_data(csv_path)
    print(f"Loaded {len(data)} data points")

    ns = [d['n'] for d in data]
    csrefs = [d['sum_1_csref'] for d in data]
    alphas = [d['sum_1_alpha'] for d in data]
    Js = [d['J'] for d in data]
    deltas = [d['delta'] for d in data]

    # Find J-transitions
    transitions = []
    for i in range(1, len(data)):
        if Js[i] != Js[i-1]:
            transitions.append(i)

    # Find peak per J-regime (contiguous runs of same J)
    peaks_n, peaks_csref = [], []
    current_J_start = 0
    for i in range(1, len(data) + 1):
        if i == len(data) or Js[i] != Js[current_J_start]:
            best_idx = current_J_start
            for j in range(current_J_start, i):
                if csrefs[j] > csrefs[best_idx]:
                    best_idx = j
            peaks_n.append(ns[best_idx])
            peaks_csref.append(csrefs[best_idx])
            current_J_start = i

    # === Two-panel figure ===
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), height_ratios=[3, 1],
        sharex=True, gridspec_kw={'hspace': 0.08}
    )

    # ---- Top panel: FMC sum ----
    ax1.plot(ns, csrefs, 'b-', linewidth=0.8, alpha=0.9, zorder=2,
             label=r'$\sum_j 1/\mathrm{CS}_{\mathrm{ref},j}$')
    ax1.plot(ns, alphas, 'g-', linewidth=0.5, alpha=0.6, zorder=1,
             label=r'$\sum_j 1/\alpha_j$ (greedy)')

    # Peak envelope
    ax1.plot(peaks_n, peaks_csref, 'r--', linewidth=1.2, alpha=0.7,
             label='Peak envelope', zorder=3)
    ax1.scatter(peaks_n, peaks_csref, c='red', s=20, zorder=4)

    # J-transition verticals
    for idx in transitions:
        ax1.axvline(x=ns[idx], color='gray', linewidth=0.3, alpha=0.5)

    # Critical threshold
    ax1.axhline(y=1.0, color='red', linewidth=1.5, linestyle=':', alpha=0.8,
                label='Threshold = 1')

    # Margin shading
    ax1.fill_between(ns, csrefs, 1.0, alpha=0.08, color='green',
                     label=f'Margin ({(1 - max(csrefs)) * 100:.1f}%)')

    # Global max annotation
    max_idx = csrefs.index(max(csrefs))
    ax1.annotate(
        f'Global max = {csrefs[max_idx]:.3f}\n(n = {ns[max_idx]:,})',
        xy=(ns[max_idx], csrefs[max_idx]),
        xytext=(ns[max_idx] * 2.0, csrefs[max_idx] + 0.04),
        fontsize=9, ha='left',
        arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9)
    )

    # J labels at transitions
    J_label_positions = {}
    for idx in transitions:
        J_new = Js[idx]
        if J_new not in J_label_positions:
            J_label_positions[J_new] = ns[idx]
    for J_val, x_pos in J_label_positions.items():
        ax1.annotate(f'J={J_val}', xy=(x_pos, 0.42), fontsize=7,
                     color='gray', ha='center', va='bottom')

    ax1.set_ylabel(
        r'$\sum_j 1/\alpha_j$  or  $\sum_j 1/\mathrm{CS}_{\mathrm{ref},j}$',
        fontsize=12)
    ax1.set_ylim(0.38, 1.05)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_title(
        r'FMC Sum $\sum 1/\alpha_j$: Sawtooth Pattern with $J$-Transition Shockwaves',
        fontsize=13)
    ax1.grid(True, alpha=0.3)

    # ---- Bottom panel: δ(n) and J(n) ----
    color_delta = 'tab:orange'
    ax2.plot(ns, deltas, color=color_delta, linewidth=1.0)
    ax2.set_ylabel(r'$\delta(n)$', color=color_delta, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color_delta)
    ax2.set_xlabel(r'$n$', fontsize=12)
    ax2.set_xscale('log')

    ax2_twin = ax2.twinx()
    ax2_twin.step(ns, Js, color='purple', linewidth=0.8, alpha=0.7,
                  where='post')
    ax2_twin.set_ylabel('$J$ (number of intervals)', color='purple',
                        fontsize=11)
    ax2_twin.tick_params(axis='y', labelcolor='purple')

    ax2.grid(True, alpha=0.3)

    fig.subplots_adjust(left=0.08, right=0.92, top=0.94, bottom=0.08, hspace=0.08)
    fig.savefig('fig_sawtooth.png', dpi=200)
    fig.savefig('fig_sawtooth.pdf')
    print("Saved fig_sawtooth.{png,pdf}")


if __name__ == '__main__':
    main()
