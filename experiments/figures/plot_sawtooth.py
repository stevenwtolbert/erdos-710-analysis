#!/usr/bin/env python3
"""
ERDŐS 710 — Sawtooth Plot: Σ 1/CS_ref vs n

Beautiful visualization showing:
- The sawtooth pattern of Σ 1/CS_ref
- J-transition shockwaves
- The peak envelope
- The critical threshold at 1.0
- Comparison with Σ 1/α

For the paper (Figure X).
"""

import csv
import math
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available, generating ASCII plot instead")


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


def plot_matplotlib(data, output_path):
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

    # Find peak per J-regime
    peaks_n = []
    peaks_csref = []
    current_J_start = 0
    for i in range(1, len(data) + 1):
        if i == len(data) or Js[i] != Js[current_J_start]:
            # End of current J-regime
            best_idx = current_J_start
            for j in range(current_J_start, i):
                if csrefs[j] > csrefs[best_idx]:
                    best_idx = j
            peaks_n.append(ns[best_idx])
            peaks_csref.append(csrefs[best_idx])
            current_J_start = i

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={'hspace': 0.08})

    # Main plot: Σ 1/CS_ref
    ax1.plot(ns, csrefs, 'b-', linewidth=0.8, label=r'$\sum 1/\mathrm{CS}_{\mathrm{ref},j}$',
             alpha=0.9, zorder=2)
    ax1.plot(ns, alphas, 'g-', linewidth=0.5, label=r'$\sum 1/\alpha_j$',
             alpha=0.6, zorder=1)

    # Peak envelope
    ax1.plot(peaks_n, peaks_csref, 'r--', linewidth=1.2, alpha=0.7,
             label='Peak envelope', zorder=3)
    ax1.scatter(peaks_n, peaks_csref, c='red', s=20, zorder=4)

    # Mark J-transitions
    for idx in transitions:
        ax1.axvline(x=ns[idx], color='gray', linewidth=0.3, alpha=0.5)

    # Critical threshold
    ax1.axhline(y=1.0, color='red', linewidth=1.5, linestyle=':', alpha=0.8,
                label='Threshold = 1')

    # Global maximum annotation
    max_idx = csrefs.index(max(csrefs))
    ax1.annotate(f'Global max = {csrefs[max_idx]:.3f}\n(n = {ns[max_idx]:,})',
                xy=(ns[max_idx], csrefs[max_idx]),
                xytext=(ns[max_idx] * 1.5, csrefs[max_idx] + 0.03),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    # Margin annotation
    ax1.fill_between(ns, csrefs, 1.0, alpha=0.08, color='green',
                     label=f'Margin ({(1-max(csrefs))*100:.1f}%)')

    ax1.set_ylabel(r'$\sum_j 1/\alpha_j$ or $\sum_j 1/\mathrm{CS}_{\mathrm{ref},j}$',
                   fontsize=12)
    ax1.set_ylim(0.4, 1.05)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_title(r'FMC Sum $\sum 1/\alpha_j$: Sawtooth Pattern with J-Transition Shockwaves',
                  fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Add J labels at transitions
    J_label_positions = {}
    for idx in transitions:
        J_new = Js[idx]
        if J_new not in J_label_positions:
            J_label_positions[J_new] = ns[idx]
    for J_val, x_pos in J_label_positions.items():
        ax1.annotate(f'J={J_val}', xy=(x_pos, 0.42), fontsize=7,
                    color='gray', ha='center', va='bottom')

    # Bottom plot: δ and J
    color_delta = 'tab:orange'
    ax2.plot(ns, deltas, color=color_delta, linewidth=1.0)
    ax2.set_ylabel(r'$\delta(n)$', color=color_delta, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color_delta)
    ax2.set_xlabel(r'$n$', fontsize=12)
    ax2.set_xscale('log')

    ax2_twin = ax2.twinx()
    ax2_twin.step(ns, Js, color='purple', linewidth=0.8, alpha=0.7, where='post')
    ax2_twin.set_ylabel('J (intervals)', color='purple', fontsize=11)
    ax2_twin.tick_params(axis='y', labelcolor='purple')

    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Plot saved: {output_path}")


def plot_ascii(data):
    """Fallback ASCII plot."""
    ns = [d['n'] for d in data]
    csrefs = [d['sum_1_csref'] for d in data]
    Js = [d['J'] for d in data]

    width = 80
    height = 25
    min_y, max_y = 0.4, 1.05

    # Sample data points for ASCII
    step = max(1, len(data) // width)
    sampled = data[::step]

    print("\n  Σ 1/CS_ref vs n (log scale)")
    print("  " + "=" * (width + 2))

    for row in range(height, -1, -1):
        y = min_y + (max_y - min_y) * row / height
        label = f"{y:.2f}" if row % 5 == 0 else "    "
        line = f"  {label:>5}|"
        for d in sampled:
            val = d['sum_1_csref']
            y_pos = (val - min_y) / (max_y - min_y) * height
            if abs(y_pos - row) < 0.5:
                line += '*'
            elif abs(y - 1.0) < 0.01:
                line += '-'
            else:
                line += ' '
        print(line)

    # X-axis
    print("  " + " " * 6 + "+" + "-" * len(sampled))
    x_labels = "  " + " " * 6
    for i, d in enumerate(sampled):
        if i % (len(sampled) // 5) == 0:
            x_labels += f"{d['n']:>6}"
        else:
            x_labels += " "
    print(x_labels)


def main():
    csv_path = "/home/ashbringer/projects/e710_new_H/data_z65_sawtooth.csv"

    try:
        data = load_data(csv_path)
    except FileNotFoundError:
        print(f"  Data file not found: {csv_path}")
        print("  Run hpc_z65_sawtooth_plot.py first to generate data.")
        sys.exit(1)

    print(f"  Loaded {len(data)} data points from {csv_path}")

    if HAS_MPL:
        output_path = "/home/ashbringer/projects/e710_new_H/manuscript/figures/sawtooth_fmc_sum.png"
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plot_matplotlib(data, output_path)
    else:
        plot_ascii(data)


if __name__ == '__main__':
    main()
