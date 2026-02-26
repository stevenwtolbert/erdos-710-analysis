#!/usr/bin/env python3
"""Figure 10: Visual summary heatmap of 43 approaches by category and outcome."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

categories = [
    'Cauchy-Schwarz',
    'Matching/Fractional',
    'Probabilistic',
    'Sieve/IE',
    'Graph-Theoretic',
    'Partition/Structural',
    'Other',
]

# (approach_name, category_idx, outcome)
# outcome: 'dead' = totally dead, 'partial' = works partially, 'promising' = promising direction
approaches = [
    ('Standard CS', 0, 'dead'),
    ('Per-interval CS', 0, 'partial'),
    ('Weighted CS', 0, 'partial'),
    ('Optimal CS ($C^{-1}d$)', 0, 'partial'),
    ('Filtered CS', 0, 'dead'),
    ('Truncated CS', 0, 'dead'),
    ('Variable-constant CS', 0, 'dead'),
    ('Greedy matching', 1, 'dead'),
    ('Uniform fractional', 1, 'dead'),
    ('Nonuniform fractional', 1, 'dead'),
    ('FMC dyadic', 1, 'dead'),
    ('FMC three-block', 1, 'partial'),
    ('FMC $V_{min}/V_{rest}/S_-$', 1, 'partial'),
    ('Sinkhorn', 1, 'dead'),
    ('CLP factoring', 1, 'dead'),
    (r'$\sqrt{2}$-partition', 1, 'partial'),
    ('Symmetric LLL', 2, 'dead'),
    ('Target-centered LLL', 2, 'dead'),
    ('Janson', 2, 'dead'),
    ('Erdos-Spencer', 2, 'dead'),
    ('Semi-random nibble', 2, 'dead'),
    ('Bonferroni (order 2)', 3, 'dead'),
    ('Bonferroni (higher)', 3, 'dead'),
    ('Product-formula sieve', 3, 'dead'),
    ('Unique multiple sieve', 3, 'dead'),
    ('Standard sieve', 3, 'dead'),
    ('Spectral gap', 4, 'dead'),
    ('Haxell', 4, 'dead'),
    ('Turan/MWIS', 4, 'dead'),
    ('Degeneracy', 4, 'dead'),
    ('Ford divisor cap', 4, 'dead'),
    ('Mult. energy (K-M)', 4, 'promising'),
    ('Dyadic partition', 5, 'partial'),
    (r'$\sqrt{2}$-partition', 5, 'partial'),
    ('Fine partition', 5, 'partial'),
    ('Stratified Hall', 5, 'partial'),
    ('$V_{min}$ disjointness', 5, 'partial'),
    ('Derandomization', 6, 'dead'),
    ('Modular remainder', 6, 'dead'),
    ('Recursive doubling', 6, 'dead'),
    ('Surplus-excess', 6, 'dead'),
    ('Neumann series', 6, 'dead'),
    ('Quasi-independence', 6, 'dead'),
]

color_map = {'dead': '#e41a1c', 'partial': '#ff7f00', 'promising': '#4daf4a'}

fig, ax = plt.subplots(figsize=(12, 8))

y_pos = 0
yticks, ylabels = [], []
for cat_idx, cat in enumerate(categories):
    cat_approaches = [(name, out) for name, ci, out in approaches if ci == cat_idx]
    for i, (name, outcome) in enumerate(cat_approaches):
        ax.barh(y_pos, 1, color=color_map[outcome], edgecolor='white', linewidth=0.5)
        ax.text(1.05, y_pos, name, va='center', fontsize=7)
        yticks.append(y_pos)
        y_pos += 1
    y_pos += 0.5  # gap between categories

# Category labels
y_pos = 0
for cat_idx, cat in enumerate(categories):
    cat_approaches = [(name, out) for name, ci, out in approaches if ci == cat_idx]
    mid = y_pos + len(cat_approaches) / 2 - 0.5
    ax.text(-0.3, mid, cat, va='center', ha='right', fontsize=9, fontweight='bold')
    y_pos += len(cat_approaches) + 0.5

legend_patches = [
    mpatches.Patch(color='#e41a1c', label='Dead (ratio < 1 or fundamental obstruction)'),
    mpatches.Patch(color='#ff7f00', label='Partial (works locally or computationally)'),
    mpatches.Patch(color='#4daf4a', label='Promising (not yet quantitative)'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9)

ax.set_xlim(-0.5, 4)
ax.set_ylim(-1, y_pos)
ax.invert_yaxis()
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('43 Approaches to Erdős Problem #710: Status Summary', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

fig.tight_layout()
fig.savefig('fig_approach_summary.png', dpi=150, bbox_inches='tight')
fig.savefig('fig_approach_summary.pdf', bbox_inches='tight')
print("Saved fig_approach_summary.{png,pdf}")
