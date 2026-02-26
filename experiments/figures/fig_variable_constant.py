#!/usr/bin/env python3
"""Figure 7: CS(T₀) vs n for multiple C_mult values (Z114 data)."""
import matplotlib.pyplot as plt
import numpy as np

# Data from Z114 state file
ns = [2000, 10000, 50000, 100000]

data = {
    1.00: [1.027, 0.991, 0.998, 0.988],
    1.01: [1.052, 1.012, 1.010, 1.001],
    1.05: [1.119, 1.076, 1.066, 1.061],
    1.10: [1.237, 1.163, 1.140, 1.138],
    1.20: [1.422, 1.319, 1.296, 1.295],
    1.50: [2.078, 1.817, 1.739, 1.742],
}

fig, ax = plt.subplots(figsize=(8, 5))

colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3', '#666666']
for i, (cmult, vals) in enumerate(data.items()):
    style = '--' if cmult == 1.00 else '-'
    ax.plot(ns, vals, f'{style}o', color=colors[i], markersize=5,
            linewidth=1.5, label=f'$C_{{\\mathrm{{mult}}}} = {cmult}$')

ax.axhline(y=1.0, color='black', linestyle=':', linewidth=0.8)
ax.set_xscale('log')
ax.set_xlabel('$n$')
ax.set_ylabel('$\\mathrm{CS}(T_0)$')
ax.set_title('Variable-constant CS: $\\mathrm{CS}(T_0)$ stabilizes, does not grow')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('fig_variable_constant.png', dpi=150)
fig.savefig('fig_variable_constant.pdf')
print("Saved fig_variable_constant.{png,pdf}")
