#!/usr/bin/env python3
"""Figure 9: Semi-random nibble residual fraction vs n (Z115 data)."""
import matplotlib.pyplot as plt
import numpy as np

# Data from state_115_semi_random_nibble.md
ns =          [1000, 2000, 5000, 10000, 20000, 30000, 50000]
resid_pct =   [52.2, 48.9, 43.3, 38.7,  34.4,  32.3,  29.5]
unmatch_pct = [29.7, None, None, 29.4,  None,  24.8,  None]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(ns, resid_pct, 'bo-', markersize=6, linewidth=1.5,
        label='Residual after nibble')

# Plot unmatchable points where available
um_ns = [n for n, u in zip(ns, unmatch_pct) if u is not None]
um_vs = [u for u in unmatch_pct if u is not None]
ax.plot(um_ns, um_vs, 'rs-', markersize=8, linewidth=1.5,
        label='Permanently unmatchable (HK fails)')

ax.axhline(y=0, color='green', linestyle='--', linewidth=1,
           label='Target (0% residual)')

for n, u in zip(um_ns, um_vs):
    ax.annotate(f'HK FAILS\n({u:.1f}%)', (n, u),
                textcoords='offset points', xytext=(15, 10),
                fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

ax.set_xscale('log')
ax.set_xlabel('$n$')
ax.set_ylabel('Fraction of $|V|$ (\\%)')
ax.set_title('Semi-random nibble: residual shrinks but remains unmatchable')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 60)
fig.tight_layout()
fig.savefig('fig_nibble_residual.png', dpi=150)
fig.savefig('fig_nibble_residual.pdf')
print("Saved fig_nibble_residual.{png,pdf}")
