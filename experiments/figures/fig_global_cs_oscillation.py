#!/usr/bin/env python3
"""Figure 4: Global CS(T₀) vs n at base constant (Z113b/Z114 data)."""
import matplotlib.pyplot as plt
import numpy as np

# Data from Z114 state file (C_mult = 1.00)
ns =      [2000,  5000,  7000,  10000, 20000, 50000, 100000]
cs_vals = [1.0272, 1.0319, 1.0050, 0.9910, 0.9923, 0.9980, 0.9876]
t0_frac = [0.454, 0.473, 0.481, 0.489, 0.503, 0.520, 0.531]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

# Top: CS ratio
ax1.plot(ns, cs_vals, 'bo-', markersize=6, linewidth=1.5)
ax1.axhline(y=1.0, color='r', linestyle='--', linewidth=1, label='Threshold = 1')
ax1.fill_between(ns, 0.97, 1.0, alpha=0.1, color='red')
ax1.set_ylabel('$\\mathrm{CS}(T_0)$')
ax1.set_title('Global Cauchy--Schwarz ratio for adversarial $T_0$ (base constant)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.97, 1.05)

# Bottom: T0 fraction
ax2.plot(ns, [100*f for f in t0_frac], 'gs-', markersize=6, linewidth=1.5)
ax2.set_xlabel('$n$')
ax2.set_ylabel('$|T_0|/|V|$ (\\%)')
ax2.set_title('Adversarial subset fraction')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('fig_global_cs_oscillation.png', dpi=150)
fig.savefig('fig_global_cs_oscillation.pdf')
print("Saved fig_global_cs_oscillation.{png,pdf}")
