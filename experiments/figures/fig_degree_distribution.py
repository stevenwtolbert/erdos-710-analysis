#!/usr/bin/env python3
"""Figure 1: Degree distribution d(k) vs k for several n."""
import numpy as np
import matplotlib.pyplot as plt
import math

def compute_params(n, eps=0.05):
    C = 2.0 / math.sqrt(math.e)
    L = int((C + eps) * n * math.sqrt(math.log(n) / math.log(math.log(n))))
    M = L - n
    N = n // 2
    return L, M, N

def degree(k, n, L):
    return (n + L) // k - (2 * n) // k

fig, ax = plt.subplots(figsize=(8, 5))

for n in [5000, 20000, 100000]:
    L, M, N = compute_params(n)
    ks = np.arange(1, N + 1)
    degs = np.array([degree(k, n, L) for k in ks])
    ax.plot(ks / N, degs, label=f'$n = {n:,}$', alpha=0.8, linewidth=0.8)

ax.set_xlabel('$k / N$')
ax.set_ylabel('$d(k)$')
ax.set_title('Degree distribution $d(k) = \\lfloor(n+L)/k\\rfloor - \\lfloor 2n/k\\rfloor$')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('fig_degree_distribution.png', dpi=150)
fig.savefig('fig_degree_distribution.pdf')
print("Saved fig_degree_distribution.{png,pdf}")
