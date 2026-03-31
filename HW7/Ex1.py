import numpy as np
import matplotlib.pyplot as plt

lam = np.linspace(0, 10, 400)

g_lam = (-lam**2 + 9*lam + 1) / (lam + 1)

plt.figure(figsize=(8, 5))
plt.plot(lam, g_lam, label=r'$g(\lambda) = \frac{-\lambda^2 + 9\lambda + 1}{\lambda + 1}$', color='blue', linewidth=2)

plt.scatter(2, 5, color='red', zorder=5)
plt.annotate(r'Optimal point: $\lambda^* = 2, g^* = 5$',
             xy=(2, 5), xytext=(3, 4.5),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=12)

plt.title('Dual function plot', fontsize=14, fontweight='bold')
plt.xlabel(r'Lagrangian multiplier $\lambda$', fontsize=12)
plt.ylabel(r'$g(\lambda)$', fontsize=12)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

plt.show()