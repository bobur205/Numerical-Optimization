import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import cvxpy as cp

# ==========================================
# --- 1. DATA GENERATION (Provided) --------
# ==========================================
seed = 211
rng = np.random.default_rng(seed)

n = 2500
m = 250
fs = 8192                               # sampling frequency
t = np.arange(n) / fs
# Original signal (sum of two sine waves: 521 Hz and 1233 Hz)
y_orig = (np.sin(2*np.pi*521*t) + np.sin(2*np.pi*1233*t)) / 2.0

# Discrete Cosine Transform Matrix
D = scipy.fftpack.dct(np.eye(n), norm='ortho').transpose()

# Sample 10% of the data points randomly
k = np.sort(rng.choice(range(n), m, replace=False))
A = D[k, :]
b = y_orig[k]

# ==========================================
# --- 2. L2 NORM MINIMIZATION --------------
# ==========================================
print("Solving L2 norm minimization via KKT conditions...")

# We need to solve the KKT system:
# [ I_n   A^T ] [ x  ]   [ 0 ]
# [  A     0  ] [ nu ] = [ b ]
#
# Alternatively, solve (A * A^T) * nu = -b, then x = -A^T * nu
AAt = A @ A.T
# Solve for Lagrange multipliers nu
nu = np.linalg.solve(AAt, -b)
# Reconstruct frequency vector x
x_l2 = -A.T @ nu

# Reconstruct time-domain signal from frequency content
y_l2 = D @ x_l2


# ==========================================
# --- 3. L1 NORM MINIMIZATION (CVXPY) ------
# ==========================================
print("Solving L1 norm minimization (Linear Program) using CVXPY...")

# Define CVXPY variable
x_cvx = cp.Variable(n)

# Define objective: Minimize ||x||_1
objective = cp.Minimize(cp.norm1(x_cvx))

# Define constraint: Ax = b
constraints = [A @ x_cvx == b]

# Setup and solve problem
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS) # ECOS or OSQP are good for this

# Extract frequency vector
x_l1 = x_cvx.value

# Reconstruct time-domain signal
y_l1 = D @ x_l1


# ==========================================
# --- 4. VISUALIZATION AND COMPARISON ------
# ==========================================
plt.figure(figsize=(15, 10))

# 4.1 Plot frequency domain (L2 vs L1)
plt.subplot(2, 2, 1)
plt.plot(x_l2, label='L2 Reconstruction', color='orange')
plt.title("Frequency Content (L2 Minimization)")
plt.xlim([0, n])
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x_l1, label='L1 Reconstruction', color='green')
plt.title("Frequency Content (L1 Minimization)")
plt.xlim([0, n])
plt.legend()

# 4.2 Plot time domain comparison (zoom in to see detail)
# Plot the first 200 points to clearly see the wave shape
zoom_idx = 200

plt.subplot(2, 1, 2)
plt.plot(t[:zoom_idx], y_orig[:zoom_idx], 'k-', linewidth=2, label='Original Signal (Unknown)')
plt.plot(t[:zoom_idx], y_l2[:zoom_idx], 'r--', label='L2 Reconstruction')
plt.plot(t[:zoom_idx], y_l1[:zoom_idx], 'g-.', label='L1 Reconstruction')

# Plot the randomly sampled points that fall within this zoom window
sampled_in_view = [idx for idx in k if idx < zoom_idx]
plt.plot(t[sampled_in_view], y_orig[sampled_in_view], 'bo', markersize=5, label='Sampled Data Points (10%)')

plt.title("Time Domain Signal Reconstruction Comparison")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()