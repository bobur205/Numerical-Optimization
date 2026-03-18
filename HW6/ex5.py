import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# Load the original image
U0 = plt.imread('bwicon.png')
if U0.ndim == 3:
    U0 = U0[:,:,0]
m, n = U0.shape

# Create 50% mask
np.random.seed(211)
unknown = np.random.rand(m, n) < 0.5
U1 = U0 * (1 - unknown) + 0.5 * unknown

# Known pixels
known = ~unknown
known_rows, known_cols = np.where(known)
known_vals = U1[known_rows, known_cols]

# Variables
X = cp.Variable((m, n))
obj = cp.sum(cp.abs(X[:, 1:] - X[:, :-1])) + cp.sum(cp.abs(X[1:, :] - X[:-1, :]))
constraints = [X[known_rows, known_cols] == known_vals]
problem = cp.Problem(cp.Minimize(obj), constraints)

# Solve with SCS
problem.solve(solver=cp.SCS, verbose=False)

X_rec = X.value

# Display
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(U0, cmap='gray'); plt.title('Original'); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(U1, cmap='gray'); plt.title('Obscured'); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(X_rec, cmap='gray'); plt.title('TV reconstruction'); plt.axis('off')
plt.tight_layout()
plt.show()