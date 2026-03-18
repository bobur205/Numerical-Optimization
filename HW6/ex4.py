import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import block_diag, kron, eye, csr_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve
import matplotlib
import warnings
warnings.filterwarnings("ignore")

#  I read original pictur
U0 = plt.imread('bwicon.png')  # values in [0,1]
if U0.ndim == 3:
    U0 = U0[:,:,0]  # convert to grayscale if needed
m, n = U0.shape

# I Created 50% mask of known pixels (same as given)
np.random.seed(211)
unknown = np.random.rand(m, n) < 0.5
U1 = U0 * (1 - unknown) + 0.5 * unknown   # gray color in missing areas

# we need Build the sparse matrices D_h and D_v

# D_h: horizontal differences (between columns)
# For each column j from 1 to n-1, we have differences for each row i
# So D_h has (n-1)*m rows and m*n columns
row_indices = []
col_indices = []
data = []
for j in range(n-1):
    for i in range(m):
        # difference between X(i, j+1) and X(i, j)
        # column indices: (j)*m + i  (current column) and (j+1)*m + i (next column)
        row = j*m + i
        col_curr = j*m + i
        col_next = (j+1)*m + i
        # D_h * x gives next - current: so we put -1 at current, +1 at next
        row_indices.append(row)
        col_indices.append(col_curr)
        data.append(-1.0)
        row_indices.append(row)
        col_indices.append(col_next)
        data.append(1.0)
D_h = csr_matrix((data, (row_indices, col_indices)), shape=((n-1)*m, m*n))

# D_v: vertical differences (within each column)
# For each column j, we have differences for rows i from 1 to m-1
row_indices = []
col_indices = []
data = []
for j in range(n):
    for i in range(m-1):
        # difference between X(i+1, j) and X(i, j)
        row = j*(m-1) + i
        col_curr = j*m + i
        col_next = j*m + (i+1)
        row_indices.append(row)
        col_indices.append(col_curr)
        data.append(-1.0)
        row_indices.append(row)
        col_indices.append(col_next)
        data.append(1.0)
D_v = csr_matrix((data, (row_indices, col_indices)), shape=(n*(m-1), m*n))

# Build Q = D_h'*D_h + D_v'*D_v
Q = D_h.T @ D_h + D_v.T @ D_v

# Equality constraints: known pixels
known = ~unknown  # boolean mask of known pixels
known_indices = np.where(known.ravel(order='F'))[0]  # column-major order
b = U1.ravel(order='F')[known_indices]

# Build A_eq: each row picks a known pixel
A_eq = csr_matrix((np.ones(len(known_indices)), (np.arange(len(known_indices)), known_indices)),
                  shape=(len(known_indices), m*n))

# KKT system: [Q, A_eq.T; A_eq, 0] * [x; nu] = [0; b]
n_vars = m*n
n_eq = len(known_indices)
# Assemble top-left block: Q (sparse)
# top-right: A_eq.T
# bottom-left: A_eq
# bottom-right: zero block
from scipy.sparse import bmat
KKT = bmat([[Q, A_eq.T], [A_eq, None]], format='csr')
rhs = np.concatenate([np.zeros(n_vars), b])

# Solve
sol = spsolve(KKT, rhs)
x_opt = sol[:n_vars]
nu = sol[n_vars:]  # Lagrange multipliers (not used here)

# Reshape to image (column-major)
X_rec = x_opt.reshape((m, n), order='F')

# Display results
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(U0, cmap='gray')
plt.title('Original')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(U1, cmap='gray')
plt.title('Obscured (50% missing)')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(X_rec, cmap='gray')
plt.title('Reconstructed')
plt.axis('off')
plt.tight_layout()
plt.show()