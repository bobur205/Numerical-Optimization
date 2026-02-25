import numpy as np
import matplotlib.pyplot as plt


A = np.array([[6, -2],
              [-2, 2]])
b = np.array([-1, 2])
c = 1

x_min = np.linalg.solve(A, -b)
print(f"Min point: x1={x_min[0]}, x2={x_min[1]}")

# MEshgrid
# Neigborhood of Minimum point (-2 ; 2)
x1_range = np.linspace(x_min[0]-1, x_min[0]+1, 100)
x2_range = np.linspace(x_min[1]-1, x_min[1]+1, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)

# f(x) = 3x1^2 - 2x1x2 + x2^2 - x1 + 2x2 + 1
Z = 3*X1**2 - 2*X1*X2 + X2**2 - X1 + 2*X2 + 1

fig = plt.figure(figsize=(14, 6)) #
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)


ax1.set_title('3D Surface View')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')

ax2 = fig.add_subplot(1, 2, 2)
cp = ax2.contour(X1, X2, Z, levels=20, cmap='viridis')
plt.colorbar(cp, ax=ax2, label='f(x)')

# I Highlited Min point with plus sign
ax2.plot(x_min[0], x_min[1], 'r+', markersize=15, label='Minimum (-0.25, -1.25)')

ax2.set_title('Function contours and its min point')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.legend()
ax2.grid(True)
ax2.axis('equal')

plt.tight_layout()
plt.show()