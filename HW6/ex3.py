import numpy as np
import cvxpy as cp

# ==========================================
# --- 1. INPUT YOUR BLACKBOARD DATA HERE ---
# ==========================================

# Replace these example arrays with the actual data for Polygon C (A1*x <= b1)
A1 = np.array( [
      [5.8479532e-01,  -1.9354839e+00],
      [2.3859649e+00,   1.1428571e+00],
      [7.0175439e-01,   1.2350230e+00],
      [-1.0292398e+00,  6.8202765e-01],
      [-1.1695906e+00,  5.5299539e-02],
      [-1.4736842e+00, -1.1797235e+00]
      ] )

b1 = np.array( [
      3.3837281e+00,
      9.5981890e-01,
      1.1496483e+00,
      2.4695071e+00,
      2.3474816e+00,
      3.6227127e+00
      ] )

A2 = np.array( [
      [7.0175439e-02,  -2.2304147e+00],
      [2.4795322e+00,   5.5299539e-02],
      [1.1228070e+00,   1.6774194e+00],
      [-1.0994152e+00,  1.1244240e+00],
      [-2.5730994e+00, -6.2672811e-01]
      ] )

b2 = np.array( [
      -6.9765812e-01,
      9.0161964e+00,
      8.8853316e+00,
      2.4482712e+00,
      -3.8164228e+00
      ] )


# ==========================================
# --- 2. OPTIMIZATION PROBLEM SETUP --------
# ==========================================

# Define variables: 2D coordinates for point x (in C) and point y (in D)
x = cp.Variable(2)
y = cp.Variable(2)

# Define the objective: Minimize the squared Euclidean distance ||x - y||^2
objective = cp.Minimize(cp.sum_squares(x - y))

# Define the kinematic constraints for the polygons
constraints = [
    A1 @ x <= b1,
    A2 @ y <= b2
]

# Formulate and solve the Quadratic Program
prob = cp.Problem(objective, constraints)
prob.solve()

# ==========================================
# --- 3. RESULTS AND MULTIPLIERS -----------
# ==========================================

print("=== OPTIMIZATION RESULTS ===")
print(f"Status: {prob.status}")
print(f"Minimum Distance: {np.sqrt(prob.value):.4f}")
print(f"Squared Distance: {prob.value:.4f}")

print("\n=== OPTIMAL COORDINATES ===")
print(f"Closest point in C (x*): [{x.value[0]:.4f}, {x.value[1]:.4f}]")
print(f"Closest point in D (y*): [{y.value[0]:.4f}, {y.value[1]:.4f}]")

print("\n=== LAGRANGE MULTIPLIERS (DUAL VARIABLES) ===")
# A multiplier is > 0 if the corresponding constraint (polygon edge) is active
print("Multipliers for Polygon C (lambda):")
print(np.round(constraints[0].dual_value, 4))

print("\nMultipliers for Polygon D (mu):")
print(np.round(constraints[1].dual_value, 4))