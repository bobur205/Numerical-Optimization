import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


# 1. DATA GENERATION ( professor provided
seed = 211
rng = np.random.default_rng(seed)
nd = 300
P = 2 * rng.random((nd, 2))
a_true = np.array([4, 2])
b_true = 6

blue = (P @ a_true.transpose() - b_true * np.ones(nd)) > 1.0
red = (P @ a_true.transpose() - b_true * np.ones(nd)) < -1.0
X = np.vstack((P[blue, :], P[red, :]))
mblue = P[blue, :].shape[0]
mred = P[red, :].shape[0]
m = mblue + mred

# Create Diagonal matrix D (+1 for blue, -1 for red)
y = np.hstack([np.ones(mblue), -np.ones(mred)])
D = np.diag(y)

# PART 1: Hard-Margin SVM (Separable Case)

print("--- Part 1: Hard-Margin SVM ---")
a = cp.Variable(2)
b = cp.Variable()


objective_hard = cp.Minimize(0.5 * cp.sum_squares(a))

constraints_hard = [D @ (X @ a - b * np.ones(m)) >= 1]

prob_hard = cp.Problem(objective_hard, constraints_hard)
prob_hard.solve()

a_opt = a.value
b_opt = b.value
print(f"Optimal a: {a_opt}")
print(f"Optimal b: {b_opt}")

# PART 2: Infeasible Case (Rogue Data Point)

print("\n--- Part 2: Infeasibility Check ---")
# Add a blue point (+1) deep in the red territory
rogue_point = np.array([[0.5, 0.5]])  # Deep in bottom-left
X_rogue = np.vstack([X, rogue_point])
y_rogue = np.hstack([y, 1.0])  # Label it as blue (+1)
D_rogue = np.diag(y_rogue)
m_rogue = m + 1

a_r = cp.Variable(2)
b_r = cp.Variable()
prob_rogue = cp.Problem(
    cp.Minimize(0.5 * cp.sum_squares(a_r)),
    [D_rogue @ (X_rogue @ a_r - b_r * np.ones(m_rogue)) >= 1]
)
try:
    prob_rogue.solve()
    print(f"Problem Status: {prob_rogue.status}")
except Exception as e:
    print(f"Problem Status: infeasible (Solver failed as expected)")


# PART 3: Soft-Margin SVM (Pareto Curve)
print("\n--- Part 3: Soft-Margin SVM & Pareto Curve ---")
alphas = np.logspace(-2, 2, 20)
margin_widths = []
slack_penalties = []

a_soft = cp.Variable(2)
b_soft = cp.Variable()
u = cp.Variable(m_rogue)

for alpha in alphas:
    objective_soft = cp.Minimize(0.5 * cp.sum_squares(a_soft) + alpha * cp.sum(u))
    constraints_soft = [
        D_rogue @ (X_rogue @ a_soft - b_soft * np.ones(m_rogue)) >= 1 - u,
        u >= 0
    ]
    prob_soft = cp.Problem(objective_soft, constraints_soft)
    prob_soft.solve()

    margin_widths.append(2.0 / np.linalg.norm(a_soft.value))
    slack_penalties.append(np.sum(u.value))


exact_alpha = 1.0
objective_plot = cp.Minimize(0.5 * cp.sum_squares(a_soft) + exact_alpha * cp.sum(u))
prob_plot = cp.Problem(objective_plot, constraints_soft)
prob_plot.solve()

a_plot = a_soft.value
b_plot = b_soft.value


plt.figure('Pareto Tradeoff Curve')
plt.plot(slack_penalties, margin_widths, 'k-o', linewidth=2)
plt.xlabel(r'Total Misclassification Slack ($\sum u_i$)')
plt.ylabel(r'Margin Width ($2/||a||_2$)')
plt.title('Pareto Optimal Tradeoff Curve')
plt.grid(True)
plt.savefig('pareto_curve.png')
plt.show()


plt.figure('Soft-Margin Classification')
plt.plot(X_rogue[:mblue, 0], X_rogue[:mblue, 1], 'bo', markersize=3, label='Blue Class')
plt.plot(X_rogue[mblue:m, 0], X_rogue[mblue:m, 1], 'ro', markersize=3, label='Red Class')
plt.plot(rogue_point[0, 0], rogue_point[0, 1], 'co', markersize=6, label='Added Rogue Point')


x_vals = np.array([0, 2])
y_bound = (b_plot - a_plot[0] * x_vals) / a_plot[1]
y_margin_pos = (b_plot + 1 - a_plot[0] * x_vals) / a_plot[1]
y_margin_neg = (b_plot - 1 - a_plot[0] * x_vals) / a_plot[1]

plt.plot(x_vals, y_bound, 'k-', lw=2, label=r'Decision Boundary ($a^Tx - b = 0$)')
plt.plot(x_vals, y_margin_pos, 'k--', lw=1, label=r'Margin ($a^Tx - b = 1$)')
plt.plot(x_vals, y_margin_neg, 'k--', lw=1, label=r'Margin ($a^Tx - b = -1$)')

plt.xlim(0, 2)
plt.ylim(0, 2)
plt.legend(loc='best')
plt.title(r'Soft-Margin Classification ($\alpha = 1.0$)')
plt.savefig('soft_margin_solution.png')  # For LaTeX
plt.show()