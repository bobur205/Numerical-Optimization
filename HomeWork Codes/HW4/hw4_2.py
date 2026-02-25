import numpy as np
import matplotlib.pyplot as plt


# Rosen
def rosenbrock(x):
    return (1.0 - x[0]) ** 2 + 10 * (x[1] - x[0] ** 2) ** 2

# rosen gradient
def rosenbrock_grad(a):
    g=np.zeros(2)
    x1=a[0]; x2=a[1]
    g[0]=-40*x1*(x2-x1**2)+2*x1-2
    g[1]=20*x2-20*x1**2
    return g

#rosen hessian
def rosenbrock_hess(a):
    x1 = a[0]; x2 = a[1]
    f_x1x1=2-40*(x2-x1**2)+80*x1**2
    f_x1x2 = -40.0 * x1
    f_x2x1 = -40.0 * x1
    f_x2x2 = 20
    return np.array([[f_x1x1, f_x1x2], [f_x2x1, f_x2x2]])


# Backtracking line search (Armijo condition)
def backtracking_line_search(f, grad_f, x, p, alpha=0.1, beta=0.8):
    t = 1.0
    while f(x + t * p) > f(x) + alpha * t * (grad_f(x)@p):
        t *= beta
    return t

# BFGS Method
def bfgs(x0, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    H = np.eye(len(x))  # I took inverse Hessian approximation here

    history = [x.copy()]

    for i in range(max_iter):
        grad = rosenbrock_grad(x)
        if np.linalg.norm(grad) < tol:
            break
        # taking first Quasi-Newton direction
        p = -(H@grad)

        t = backtracking_line_search(rosenbrock, rosenbrock_grad, x, p)

        x_new = x + t * p
        history.append(x_new.copy())

        s = x_new - x
        y = rosenbrock_grad(x_new) - grad

        # Curvature condition check: s^T y > 0
        sty = (s@y)
        if sty > 0:
            # Update inverse Hessian using the formula from the lecture notes
            term1 = (sty + (y@(H@y))) / (sty ** 2)
            term2 = np.outer(s, s)
            term3 = (H@np.outer(y, s)) + (np.outer(s, y)@H)

            H = H + term1 * term2 - term3 / sty
        x = x_new
    return x, i, history

# Newton's Method
def newton(x0, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    history = [x.copy()]

    for i in range(max_iter):
        grad = rosenbrock_grad(x)
        if np.linalg.norm(grad) < tol:
            break
        hess = rosenbrock_hess(x)
        # newton direction finding
        try:
            p = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            # if Hessian is not positive definite / singular
            p = -grad

        t= backtracking_line_search(rosenbrock, rosenbrock_grad, x, p)
        x = x + t* p
        history.append(x.copy())

    return x, i, history


# ---------------- Comparison Execution ----------------
if __name__ == "__main__":
    x0 = [-1.2, 1.0]  # I choose a standart  point  from previuos HW

    print("Starting point:", x0)
    print("-" * 50)

    x_bfgs, iters_bfgs, hist_bfgs = bfgs(x0)
    print(f"BFGS Method:")
    print(f"Iterations: {iters_bfgs}")
    print(f"Final Minimum: {x_bfgs}")
    print(f"Final Function Value: {rosenbrock(x_bfgs):.10e}")
    print("-" * 50)

    x_newton, iters_newton, hist_newton = newton(x0)
    print(f"Newton's Method:")
    print(f"Iterations: {iters_newton}")
    print(f"Final Minimum: {x_newton}")
    print(f"Final Function Value: {rosenbrock(x_newton):.10e}")



# Visualization / Comparison Plots

f_vals_bfgs = [rosenbrock(x) for x in hist_bfgs]
f_vals_newton = [rosenbrock(x) for x in hist_newton]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(f_vals_bfgs, 'r.-', label='BFGS', markersize=8)
plt.plot(f_vals_newton, 'b.--', label="Newton's Method", markersize=8)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('f(x) - Log Scale')
plt.title('Convergence Rate Comparison')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# 2. 2D Contour Plot of Trajectories
plt.subplot(1, 2, 2)
x_mesh = np.linspace(-1.5, 1.5, 400)
y_mesh = np.linspace(-0.5, 1.5, 400)
X, Y = np.meshgrid(x_mesh, y_mesh)
Z = (1.0 - X)**2 + 10.0 * (Y - X**2)**2

# Logarithmic contour levels for the steep Rosenbrock valley
levels = np.logspace(-2, 3, 25)
plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.5)

# Extract coordinates
bfgs_x, bfgs_y = zip(*hist_bfgs)
newton_x, newton_y = zip(*hist_newton)

plt.plot(bfgs_x, bfgs_y, 'r.-', label='BFGS Path', linewidth=1.5)
plt.plot(newton_x, newton_y, 'b.--', label='Newton Path', linewidth=1.5)
plt.plot(1, 1, 'k*', markersize=12, label='Minimum (1,1)')

plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('Optimization Trajectories')
plt.legend()
plt.tight_layout()
plt.show()