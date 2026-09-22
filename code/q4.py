import time

import numpy as np

from model import f_lambda_and_grad, lipschitz_constant, RESULTS_DIR

GRAD_TOL_REL = 1e-3
TIME_LIMIT_SEC = 180


def gradient_descent(theta0, X, s, lam, alpha,
                      grad_tol_rel=GRAD_TOL_REL, time_limit_sec=TIME_LIMIT_SEC):
    theta = theta0.copy()
    f, g = f_lambda_and_grad(theta, X, s, lam)
    g0_norm = np.linalg.norm(g)
    obj_hist, gradnorm_hist = [f], [g0_norm]

    start = time.perf_counter()
    stop_reason = "time_limit"
    while True:
        if gradnorm_hist[-1] <= grad_tol_rel * g0_norm:
            stop_reason = "grad_tol"
            break
        if time.perf_counter() - start > time_limit_sec:
            stop_reason = "time_limit"
            break
        theta = theta - alpha * g
        f, g = f_lambda_and_grad(theta, X, s, lam)
        obj_hist.append(f)
        gradnorm_hist.append(np.linalg.norm(g))

    elapsed = time.perf_counter() - start
    return theta, np.array(obj_hist), np.array(gradnorm_hist), stop_reason, elapsed


def save_history(obj_hist, gradnorm_hist):
    np.savetxt(RESULTS_DIR / "q4_gradient_descent.csv",
               np.column_stack([np.arange(len(obj_hist)), obj_hist, gradnorm_hist]),
               delimiter=",", header="iter,objective,grad_norm", comments="")


def save_summary(L, alpha, stop_reason, n_iter, elapsed):
    lines = [
        f"Lipschitz constant L = sigma_max(X)^2 + lambda: {L:.6e}",
        f"step size alpha = 1/L: {alpha:.6e}",
        f"stopping reason: {stop_reason}",
        f"iterations: {n_iter}",
        f"elapsed time (s): {elapsed:.2f}",
    ]
    (RESULTS_DIR / "q4_summary.txt").write_text("\n".join(lines) + "\n")


def run(X_train, s_train, lam):
    rng = np.random.default_rng(2)
    theta0 = rng.standard_normal(X_train.shape[0]) * 0.01

    L = lipschitz_constant(X_train, lam)
    alpha = 1.0 / L

    theta_final, obj_hist, gradnorm_hist, stop_reason, elapsed = gradient_descent(
        theta0, X_train, s_train, lam, alpha)

    save_history(obj_hist, gradnorm_hist)
    save_summary(L, alpha, stop_reason, len(obj_hist) - 1, elapsed)

    return theta_final, obj_hist, gradnorm_hist
