import time

import numpy as np

from model import f_lambda_and_grad, lipschitz_constant, RESULTS_DIR

GRAD_TOL_REL = 1e-3
TIME_LIMIT_SEC = 180


def gradient_descent(
    theta0,
    X_tilde,
    s,
    lam,
    alpha,
    grad_tol_rel=GRAD_TOL_REL,
    time_limit_sec=TIME_LIMIT_SEC,
):
    theta = theta0.copy()
    fk, grad_fk = f_lambda_and_grad(theta, X_tilde, s, lam)
    g0_norm = np.linalg.norm(grad_fk)
    obj_hist, gradnorm_hist = [fk], [g0_norm]

    start = time.perf_counter()
    stop_reason = "time limit attained"
    run = True
    while run:
        if gradnorm_hist[-1] <= grad_tol_rel * g0_norm:
            stop_reason = "grad tolerance matched"
            run = False
        if time.perf_counter() - start > time_limit_sec:
            stop_reason = "time limit attained"
            run = False
        theta = theta - alpha * grad_fk
        fk, grad_fk = f_lambda_and_grad(theta, X_tilde, s, lam)
        obj_hist.append(fk)
        gradnorm_hist.append(np.linalg.norm(grad_fk))

    elapsed = time.perf_counter() - start
    return theta, np.array(obj_hist), np.array(gradnorm_hist), stop_reason, elapsed


def save_history(obj_hist, gradnorm_hist):
    np.savetxt(
        RESULTS_DIR + "q4_gradient_descent.csv",
        np.column_stack([np.arange(len(obj_hist)), obj_hist, gradnorm_hist]),
        delimiter=",",
        header="iter,objective,grad_norm",
        comments="",
    )


def save_summary(L, alpha, stop_reason, n_iter, elapsed):
    lines = [
        f"Lipschitz constant L = sigma_max(X)^2 + lambda: {L:.6e}\n",
        f"step size alpha = 1/L: {alpha:.6e}\n",
        f"stopping reason: {stop_reason}\n",
        f"iterations: {n_iter}\n",
        f"elapsed time (s): {elapsed:.4f}\n",
    ]
    with open(RESULTS_DIR + "q4_summary.txt", "w") as result_file:
        result_file.writelines(lines)


def run(X_train, s_train, lam):
    print("Question 4 : starting computation ...")
    t0 = time.perf_counter()

    rng = np.random.default_rng(2)
    theta0 = rng.standard_normal(X_train.shape[0]) * 0.01

    L = lipschitz_constant(X_train, lam)
    alpha = 1.0 / L

    theta_final, obj_hist, gradnorm_hist, stop_reason, elapsed = gradient_descent(
        theta0, X_train, s_train, lam, alpha
    )

    save_history(obj_hist, gradnorm_hist)
    save_summary(L, alpha, stop_reason, len(obj_hist) - 1, elapsed)

    print(
        f"""Question 4 done : saved summary as ../results/q4_summary.txt
        and saved data history as ../results/q4_gradient_descent.csv : computation took {
            (time.perf_counter() - t0):.4f} s"""
    )
    return theta_final, obj_hist, gradnorm_hist
