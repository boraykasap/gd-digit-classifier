from model import phi, dphi, RESULTS_DIR
import matplotlib.pyplot as plt
import numpy as np
import time


def q3_check_gradient_compute(theta, X_tilde, s, lam, t, v):
    z = X_tilde.T @ theta
    theta_norm = np.dot(theta, theta)
    delta = X_tilde.T @ v
    f0 = np.sum(phi(s * z)) + 0.5 * lam * theta_norm
    grad_f0 = X_tilde @ (s * dphi(s * z)) + lam * theta
    # For vectorization, we resize t =[a1,a2,...] as t_reshape = [[a1],[a2],...],
    # as this enables us to use numpy's broadcasting system'
    t_reshaped = t[:, np.newaxis]
    offset = theta + t_reshaped * v
    ft = np.sum(phi(s * (z + t_reshaped * delta)), axis=1) + 0.5 * lam * np.sum(
        offset * offset, axis=1
    )
    return np.abs(ft - f0 - t * np.dot(v, grad_f0))


def save_plot(t, errors):
    plt.figure()
    plt.loglog(t, errors)
    plt.xlabel("t")
    plt.ylabel(
        r"$|f_\lambda(\theta+tv)-f_\lambda(\theta)-t\langle v,\nabla f_\lambda(\theta)\rangle|$"
    )
    plt.title("Gradient check")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + "q3_gradient_check.pdf")
    plt.close()


def save_data(t, errors):
    np.savetxt(
        RESULTS_DIR + "q3_gradient_check.csv",
        np.column_stack([t, errors]),
        delimiter=",",
        header="t,error",
        comments="",
    )


def run(X_train, s_train, lam):
    print("Question 3 : starting computation ...")
    t0 = time.perf_counter()
    rng = np.random.default_rng(1)
    theta = rng.standard_normal(X_train.shape[0]) * 0.01
    v = rng.standard_normal(X_train.shape[0])
    v /= np.linalg.norm(v)
    t = np.logspace(-8.0, 0.0, 101)
    errors = q3_check_gradient_compute(theta, X_train, s_train, lam, t, v)
    save_plot(t, errors)
    save_data(t, errors)
    print(
        f"""Question 3 done : saved data as ../results/q3_gradient_check.csv
            and saved plot as ../results/q3_gradient_check.pdf : computation took {
            (time.perf_counter() - t0):.4f} s"""
    )
