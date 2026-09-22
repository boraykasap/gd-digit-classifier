import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import phi, f_lambda, grad_f_lambda, RESULTS_DIR


def gradient_check(theta, v, X, s, lam):
    t = np.logspace(-8, 0, 101)
    f0 = f_lambda(theta, X, s, lam)
    directional = np.dot(v, grad_f_lambda(theta, X, s, lam))

    # z(t) = s * (X.T @ (theta + t*v)) is affine in t, so compute the two
    # matrix-vector products once and reuse them for every t instead of
    # calling f_lambda (and its own X.T @ ... product) 101 times.
    z0 = s * (X.T @ theta)
    delta = s * (X.T @ v)
    z_t = z0[None, :] + t[:, None] * delta[None, :]  # (len(t), m)
    reg_t = 0.5 * lam * (np.dot(theta, theta) + 2 * t * np.dot(theta, v) + t ** 2 * np.dot(v, v))
    f_t = phi(z_t).sum(axis=1) + reg_t

    errors = np.abs(f_t - f0 - t * directional)
    return t, errors


def save_plot(t, errors):
    plt.figure()
    plt.loglog(t, errors)
    plt.xlabel("t")
    plt.ylabel(r"$|f_\lambda(\theta+tv)-f_\lambda(\theta)-t\langle v,\nabla f_\lambda(\theta)\rangle|$")
    plt.title("Gradient check")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "q3_gradient_check.pdf")
    plt.close()


def save_data(t, errors):
    np.savetxt(RESULTS_DIR / "q3_gradient_check.csv",
               np.column_stack([t, errors]), delimiter=",",
               header="t,error", comments="")


def run(X_train, s_train, lam):
    rng = np.random.default_rng(1)
    theta = rng.standard_normal(X_train.shape[0]) * 0.01
    v = rng.standard_normal(X_train.shape[0])
    v /= np.linalg.norm(v)

    t, errors = gradient_check(theta, v, X_train, s_train, lam)
    save_plot(t, errors)
    save_data(t, errors)
