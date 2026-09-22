import numpy as np
from pathlib import Path
from scipy.io import loadmat

DATA_DIR = Path(__file__).resolve().parent / ".." / "data"
RESULTS_DIR = Path(__file__).resolve().parent / ".." / "results"

LAMBDA = 0.005


def load_data():
    data = loadmat(DATA_DIR / "mnist_train_test.mat",
                    squeeze_me=True, struct_as_record=False)
    train, test = data["train"], data["test"]
    return train.X, train.y.astype(float), test.X, test.y.astype(float)


def labels_to_signs(y):
    return 1.0 - 2.0 * y


def phi(z):
    return np.where(z <= -1, 0.0,
                     np.where(z <= 0, 0.5 * (1.0 + z) ** 2, 0.5 + z))


def dphi(z):
    return np.where(z <= -1, 0.0,
                     np.where(z <= 0, 1.0 + z, 1.0))


def f_lambda(theta, X, s, lam, vectorized=True):
    if vectorized:
        z = s * (X.T @ theta)
        return np.sum(phi(z)) + 0.5 * lam * np.dot(theta, theta)
    total = sum(phi(s[i] * np.dot(X[:, i], theta)) for i in range(X.shape[1]))
    return total + 0.5 * lam * np.dot(theta, theta)


def grad_f_lambda(theta, X, s, lam, vectorized=True):
    if vectorized:
        z = s * (X.T @ theta)
        return X @ (s * dphi(z)) + lam * theta
    grad = np.zeros_like(theta)
    for i in range(X.shape[1]):
        z_i = s[i] * np.dot(X[:, i], theta)
        grad += dphi(z_i) * s[i] * X[:, i]
    return grad + lam * theta


def f_lambda_and_grad(theta, X, s, lam):
    """Vectorized f_lambda and grad f_lambda, sharing one z = s * (X.T @ theta)."""
    z = s * (X.T @ theta)
    f = np.sum(phi(z)) + 0.5 * lam * np.dot(theta, theta)
    grad = X @ (s * dphi(z)) + lam * theta
    return f, grad


def lipschitz_constant(X, lam):
    """L = sigma_max(X)^2 + lam, via the top eigenvalue of X @ X.T (cheaper
    than an SVD of X since X has far fewer rows than columns here)."""
    sigma_max_sq = np.linalg.eigvalsh(X @ X.T)[-1]
    return sigma_max_sq + lam
