import numpy as np
from scipy.io import loadmat

RESULTS_DIR = "../results/"

LAMBDA = 0.005


def load_data():
    data = loadmat(
        "../data/mnist_train_test.mat",
        squeeze_me=True,
        struct_as_record=False,
    )
    train, test = data["train"], data["test"]
    return train.X, train.y, test.X, test.y
    # return train.X, train.y.astype(float), test.X, test.y.astype(float)


def labels_to_signs(y):
    return 1.0 - 2.0 * y


def phi(z):
    return np.where(
        z <= -1,
        0.0,
        np.where(
            z <= 0,
            0.5 * (1.0 + z) * (1.0 + z),
            0.5 + z,
        ),
    )


def dphi(z):
    return np.where(
        z <= -1,
        0.0,
        np.where(
            z <= 0,
            1.0 + z,
            1.0,
        ),
    )


def f_lambda(
    theta,
    X_tilde,
    s,
    lam,
    vectorized=True,
):
    if vectorized:
        return np.sum(phi(s * (X_tilde.T @ theta))) + 0.5 * lam * np.dot(theta, theta)
    total = 0.0
    for index, x_tilde in enumerate(X_tilde.T):
        total += phi(s[index] * np.dot(x_tilde, theta))
    return total + 0.5 * lam * np.dot(theta, theta)


def grad_f_lambda(theta, X_tilde, s, lam, vectorized=True):
    if vectorized:
        return X_tilde @ (s * dphi(s * (X_tilde.T @ theta))) + lam * theta
    grad = np.zeros_like(theta)
    for index, x_tilde in enumerate(X_tilde.T):
        grad += dphi(s[index] * np.dot(x_tilde, theta)) * s[index] * x_tilde
    return grad + lam * theta


def f_lambda_and_grad(theta, X_tilde, s, lam):
    """
    Vectorized f_lambda and grad f_lambda together to avoid redundant computations
    """
    z = s * (X_tilde.T @ theta)
    f = np.sum(phi(z)) + 0.5 * lam * np.dot(theta, theta)
    grad = X_tilde @ (s * dphi(z)) + lam * theta
    return f, grad


def lipschitz_constant(X, lam):
    """L = sigma_max(X)^2 + lam, via the top eigenvalue of X @ X.T (cheaper
    than an SVD of X since X has far fewer rows than columns here)."""
    sigma_max_sq = np.linalg.eigvalsh(X @ X.T)[-1]
    return sigma_max_sq + lam
