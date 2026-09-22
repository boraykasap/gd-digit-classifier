import time
import platform

import numpy as np

from model import f_lambda, grad_f_lambda, RESULTS_DIR


def compare_loop_vs_vectorized(theta, X, s, lam, n_reps=5):
    f_loop, f_vec = f_lambda(theta, X, s, lam, False), f_lambda(theta, X, s, lam, True)
    g_loop, g_vec = grad_f_lambda(theta, X, s, lam, False), grad_f_lambda(theta, X, s, lam, True)

    obj_rel_diff = abs(f_loop - f_vec) / abs(f_vec)
    grad_rel_diff = np.linalg.norm(g_loop - g_vec) / np.linalg.norm(g_vec)

    def time_calls(vectorized):
        f_lambda(theta, X, s, lam, vectorized)  # warm-up, discarded
        grad_f_lambda(theta, X, s, lam, vectorized)
        t0 = time.perf_counter()
        for _ in range(n_reps):
            f_lambda(theta, X, s, lam, vectorized)
            grad_f_lambda(theta, X, s, lam, vectorized)
        return (time.perf_counter() - t0) / n_reps

    loop_time, vec_time = time_calls(False), time_calls(True)
    return {
        "obj_rel_diff": obj_rel_diff,
        "grad_rel_diff": grad_rel_diff,
        "loop_time_sec": loop_time,
        "vec_time_sec": vec_time,
        "n_reps": n_reps,
        "speedup": loop_time / vec_time,
    }


def save_comparison(r):
    lines = [
        f"machine: {platform.platform()} / {platform.processor()}",
        f"timer: time.perf_counter(), {r['n_reps']} repetitions, 1 warm-up run discarded",
        f"objective relative difference |f_loop - f_vec| / |f_vec|: {r['obj_rel_diff']:.3e}",
        f"gradient relative difference ||g_loop - g_vec|| / ||g_vec||: {r['grad_rel_diff']:.3e}",
        f"loop time (s, mean of {r['n_reps']} reps): {r['loop_time_sec']:.4f}",
        f"vectorized time (s, mean of {r['n_reps']} reps): {r['vec_time_sec']:.4f}",
        f"speedup (loop / vectorized): {r['speedup']:.1f}x",
    ]
    (RESULTS_DIR / "q2_comparison.txt").write_text("\n".join(lines) + "\n")


def run(X_train, s_train, lam):
    theta = np.random.default_rng(0).standard_normal(X_train.shape[0]) * 0.01
    save_comparison(compare_loop_vs_vectorized(theta, X_train, s_train, lam))
