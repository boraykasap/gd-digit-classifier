import time
import platform

import numpy as np

from model import f_lambda, grad_f_lambda, RESULTS_DIR


def compare_loop_vs_vectorized_mutual_error(theta, X_tilde, s, lam):
    f_lambda_loop, f_lambda_vec = (
        f_lambda(
            theta,
            X_tilde,
            s,
            lam,
            False,
        ),
        f_lambda(
            theta,
            X_tilde,
            s,
            lam,
            True,
        ),
    )
    grad_f_lambda_loop, grad_f_lambda_vec = (
        grad_f_lambda(
            theta,
            X_tilde,
            s,
            lam,
            False,
        ),
        grad_f_lambda(
            theta,
            X_tilde,
            s,
            lam,
            True,
        ),
    )
    f_lambda_abs_diff = abs(f_lambda_loop - f_lambda_vec)
    f_lambda_rel_diff = f_lambda_abs_diff / abs(f_lambda_vec)
    grad_f_lambda_abs_diff = np.linalg.norm(grad_f_lambda_loop - grad_f_lambda_vec)
    grad_f_lambda_rel_diff = grad_f_lambda_abs_diff / np.linalg.norm(grad_f_lambda_vec)
    return (
        f_lambda_abs_diff,
        f_lambda_rel_diff,
        grad_f_lambda_rel_diff,
        grad_f_lambda_abs_diff,
    )


def benchmark_f_lambda_grad_f_lambda(
    theta,
    X_tilde,
    s,
    lam,
    nbr_reps,
    vectorized,
):
    # warm-up, discarded is it useful ??
    # f_lambda(theta, X_tilde, s, lam, vectorized)
    # grad_f_lambda(theta, X_tilde, s, lam, vectorized)

    t0 = time.perf_counter()
    for _ in range(nbr_reps):
        f_lambda(theta, X_tilde, s, lam, vectorized)
        grad_f_lambda(theta, X_tilde, s, lam, vectorized)
    return (time.perf_counter() - t0) / float(nbr_reps)


def compare_loop_vs_vectorized(theta, X_tilde, s, lam, nbr_reps):
    t0 = time.perf_counter()
    (
        f_lambda_abs_diff,
        f_lambda_rel_diff,
        grad_f_lambda_rel_diff,
        grad_f_lambda_abs_diff,
    ) = compare_loop_vs_vectorized_mutual_error(
        theta,
        X_tilde,
        s,
        lam,
    )

    loop_time, vec_time = (
        benchmark_f_lambda_grad_f_lambda(
            theta,
            X_tilde,
            s,
            lam,
            nbr_reps,
            False,
        ),
        benchmark_f_lambda_grad_f_lambda(
            theta,
            X_tilde,
            s,
            lam,
            nbr_reps,
            True,
        ),
    )
    return {
        "f_lambda_abs_diff": f_lambda_abs_diff,
        "f_lambda_rel_diff": f_lambda_rel_diff,
        "grad_f_lambda_abs_diff": grad_f_lambda_abs_diff,
        "grad_f_lambda_rel_diff": grad_f_lambda_rel_diff,
        "loop_time_sec": loop_time,
        "vec_time_sec": vec_time,
        "nbr_reps": nbr_reps,
        "speedup": loop_time / vec_time,
        "total_benchmark_time": time.perf_counter() - t0,
    }


def save_comparison(r):
    # NOTE: that platform.platform() and platform.processor() are not always
    #       able to retrieve information on specific machines
    lines = [
        f"machine: {platform.platform()} / {platform.processor()}\n",
        f"timer: time.perf_counter(), {r['nbr_reps']} repetitions\n",
        # f"f_lambda absolute difference |f_loop - f_vec| : {
        #     r['f_lambda_abs_diff']:.4e} \n",
        f"f_lambda relative difference |f_loop - f_vec| / |f_vec|: {
            r['f_lambda_rel_diff']:.4e}\n",
        # f"gradient f_lambda relative difference ||grad_f_loop - grad_f_vec|| / ||grad_vec||: {
        #     r['grad_f_lambda_abs_diff']:.4e}\n",
        f"gradient f_lambda relative difference ||grad_f_loop - grad_vec|| / ||grad_f_vec||: {
            r['grad_f_lambda_rel_diff']:.4e}\n",
        f"loop time (s, mean of {r['nbr_reps']} reps): {r['loop_time_sec']:.4f}\n",
        f"vectorized time (s, mean of {r['nbr_reps']} reps): {r['vec_time_sec']:.4f}\n",
        f"speedup (loop / vectorized): {r['speedup']:.3f}x\n",
        f"total benchmark time {r['total_benchmark_time']:.3f}\n",
    ]
    with open(RESULTS_DIR + "q2_comparison.txt", "w") as result_file:
        result_file.writelines(lines)


def run(X_train, s_train, lam):
    print("Question 2 : starting computation ...")
    t0 = time.perf_counter()
    theta = np.random.default_rng(0).standard_normal(X_train.shape[0]) * 0.01
    save_comparison(compare_loop_vs_vectorized(theta, X_train, s_train, lam, 25))
    print(
        f"Question 2 done : saved results as ../results/q2_comparison.txt : computation took {
            (time.perf_counter() - t0):.4f} s"
    )
