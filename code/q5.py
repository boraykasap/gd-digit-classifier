import numpy as np
import matplotlib.pyplot as plt
import time


def plot(obj_hist, gradnorm_hist):
    print("Question 5 : started rendering plots ...")
    t0 = time.perf_counter()

    k = np.arange(len(obj_hist))

    plt.figure(figsize=(10, 4))

    # Panel 1 : objective values
    plt.subplot(1, 2, 1)
    plt.plot(k, obj_hist)
    plt.xlabel("Iteration k")
    plt.ylabel(r"$f_\lambda(\theta_k)$")
    plt.yscale("log")

    plt.title("Objective value vs iteration")

    # Panel 2 : gradient norms
    plt.subplot(1, 2, 2)
    plt.plot(k, gradnorm_hist)
    plt.xlabel("Iteration k")
    plt.ylabel(r"$\|\nabla f_\lambda(\theta_k)\|$")
    plt.yscale("log")
    plt.title("Gradient norm vs iteration")

    plt.tight_layout()
    plt.savefig("../results/q5_convergence.pdf")
    # plt.show()
    print(
        f"Question 5 done : saved plot as ../results/q5_convergence.txt : rendering plot took {
            (time.perf_counter() - t0):.4f} s"
    )
