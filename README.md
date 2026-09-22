# Packages

- numpy
- scipy
- matplotlib

(install with `pip install numpy scipy matplotlib`)

# Running

```
cd code
python main.py
```

Runtime on our machine (MacBook, Apple Silicon): about 7 seconds (Q2-Q4;
Q4's gradient descent converges by the gradient-norm criterion in ~4.7s /
866 iterations, well before its 3-minute cap. Update this once `main.py`
covers all questions).

# Results files

- `results/q2_comparison.txt`: for Q2, the relative difference between the
  loop and vectorized implementations (objective and gradient), and their
  runtime comparison (`time.perf_counter()`, 5 repetitions, 1 warm-up run
  discarded).
- `results/q3_gradient_check.pdf`: for Q3, log-log plot of
  `|f_lambda(theta+t*v) - f_lambda(theta) - t*<v, grad f_lambda(theta)>|`
  vs. `t`, for `t = logspace(-8, 0, 101)`.
- `results/q3_gradient_check.csv`: the numerical data behind that plot, one
  row per `t`, columns `t,error` (header row included).
- `results/q4_gradient_descent.csv`: for Q4, one row per gradient descent
  iteration, columns `iter,objective,grad_norm`.
- `results/q4_summary.txt`: for Q4, the Lipschitz constant `L`, the constant
  step size `alpha = 1/L`, the stopping reason (gradient tolerance vs. the
  3-minute cap), iteration count, and elapsed time.
