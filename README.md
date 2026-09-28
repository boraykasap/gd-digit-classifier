# Packages 
Using python 3.13 and onwards. 
Packages required are : 

* numpy
* scipy
* matplotlib

# Running

```
cd code
python main.py
```

Runtime on our machine (MacBook, Apple Silicon): about 7 seconds (Q2-Q4;
Q4's gradient descent converges by the gradient-norm criterion in ~4.7s /
867 iterations, well before its 3-minute cap. Update this once `main.py`
c

| Machine                                              | Runtime                      |
| ---------------------------------------------------- |:----------------------------:|
| Intel(R) Core(TM) Ultra 9 185H (16) @ 5.10 GHz       | 6.7995 s over 867 iterations |
| MacBook Apple Silicon                                | 4.7940 s over 867 iterations |
| Intel(R) 13th Gen Core(TM) i9-13900H (20) @ 5.40 GHz | 2.7790 s over 867 iterations |

# Results files

- `results/q2_comparison.txt`: for Q2, containing the relative difference between the
  loop and vectorized implementations (objective and gradient), and their
  runtime comparison over 100 repetition using`time.perf_counter()`.
- `results/q3_gradient_check.pdf`: for Q3, containing log-log plot of
  `|f_lambda(theta+t*v) - f_lambda(theta) - t*<v, grad f_lambda(theta)>|`
  vs. `t`, for `t = logspace(-8, 0, 101)`.
- `results/q3_gradient_check.csv`: for Q3, containing the numerical data behind that plot, one
  row per `t`, columns `t,error` (header row included).
- `results/q4_gradient_descent.csv`: for Q4, one row per gradient descent
  iteration, columns `iter,objective,grad_norm`.
- `results/q4_summary.txt`: for Q4, containing the Lipschitz constant `L`, the constant
  step size `alpha = 1/L`, the stopping reason (gradient tolerance vs. the
  3-minute cap), iteration count, and elapsed time.
- `results/q5_convergence.pdf`: for Q5, containing the two-pannels plots of `f_lambda(theta_k)` and `||grad f_lambda(theta_k)||`
- `results/theta_final.npy` : for Q7, containing the final parameter vector `theta_final` obtained from the run of Q5
- `results/q7_errors.txt` : for Q7, the classification error rates (for train and test)
