from model import load_data, labels_to_signs, LAMBDA
import time
import q2
import q3
import q4
import q5
import q7


def main():
    print("Loading data ...")
    t0 = time.perf_counter()
    X_train, y_train, X_test, y_test = load_data()
    s_train = labels_to_signs(y_train)
    print(f"Finished loading data : took {(time.perf_counter() - t0):.4f} s")

    q2.run(X_train, s_train, LAMBDA)
    q3.run(X_train, s_train, LAMBDA)
    theta_final, obj_hist, gradnorm_hist = q4.run(X_train, s_train, LAMBDA)
    q5.plot(obj_hist, gradnorm_hist)

    q7.run(X_train, y_train, X_test, y_test, theta_final)
    print(f"Overall the computations took : {(time.perf_counter() - t0):.4f} s")


if __name__ == "__main__":
    main()
