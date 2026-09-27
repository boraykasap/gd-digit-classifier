import numpy as np
from model import RESULTS_DIR
import time


def predict(X_tilde, theta):
    scores = X_tilde.T @ theta
    return (scores > 0).astype(int)


def classification_error(pred, true_labels):
    return np.mean(pred != true_labels)


def run(X_train, y_train, X_test, y_test, theta_final):
    print("Question 7 : started computing the predictions ...")
    t0 = time.perf_counter()
    # Predictions
    y_pred_train = predict(X_train, theta_final)
    y_pred_test = predict(X_test, theta_final)

    # Error
    err_train = classification_error(y_pred_train, y_train)
    err_test = classification_error(y_pred_test, y_test)

    # Save outputs
    np.save(RESULTS_DIR + "theta_final.npy", theta_final)
    with open(RESULTS_DIR + "q7_errors.txt", "w") as result_file:
        result_file.write(f"train error: {err_train}\n")
        result_file.write(f"test error: {err_test}\n")
    print(
        f"""Question 7 done : saved prediction error as ../results/q7_errors.txt and
        saved the final theta as ../results/theta_final.npy : computing prediction took {
            (time.perf_counter() - t0):.4f} s"""
    )
