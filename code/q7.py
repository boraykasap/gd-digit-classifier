import numpy as np
from model import load_data

def predict(theta, X):
    scores = X.T @ theta
    return (scores>0).astype(int)

def classification_error(pred,true_labels):
    return np.mean(pred != true_labels)

def run(theta_final):
    X_train, y_train, X_test, y_test = load_data()

    # Predictions 
    y_pred_train = predict(theta_final,X_train)
    y_pred_test = predict(theta_final, X_test)

    # Error
    err_train = classification_error(y_pred_train, y_train)
    err_test = classification_error(y_pred_test, y_test)

    # Save outputs
    np.save("../results/theta_final.npy", theta_final)
    with open("../results/q7_errors.txt", "w") as f:
        f.write(f"train_error: {err_train}\n")
        f.write(f"test_error: {err_test}\n")

    return err_train, err_test
