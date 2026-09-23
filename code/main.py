from model import load_data, labels_to_signs, LAMBDA
import q2
import q3
import q4
from q5 import q5_plot


def main():
    X_train, y_train, X_test, y_test = load_data()
    s_train = labels_to_signs(y_train)

    q2.run(X_train, s_train, LAMBDA)
    q3.run(X_train, s_train, LAMBDA)
    theta_final, obj_hist, gradnorm_hist = q4.run(X_train, s_train, LAMBDA)
    q5_plot(obj_hist, gradnorm_hist)
    

if __name__ == "__main__":
    main()
