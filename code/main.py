from model import load_data, labels_to_signs, LAMBDA
import q2
import q3
import q4


def main():
    X_train, y_train, X_test, y_test = load_data()
    s_train = labels_to_signs(y_train)

    q2.run(X_train, s_train, LAMBDA)
    q3.run(X_train, s_train, LAMBDA)
    q4.run(X_train, s_train, LAMBDA)


if __name__ == "__main__":
    main()
