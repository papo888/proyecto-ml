from sklearn.metrics import accuracy_score, f1_score


def get_train_test_metrics(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "train_f1_macro": f1_score(y_train, y_train_pred, average="macro"),
        "test_f1_macro": f1_score(y_test, y_test_pred, average="macro"),
        "train_f1_weighted": f1_score(y_train, y_train_pred, average="weighted"),
        "test_f1_weighted": f1_score(y_test, y_test_pred, average="weighted"),
    }

    return metrics