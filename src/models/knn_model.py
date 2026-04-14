from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from src.models.model_utils import prepare_data, build_preprocessor, evaluate_model


def train_knn_model(X, y):
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    preprocessor = build_preprocessor(X)

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=5))
    ])

    model.fit(X_train, y_train)
    report = evaluate_model(model, X_test, y_test)

    return model, report