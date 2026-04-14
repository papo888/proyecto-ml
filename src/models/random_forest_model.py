from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.models.model_utils import prepare_data, build_preprocessor, evaluate_model


def train_random_forest_model(X, y):
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    preprocessor = build_preprocessor(X)

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    report = evaluate_model(model, X_test, y_test)

    return model, report