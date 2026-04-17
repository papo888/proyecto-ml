from imblearn.pipeline import Pipeline  # 🔥 IMPORTANTE
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

from src.models.model_utils import prepare_data, build_preprocessor, evaluate_model


def train_gradient_boosting_model(X, y):
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    preprocessor = build_preprocessor(X)

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    report = evaluate_model(model, X_test, y_test)

    return model, report