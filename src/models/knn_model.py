from imblearn.pipeline import Pipeline  # CORREGIDO: antes era sklearn.pipeline
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier

from src.models.model_utils import prepare_data, build_preprocessor, evaluate_model


def train_knn_model(X, y):
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    preprocessor = build_preprocessor(X)

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(sampling_strategy='not majority', k_neighbors=3, random_state=42)),
        ('classifier', KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',  # penaliza vecinos lejanos, mejora con clases desbalanceadas
            metric='euclidean',
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    report = evaluate_model(model, X_test, y_test)

    return model, report