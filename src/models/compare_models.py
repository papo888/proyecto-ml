from src.models.logistic_model import train_logistic_model
from src.models.random_forest_model import train_random_forest_model
from src.models.gradient_boosting_model import train_gradient_boosting_model
from src.models.knn_model import train_knn_model


def compare_models(X, y):
    results = {}

    models = {
        "Logistic Regression": train_logistic_model,
        "Random Forest": train_random_forest_model,
        "Gradient Boosting": train_gradient_boosting_model,
        "KNN": train_knn_model
    }

    for name, train_function in models.items():
        print(f"Entrenando {name}...")
        model, report = train_function(X, y)

        results[name] = {
            "model": model,
            "report": report
        }

    return results