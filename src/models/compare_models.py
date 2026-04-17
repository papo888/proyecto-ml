from src.models.logistic_model import train_logistic_model
from src.models.random_forest_model import train_random_forest_model
from src.models.gradient_boosting_model import train_gradient_boosting_model
from src.models.knn_model import train_knn_model
from src.models.lightgbm_model import train_lightgbm_model


def compare_models(X, y):
    """
    Compara los 5 modelos base.
    Usa add_features() de model_utils sobre X antes de llamar esta función
    para aprovechar las features de interacción.
    """
    results = {}

    models = {
        "Logistic Regression": train_logistic_model,
        "Random Forest":       train_random_forest_model,
        "Gradient Boosting":   train_gradient_boosting_model,
        "KNN":                 train_knn_model,
        "LightGBM":            train_lightgbm_model,
    }

    for name, train_function in models.items():
        print(f"Entrenando {name}...")
        model, report = train_function(X, y)

        results[name] = {
            "model":  model,
            "report": report
        }

    return results