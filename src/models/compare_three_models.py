from src.models.random_forest_model import train_random_forest_model
from src.models.gradient_boosting_model import train_gradient_boosting_model
from src.models.XGBoost import train_xgboost_model

def compare_three_models(X, y):
    results = {}

    models = {
        "Random Forest": train_random_forest_model,
        "Gradient Boosting": train_gradient_boosting_model,
        "XGBoost": train_xgboost_model
    }

    for name, train_function in models.items():
        print(f"Entrenando {name}...")
        model, report = train_function(X, y)

        results[name] = {
            "model": model,
            "report": report
        }

    return results


