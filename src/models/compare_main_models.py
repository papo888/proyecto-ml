from src.models.gradient_boosting_model import train_gradient_boosting_model
from src.models.lightgbm_model import train_lightgbm_model
from src.models.XGBoost import train_xgboost_model


def compare_main_models(X, y):
    """
    Compara solo los modelos de boosting, que son los que mejor funcionan
    para clasificación de géneros musicales.
    """
    results = {}

    models = {
        "Gradient Boosting": train_gradient_boosting_model,
        "LightGBM":          train_lightgbm_model,
        "XGBoost":           train_xgboost_model,
    }

    for name, train_function in models.items():
        print(f"Entrenando {name}...")
        model, report = train_function(X, y)

        results[name] = {
            "model":  model,
            "report": report
        }

    return results