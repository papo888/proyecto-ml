import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def get_train_test_metrics(model, X_train, y_train, X_test, y_test):
    """
    Compara métricas de train vs test para detectar overfitting.

    Cómo interpretar los resultados:
      - train_f1 >> test_f1 (diferencia > 0.10) → overfitting
        → solución: bajar max_depth, subir min_samples_leaf, bajar n_estimators
      - train_f1 ≈ test_f1 pero ambos bajos → underfitting
        → solución: más features, modelo más complejo, más n_estimators
      - train_f1 ≈ test_f1 y ambos altos → modelo bien calibrado
    """
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    metrics = {
        "train_accuracy":    accuracy_score(y_train, y_train_pred),
        "test_accuracy":     accuracy_score(y_test,  y_test_pred),
        "train_f1_macro":    f1_score(y_train, y_train_pred, average="macro"),
        "test_f1_macro":     f1_score(y_test,  y_test_pred,  average="macro"),
        "train_f1_weighted": f1_score(y_train, y_train_pred, average="weighted"),
        "test_f1_weighted":  f1_score(y_test,  y_test_pred,  average="weighted"),
    }

    gap_f1 = metrics["train_f1_macro"] - metrics["test_f1_macro"]
    metrics["overfitting_gap"] = round(gap_f1, 4)

    if gap_f1 > 0.10:
        metrics["diagnostico"] = "OVERFITTING — reducir complejidad del modelo"
    elif metrics["test_f1_macro"] < 0.55:
        metrics["diagnostico"] = "UNDERFITTING — agregar features o aumentar complejidad"
    else:
        metrics["diagnostico"] = "OK — modelo bien calibrado"

    return pd.Series(metrics)