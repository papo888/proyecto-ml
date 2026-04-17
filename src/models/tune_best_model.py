import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

from src.models.model_utils import prepare_data, build_preprocessor, evaluate_model
from src.models.model_utilss import get_train_test_metrics


def tune_lightgbm(X, y, n_iter=30, cv=3):
    """
    Busca los mejores hiperparámetros para LightGBM usando RandomizedSearchCV.

    Parámetros:
        X       : features (ya con add_features aplicado)
        y       : variable objetivo
        n_iter  : cuántas combinaciones probar (más = mejor pero más lento)
        cv      : número de folds de validación cruzada

    Retorna:
        best_model  : pipeline entrenado con los mejores parámetros
        best_params : diccionario con los hiperparámetros óptimos
        report      : classification report en test
        metrics     : comparación train vs test (detecta overfitting)
    """
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    preprocessor = build_preprocessor(X)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(sampling_strategy='not majority', k_neighbors=3, random_state=42)),
        ('classifier', LGBMClassifier(
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
            verbose=-1
        ))
    ])

    param_dist = {
        'classifier__n_estimators':    randint(200, 600),
        'classifier__learning_rate':   uniform(0.02, 0.10),
        'classifier__max_depth':       randint(4, 10),
        'classifier__num_leaves':      randint(30, 90),
        'classifier__subsample':       uniform(0.6, 0.35),
        'classifier__colsample_bytree': uniform(0.6, 0.35),
        'classifier__reg_alpha':       uniform(0.0, 0.5),
        'classifier__reg_lambda':      uniform(0.5, 2.0),
        'classifier__min_child_samples': randint(10, 50),
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print(f"Buscando mejores hiperparámetros ({n_iter} combinaciones, {cv} folds)...")
    search.fit(X_train, y_train)

    best_model  = search.best_estimator_
    best_params = search.best_params_

    print(f"\nMejores parámetros encontrados:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    report  = evaluate_model(best_model, X_test, y_test)
    metrics = get_train_test_metrics(best_model, X_train, y_train, X_test, y_test)

    print(f"\nResultados en test:")
    print(f"  F1 Macro:    {report['macro avg']['f1-score']:.4f}")
    print(f"  F1 Weighted: {report['weighted avg']['f1-score']:.4f}")
    print(f"  Accuracy:    {report['accuracy']:.4f}")
    print(f"\nDiagnóstico: {metrics['diagnostico']}")

    return best_model, best_params, report, metrics


def summarize_results(results: dict) -> pd.DataFrame:
    """
    Convierte el dict de resultados de compare_* en un DataFrame ordenado.
    Úsalo así:
        results = compare_models(X, y)
        df = summarize_results(results)
    """
    rows = []
    for name, res in results.items():
        r = res["report"]
        rows.append({
            "Modelo":       name,
            "Accuracy":     round(r["accuracy"], 4),
            "Macro Recall": round(r["macro avg"]["recall"], 4),
            "Macro F1":     round(r["macro avg"]["f1-score"], 4),
            "Weighted F1":  round(r["weighted avg"]["f1-score"], 4),
        })
    return pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)