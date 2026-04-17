from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
 
from src.models.model_utils import prepare_data, build_preprocessor, evaluate_model
 
 
def train_lightgbm_model(X, y):
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    preprocessor = build_preprocessor(X)
 
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(sampling_strategy='not majority', k_neighbors=3, random_state=42)),
        ('classifier', LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=50,          # controla complejidad del árbol
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,   # evita overfitting en clases pequeñas
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
            verbose=-1              # silencia los logs de entrenamiento
        ))
    ])
 
    model.fit(X_train, y_train)
    report = evaluate_model(model, X_test, y_test)
 
    return model, report