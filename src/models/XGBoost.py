from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import numpy as np

from src.models.model_utils import prepare_data, build_preprocessor, evaluate_model


def train_xgboost_model(X, y):
    # XGBoost necesita etiquetas numéricas (0, 1, 2, ...)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = prepare_data(X, y_encoded)
    preprocessor = build_preprocessor(X)

    n_classes = len(np.unique(y_encoded))

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(sampling_strategy='not majority', k_neighbors=3, random_state=42)),
        ('classifier', XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,       # regularización L1 para reducir overfitting
            reg_lambda=1.0,      # regularización L2
            objective='multi:softprob',
            num_class=n_classes,
            eval_metric='mlogloss',
            n_jobs=-1,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    report = evaluate_model(model, X_test, y_test)

    return model, report