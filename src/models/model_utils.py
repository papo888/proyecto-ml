import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
 
 
def add_features(X):
    """
    Agrega features de interacción que mejoran la separabilidad entre géneros.
    Llama esta función ANTES de pasar X a cualquier modelo.
    """
    X = X.copy()
 
    # Interacciones energéticas
    X['energy_dance']           = X['energy'] * X['danceability']
    X['mood']                   = X['valence'] * X['energy']
    X['loud_energy']            = X['loudness'] * X['energy']
    X['loud_dance']             = X['loudness'] * X['danceability']
 
    # Acústica vs energía (separa bien géneros como classical, folk, electronic)
    X['acoustic_quiet']         = X['acousticness'] * (1 - X['energy'])
 
    # Ratio habla vs instrumental (útil para rap, spoken word, classical)
    X['speech_instrument_ratio'] = X['speechiness'] / (X['instrumentalness'] + 0.01)
 
    # Tempo y baile combinados (diferencia bien EDM de baladas)
    X['tempo_dance']            = X['tempo'] * X['danceability']
 
    return X
 
 
def prepare_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test
 
 
def build_preprocessor(X):
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
 
    transformers = [('num', StandardScaler(), num_cols)]
    if cat_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
 
    preprocessor = ColumnTransformer(transformers=transformers)
 
    return preprocessor
 
 
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report