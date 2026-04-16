from src.features.clean import clean_columns
import pandas as pd

# -----------------------------
# Agrupación de géneros
# -----------------------------
def agrupar_genero(g):
    # 1. Pop, Rock y Alternativo (El núcleo comercial)
    if g in ['Pop', 'Dance', 'Indie', 'Alternative', 'Rock']:
        return 'pop_rock'
    
    # 2. Ritmos Urbanos, Latinos y del Caribe
    elif g in ['Hip-Hop', 'Rap', 'Reggaeton', 'Reggae', 'Ska']:
        return 'urban'
    
    # 3. Raíces, Tradicional y Soul (Orgánico/Vocal)
    elif g in ['Folk', 'Country', 'World', 'Blues', 'Soul', 'R&B', 'A Capella']:
        return 'roots_and_soul'
    
    # 4. Música Académica, Jazz y Bellas Artes
    elif g in ['Classical', 'Opera', 'Jazz']:
        return 'fine_arts'
    
    # 5. Multimedia, Cine y Animación
    elif g in ['Soundtrack', 'Movie', 'Anime']:
        return 'media_and_screen'
    
    # 6. Música Electrónica (Suele ser una clase muy pura en instrumentación)
    elif g in ['Electronic']:
        return 'electronic'
    
    # 7. Contenido Especial (Niños y Humor)
    # Incluimos ambas versiones de Children's Music por si acaso
    elif g in ["Children's Music", "Children’s Music", 'Comedy']:
        return 'specialty_content'
    
    # 8. Unclassified (Para evitar errores si entra un dato nuevo)
    return 'unclassified'


def apply_genre_grouping(df):
    df['genre_grouped'] = df['genre'].apply(agrupar_genero)
    return df


# -----------------------------
# Filtrado opcional
# -----------------------------
def remove_unwanted_classes(df):
    df = df[df['genre_grouped'] != 'unclassified'] 
    return df


# -----------------------------
# Separar X e y
# -----------------------------
def split_features_target(df):
    X = df.drop(columns=[
        'genre',
        'genre_grouped'
    ])
    
    y = df['genre_grouped']
    
    return X, y

def preprocessing():
    df = pd.read_csv("../data/processed/music_data_clean.csv")
    df = apply_genre_grouping(df)
    df = remove_unwanted_classes(df)
    x, y = split_features_target(df)
    return df.drop(columns=['genre']), x, y