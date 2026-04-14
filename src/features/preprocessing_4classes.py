import pandas as pd

# -----------------------------
# Limpieza básica
# -----------------------------
def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df


# -----------------------------
# Agrupación de géneros en 4 clases
# -----------------------------
def agrupar_genero_4(g):
    if g in ['Pop', 'Dance', 'Rock', 'Alternative', 'Indie', 'R&B', 'Soul']:
        return 'mainstream'
    
    elif g in ['Hip-Hop', 'Rap', 'Reggaeton', 'Reggae', 'Ska']:
        return 'urban'
    
    elif g in ['Classical', 'Opera', 'Jazz', 'Blues', 'Folk', 'Country']:
        return 'acoustic'
    
    elif g in ['Electronic', 'Soundtrack', 'Movie', 'Anime', 'Children’s Music', "Children's Music", 'World', 'Comedy']:
        return 'other'
    
    else:
        return 'other'


def apply_genre_grouping_4(df):
    df['genre'] = df['genre'].str.strip()
    df['genre_grouped_4'] = df['genre'].apply(agrupar_genero_4)
    return df


# -----------------------------
# Filtrado opcional
# -----------------------------
def remove_unwanted_classes_4(df):
    df = df[df['genre_grouped_4'].notna()]
    return df


# -----------------------------
# Separar X e y
# -----------------------------
def split_features_target_4(df):
    X = df.drop(columns=[
        'artist_name',
        'track_name',
        'track_id',
        'genre',
        'genre_grouped_4'
    ])
    
    y = df['genre_grouped_4']
    
    return X, y