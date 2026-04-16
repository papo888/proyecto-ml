import pandas as pd
from src.features.clean import clean_columns

# -----------------------------
# Agrupación de géneros en 4 clases
# -----------------------------
def agrupar_genero(g):
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


def apply_genre_grouping(df):
    df['genre_grouped_4'] = df['genre'].apply(agrupar_genero)
    return df


# -----------------------------
# Filtrado opcional
# -----------------------------
def remove_unwanted_classes(df):
    df = df[df['genre_grouped_4'].notna()]
    return df


# -----------------------------
# Separar X e y
# -----------------------------
def split_features_target(df):
    X = df.drop(columns=[
        'genre',
        'genre_grouped_4'
    ])
    
    y = df['genre_grouped_4']
    
    return X, y

def preprocessing():
    df = pd.read_csv("../data/processed/music_data_clean.csv")
    df = apply_genre_grouping(df)
    df = remove_unwanted_classes(df)
    x, y = split_features_target(df)
    return df.drop(columns=['genre']), x, y