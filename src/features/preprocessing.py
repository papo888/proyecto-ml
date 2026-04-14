import pandas as pd

# -----------------------------
# Limpieza básica
# -----------------------------
def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df


# -----------------------------
# Agrupación de géneros
# -----------------------------
def agrupar_genero(g):
    if g in ['Pop', 'Dance', 'Rock', 'Alternative', 'Indie', 'R&B', 'Soul']:
        return 'mainstream'
    
    elif g in ['Hip-Hop', 'Rap']:
        return 'hiphop'
    
    elif g in ['Electronic']:
        return 'electronic'
    
    elif g in ['Jazz', 'Blues']:
        return 'jazz_blues'
    
    elif g in ['Reggaeton']:
        return 'latin'
    
    elif g in ['Reggae', 'Ska']:
        return 'reggae'
    
    elif g in ['Classical', 'Opera']:
        return 'classical'
    
    elif g in ['Folk', 'Country']:
        return 'folk_country'
    
    elif g in ['Soundtrack', 'Movie', 'Anime']:
        return 'soundtrack'
    
    elif g in ['Children’s Music', "Children's Music"]:
        return 'kids'
    
    elif g in ['Comedy']:
        return 'comedy'
    
    elif g in ['World']:
        return 'world'
    
    else:
        return 'other'


def apply_genre_grouping(df):
    df['genre'] = df['genre'].str.strip()
    df['genre_grouped'] = df['genre'].apply(agrupar_genero)
    return df


# -----------------------------
# Filtrado opcional
# -----------------------------
def remove_unwanted_classes(df):
    df = df[df['genre_grouped'] != 'other']
    df = df[df['genre_grouped'] != 'comedy']  # opcional
    return df


# -----------------------------
# Separar X e y
# -----------------------------
def split_features_target(df):
    X = df.drop(columns=[
        'artist_name',
        'track_name',
        'track_id',
        'genre',
        'genre_grouped'
    ])
    
    y = df['genre_grouped']
    
    return X, y