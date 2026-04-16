
def clean_columns(df):

    df.columns = df.columns.str.strip()

    df['genre'] = df['genre'].str.replace('Children’s Music', "Children's Music")

    df=df.drop(columns=['artist_name','track_name','track_id'])

    df = df[df['time_signature'] != '0/4']
    df = df[df['duration_ms'] <= 1000000]

    return df