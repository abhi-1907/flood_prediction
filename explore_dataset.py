import pandas as pd
import json

df = pd.read_csv('india_pakistan_flood_balancednew.csv')

info = {
    'shape': list(df.shape),
    'columns': list(df.columns),
    'dtypes': {c: str(t) for c, t in df.dtypes.items()},
    'numeric_cols': df.select_dtypes(include='number').columns.tolist(),
    'object_cols': df.select_dtypes(include='object').columns.tolist(),
    'means': {c: round(float(v), 4) for c, v in df.select_dtypes(include='number').mean().items()},
    'mins': {c: round(float(v), 4) for c, v in df.select_dtypes(include='number').min().items()},
    'maxs': {c: round(float(v), 4) for c, v in df.select_dtypes(include='number').max().items()},
    'nulls': {c: int(v) for c, v in df.isnull().sum().items()},
    'sample_row0': {c: str(df[c].iloc[0]) for c in df.columns},
    'sample_row1': {c: str(df[c].iloc[1]) for c in df.columns},
}

if 'flood_occurred' in df.columns:
    info['target_vc'] = df['flood_occurred'].value_counts().to_dict()

print(json.dumps(info, indent=2))
