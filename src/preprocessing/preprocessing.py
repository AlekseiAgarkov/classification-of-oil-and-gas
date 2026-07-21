import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer, OneHotEncoder


def cat_diff(a, b, col):
    return set(b[col].dropna().unique()) - set(a[col].dropna().unique())


def split_strings(X, col, sep='/'):
    return X[col].fillna('').astype(str).str.split(sep)


def col_splitter(col, sep='/'):
    return FunctionTransformer(split_strings, kw_args={'col': col, 'sep': sep})


def binarizer():
    return FunctionTransformer(MultiLabelBinarizer().fit_transform)


def construct_basin_location_mapper(df):
    basin_location_info = df[['basin_name', 'onshore', 'offshore', 'onshore-offshore-calc']].to_dict("records")
    basin_location_mapper = {}

    for info in basin_location_info:
        name = info['basin_name']
        if info['onshore-offshore-calc'] > 0:
            location = 'onshore-offshore'
        elif info['onshore'] > 0 and info['offshore'] == 0:
            location = 'onshore'
        elif info['onshore'] == 0 and info['offshore'] > 0:
            location = 'offshore'
        basin_location_mapper[name] = location

    return basin_location_mapper


def construct_basin_location_mapper_with_external(df: pd.DataFrame, external_mapping: dict):
    basin_location_info = df[['basin_name', 'onshore', 'offshore', 'onshore-offshore-calc']].to_dict("records")
    basin_location_mapper = {}

    for info in basin_location_info:
        name = info['basin_name']

        if name in external_mapping.keys():
            location = external_mapping[name]
        elif info['onshore-offshore-calc'] > 0:
            location = 'onshore-offshore'
        elif info['onshore'] > 0 and info['offshore'] == 0:
            location = 'onshore'
        elif info['onshore'] == 0 and info['offshore'] > 0:
            location = 'offshore'
        basin_location_mapper[name] = location

    return basin_location_mapper


def split_binarize_encode(df, col):
    binarizer_ = Pipeline([
        ('extract', col_splitter(col=col)),
        ('binarize', binarizer())
    ])

    binarized = binarizer_.fit_transform(df)

    features = pd.DataFrame(binarized,
                            columns=[f"{col}_{c.replace(' ', '-')}" for c in
                                     binarizer_.named_steps['binarize'].func.__self__.classes_.tolist()],
                            index=df.index)
    return features, binarizer_


def onehot_encode(df, col, handle_unknown='infrequent_if_exist'):
    binarizer = Pipeline([('binarize', OneHotEncoder(handle_unknown=handle_unknown))])

    binarized = binarizer.fit_transform(df[[col]]).toarray()

    features = pd.DataFrame(
        binarized,
        columns=[f"{col}_{c.replace(' ', '-').replace('/', '-')}" for c in
                 binarizer.named_steps['binarize'].categories_[0].tolist()],
        index=df.index)
    return features, binarizer
