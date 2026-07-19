import pandas as pd
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer


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