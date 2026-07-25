import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

from src.features.geo import water_fraction, within_shape


def cat_diff(a, b, col):
    return set(b[col].dropna().unique()) - set(a[col].dropna().unique())


def split_strings(X, col, sep='/'):
    return X[col].fillna('').astype(str).str.split(sep)


def str_col_vals_to_lower(df):
    str_cols = df.select_dtypes(include='str').columns
    df[str_cols] = df[str_cols].apply(lambda x: x.str.lower())
    return df


def col_names_to_lower(df):
    df.columns = [c.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
                  for c in df.columns]

    return df


def fill_with_unknown(df, col):
    df[col] = df[col].fillna('unknown')
    return df


def fill_missing_coords(df, missing_coords_df):
    cols = ["field_name", "reservoir_unit", "latitude", "longitude"]
    key_cols = ["field_name", "reservoir_unit"]
    df_filled = df.merge(missing_coords_df[cols], on=key_cols, how='left', suffixes=('', '_fill'))
    df_filled['latitude'] = df_filled['latitude'].fillna(df_filled['latitude_fill'])
    df_filled['longitude'] = df_filled['longitude'].fillna(df_filled['longitude_fill'])
    df_filled.drop(columns=['latitude_fill', 'longitude_fill'], inplace=True)
    return df_filled


def fill_missing_basins(df, basin_locations_df):
    cols = ["basin_name", "field_name", "reservoir_unit"]
    key_cols = ["field_name", "reservoir_unit"]
    df = df.merge(basin_locations_df[cols], on=key_cols, how='left', suffixes=('', '_fill'))
    df['basin_name'] = df['basin_name'].fillna(df['basin_name_fill'])
    df.drop(columns=['basin_name_fill'], inplace=True)
    return df


def calc_water_features(gdf, water_feature_datasets):
    gdf['water_pct'] = gdf.apply(lambda row: water_fraction(point_lat=row.latitude,
                                                            point_lon=row.longitude,
                                                            water_shapes=water_feature_datasets["ocean_50m"],
                                                            radius_km=2), axis=1)
    gdf['is_in_water'] = gdf.apply(
        lambda row: within_shape(lat=row.latitude, lon=row.longitude, gdf=water_feature_datasets["ocean_50m"]),
        axis=1)
    gdf['is_on_island'] = gdf.apply(
        lambda row: within_shape(lat=row.latitude, lon=row.longitude, gdf=water_feature_datasets["all_islands"]),
        axis=1)
    gdf['is_in_gulf'] = gdf.apply(
        lambda row: within_shape(lat=row.latitude, lon=row.longitude, gdf=water_feature_datasets["gulfs"]),
        axis=1)
    gdf['is_in_strait'] = gdf.apply(
        lambda row: within_shape(lat=row.latitude, lon=row.longitude, gdf=water_feature_datasets["straits"]),
        axis=1)
    gdf['is_in_delta'] = gdf.apply(
        lambda row: within_shape(lat=row.latitude, lon=row.longitude, gdf=water_feature_datasets["deltas"]),
        axis=1)
    gdf['is_in_bay'] = gdf.apply(
        lambda row: within_shape(lat=row.latitude, lon=row.longitude, gdf=water_feature_datasets["bays"]),
        axis=1)
    return gdf


def calc_ref_features(gdf):
    names_concat = gdf['field_name'] + gdf['reservoir_unit'] + gdf['basin_name']
    gdf['ref_onshore'] = names_concat.str.contains('onshore')
    gdf['ref_offshore'] = names_concat.str.contains('offshore')
    gdf['ref_lake'] = names_concat.str.contains("lake")
    return gdf


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


def infer_basin_locations(gdf):
    basin_location = pd.crosstab(gdf['basin_name'], gdf['onshore_offshore']).drop('unknown',
                                                                                  errors='ignore').reset_index()

    loc_mask = ((basin_location['onshore'] > 0) &
                (basin_location['offshore'] > 0))

    basin_location.loc[loc_mask, 'onshore-offshore-calc'] = (
            basin_location.loc[loc_mask, 'onshore'] +
            basin_location.loc[loc_mask, 'offshore'] +
            basin_location.loc[loc_mask, 'onshore-offshore'])

    basin_location['onshore-offshore-calc'] = basin_location['onshore-offshore-calc'].fillna(0).astype(int)

    loc_mask_2 = ((basin_location['onshore'] == 0) &
                  (basin_location['offshore'] == 0) &
                  (basin_location['onshore-offshore'] > 0))

    basin_location.loc[loc_mask_2, 'onshore-offshore-calc'] = basin_location.loc[loc_mask_2, 'onshore-offshore']

    inferred_location_map = construct_basin_location_mapper(basin_location)
    return basin_location, inferred_location_map


def map_basin_location(gdf, inferred_basin_location_map, basin_location_map_combined):
    gdf['basin_location_inferred'] = gdf['basin_name'].map(inferred_basin_location_map).fillna("unknown")
    gdf['basin_location_external'] = gdf['basin_name'].map(basin_location_map_combined).fillna('unknown')
    return gdf


def rename_encoded_columns(tranformed_data, encoded_columns, col, index):
    renamed_columns = [f"{col}_{c}" for c in encoded_columns]
    return pd.DataFrame(tranformed_data, columns=renamed_columns, index=index)


def split_binarize_encode(df, col):
    df_with_split_col = split_strings(df, col)
    transformer = MultiLabelBinarizer()
    binarized = transformer.fit_transform(df_with_split_col)
    encoded_columns = transformer.classes_
    features = rename_encoded_columns(binarized, encoded_columns, col, df.index)
    return features, transformer


def split_binarize_encode_test_data(transformer, df, col):
    df_with_split_col = split_strings(df, col)
    transformed = transformer.transform(df_with_split_col)
    return rename_encoded_columns(transformed, transformer.classes_, col, df.index)


def onehot_transform_columns(transformer):
    return [c.replace(' ', '-').replace('/', '-') for c in
            transformer.named_steps['binarize'].categories_[0].tolist()]


def onehot_encode(df, col, handle_unknown='infrequent_if_exist'):
    transformer = Pipeline([('binarize', OneHotEncoder(handle_unknown=handle_unknown))])

    transformed = transformer.fit_transform(df[[col]]).toarray()
    encoded_columns = onehot_transform_columns(transformer)
    features = rename_encoded_columns(transformed, encoded_columns, col, df.index)

    return features, transformer


def onehot_encode_test_data(transformer, df, col):
    transformed = transformer.transform(df[[col]]).toarray()
    encoded_columns = onehot_transform_columns(transformer)
    return rename_encoded_columns(transformed, encoded_columns, col, df.index)
