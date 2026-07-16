from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer


def cat_diff(a, b, col):
    return set(b[col].dropna().unique()) - set(a[col].dropna().unique())


def split_strings(X, col, sep='/'):
    return X[col].fillna('').astype(str).str.split(sep)


def col_splitter(col, sep='/'):
    return FunctionTransformer(split_strings, kw_args={'col': col, 'sep': sep})


def binarizer():
    return FunctionTransformer(MultiLabelBinarizer().fit_transform)
