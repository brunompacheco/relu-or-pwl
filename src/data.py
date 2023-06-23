import numpy as np
import pandas as pd


def load_riser_data(fpath):
    df = pd.read_csv(fpath, index_col=0)

    # change MARLIM variable names to more friendly descriptions
    df = df.rename(columns={'QLSC': 'qliq', 'PMONCKTP': 'psup', 'delta_P': 'delta_p'})

    # compute pressure drop, if not already there
    if not 'delta_p' in df.columns:
        df['delta_p'] = df['PFUNDO'] - df['psup']

    # filter only relevant info
    df = df[['psup', 'rgl', 'bsw', 'qliq', 'delta_p']]

    return df.round(6)

def split_curve(df: pd.DataFrame, reduction_ratio=1/2) -> pd.DataFrame:
    var_names = ['bsw', 'rgl', 'qliq', 'psup']

    df_train = df.copy().sort_values(by=var_names[::-1])
    df_test = df_train.copy()
    var_values_train = dict()
    for var in var_names:
        var_values = df_train[var].sort_values().unique()
        n_values = len(var_values)
        # gets evenly spaced values including extrema of the interval
        train_idx = np.linspace(0, n_values-1, int(n_values * reduction_ratio)).astype(int)
        var_values_train[var] = var_values[train_idx]

    for var, values_train in var_values_train.items():
        df_train = df_train[df_train[var].isin(values_train)]
    df_test = df_test[~df_test.index.isin(df_train.index)]

    return df_train, df_test

def get_X_y(df: pd.DataFrame):
    var_names = ['bsw', 'rgl', 'qliq', 'psup']
    target_name = 'delta_p'

    X = df[var_names].values
    y = df[target_name].values.reshape(-1,1)

    return X, y
