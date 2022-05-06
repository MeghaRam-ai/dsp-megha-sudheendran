from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import joblib
import pandas as pd
import os


def drop_unrelevent_features(X: pd.DataFrame) -> pd.DataFrame:
    #  most unrelated features found by analysing
    #  the correlation between features
    most_unrelated_features = ['BsmtFinSF2',
                               'BsmtFinType1',
                               'BsmtFinType2',
                               'BsmtUnfSF',
                               'GarageYrBlt',
                               'GarageFinish',
                               'BsmtQual',
                               'GarageQual',
                               'GarageCond',
                               'Functional',
                               'BedroomAbvGr',
                               'KitchenAbvGr',
                               'HalfBath',
                               'BsmtHalfBath',
                               'BsmtFullBath',
                               'LowQualFinSF',
                               'Heating',
                               'MSSubClass',
                               'MSZoning',
                               'LotShape',
                               'LandContour',
                               'LotConfig',
                               'LandSlope',
                               'Neighborhood',
                               'Condition1',
                               'BldgType',
                               'HouseStyle',
                               'OverallCond',
                               'LotArea',
                               'RoofStyle',
                               'Exterior1st',
                               'Exterior2nd',
                               'ExterCond',
                               'BsmtCond',
                               'BsmtExposure',
                               'SaleType',
                               'MoSold',
                               'MiscVal',
                               'MiscFeature',
                               'Fence',
                               'PoolQC',
                               'PoolArea',
                               'ScreenPorch',
                               '3SsnPorch']
    # drop the most unrelated features
    X = X.drop(most_unrelated_features, axis=1)
    X = X.drop(['GarageType',
                'Street',
                'Condition2',
                'MasVnrType',
                'RoofMatl',
                'Id',
                'Alley',
                'LotFrontage',
                'FireplaceQu',
                'YrSold',
                'Utilities'], axis=1)
    # fill the fetaure columns with null values with most occuring entry
    X['MasVnrArea'].fillna(X['MasVnrArea'].value_counts().idxmax(),
                           inplace=True)
    X['Electrical'].fillna(X['Electrical'].value_counts().idxmax(),
                           inplace=True)
    X['BsmtFinSF1'].fillna(X['BsmtFinSF1'].value_counts().idxmax(),
                           inplace=True)
    X['TotalBsmtSF'].fillna(X['TotalBsmtSF'].value_counts().idxmax(),
                            inplace=True)
    X['KitchenQual'].fillna(X['KitchenQual'].value_counts().idxmax(),
                            inplace=True)
    X['GarageCars'].fillna(X['GarageCars'].value_counts().idxmax(),
                           inplace=True)
    X['GarageArea'].fillna(X['GarageArea'].value_counts().idxmax(),
                           inplace=True)
    return X


def ordinal_encode_features(X: pd.DataFrame, is_train=True) -> pd.DataFrame:
    ordinal_features = ['ExterQual',
                        'HeatingQC',
                        'PavedDrive',
                        'Electrical',
                        'Foundation',
                        'SaleCondition']
    if is_train:
        ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                                         unknown_value=15)
        ordinal_encoder.fit(X[ordinal_features])
        save_encoder(ordinal_encoder, "ordinal_encoder")
    else:
        ordinal_encoder = load_encoder('ordinal_encoder')
    X[ordinal_features] = ordinal_encoder.transform(X[ordinal_features])
    return X


def one_hot_encode_features(X: pd.DataFrame, is_train=True) -> pd.DataFrame:
    nominal_features = ['KitchenQual', 'CentralAir']
    feature_names_encoded = []
    if is_train:
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        one_hot_encoder.fit(X[nominal_features])
        save_encoder(one_hot_encoder, "one_hot_encoder")
        feature_names_encoded = one_hot_encoder.get_feature_names_out()
    else:
        one_hot_encoder = load_encoder('one_hot_encoder')
    X[feature_names_encoded] = one_hot_encoder.transform(X[nominal_features])
    X = X.drop(nominal_features, axis=1)
    return X


def scaling_features(X: pd.DataFrame, is_train=True) -> pd.DataFrame:
    features_to_scale = ['OverallQual',
                         'YearBuilt',
                         'YearRemodAdd',
                         'MasVnrArea',
                         'ExterQual',
                         'BsmtFinSF1',
                         'TotalBsmtSF',
                         'HeatingQC',
                         '1stFlrSF',
                         '2ndFlrSF',
                         'GrLivArea',
                         'FullBath',
                         'TotRmsAbvGrd',
                         'Fireplaces',
                         'GarageCars',
                         'GarageArea',
                         'PavedDrive',
                         'WoodDeckSF',
                         'OpenPorchSF',
                         'EnclosedPorch']
    if is_train:
        min_max_scalar = MinMaxScaler(copy=True)
        min_max_scalar.fit(X[features_to_scale])
        save_encoder(min_max_scalar, "min_max_scalar")
    else:
        min_max_scalar = load_encoder('min_max_scalar')
    X[features_to_scale] = min_max_scalar.transform(X[features_to_scale])
    return X


def save_encoder(encoder_object, file_name):
    dirname = os.path.dirname('../models/')
    file_to_save = os.path.join(dirname, file_name)
    joblib.dump(encoder_object, file_to_save)


def load_encoder(file_name):
    dirname = os.path.dirname('../models/')
    file_to_load = os.path.join(dirname, file_name)
    loaded_model = joblib.load(file_to_load)
    return loaded_model
