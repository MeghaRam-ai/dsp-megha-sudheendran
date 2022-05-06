import pandas as pd
import numpy as np
import os
import joblib
from house_prices.preprocess import drop_unrelevent_features


ordinal_features = ['ExterQual',
                    'HeatingQC',
                    'PavedDrive',
                    'Electrical',
                    'Foundation',
                    'SaleCondition']
nominal_features = ['KitchenQual',
                    'CentralAir']
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


def load_saved_models():
    dirname = os.path.dirname('../models/')
    file_min_max_scalar = os.path.join(dirname, 'min_max_scalar')
    file_one_hot_encoder = os.path.join(dirname, 'one_hot_encoder')
    file_ordinal_encoder = os.path.join(dirname, 'ordinal_encoder')
    file_randomForestRegressor_model = os.path.join(
        dirname, 'random_forest_regressor_model'
    )
    with open(file_min_max_scalar, 'rb') as file_object:
        min_max_scalar = joblib.load(file_object)
    with open(file_one_hot_encoder, 'rb') as file_object1:
        one_hot_encoder = joblib.load(file_object1)
    with open(file_ordinal_encoder, 'rb') as file_object2:
        ordinal_encoder = joblib.load(file_object2)
    with open(file_randomForestRegressor_model, 'rb') as file_object3:
        randomForestRegressor_model = joblib.load(file_object3)
    return (
        min_max_scalar,
        one_hot_encoder,
        ordinal_encoder,
        randomForestRegressor_model
    )


def make_predictions(
    input_data: pd.DataFrame
) -> np.ndarray:
    # the model and all the data preparation objects
    # (encoder, etc) should be loaded from the models folder
    (min_max_scalar,
     one_hot_encoder,
     ordinal_encoder,
     randomForestRegressor_model) = load_saved_models()
    input_data = drop_unrelevent_features(input_data)
    input_data[ordinal_features] = ordinal_encoder.transform(
        input_data[ordinal_features]
    )
    feature_names_encoded = one_hot_encoder.get_feature_names_out()
    input_data[feature_names_encoded] = one_hot_encoder.transform(
        input_data[nominal_features]
    )
    input_data[features_to_scale] = min_max_scalar.transform(
        input_data[features_to_scale]
    )
    input_data = input_data.drop(nominal_features, axis=1)
    prediction = randomForestRegressor_model.predict(input_data)
    return prediction
