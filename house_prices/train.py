import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from house_prices.preprocess import drop_unrelevent_features
from house_prices.preprocess import ordinal_encode_features
from house_prices.preprocess import one_hot_encode_features
from house_prices.preprocess import scaling_features
from house_prices.preprocess import save_encoder
from house_prices.preprocess import load_encoder


def split_dataset(df: pd.DataFrame) -> np.ndarray:
    y = df['SalePrice']
    X = df.drop(['SalePrice'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def model_training(X_train: np.ndarray,
                   y_train: np.ndarray) -> RandomForestRegressor:
    random_forest_regressor_model = RandomForestRegressor(max_depth=2,
                                                          random_state=0)
    random_forest_regressor_model.fit(X_train, y_train)
    save_encoder(random_forest_regressor_model,
                 'random_forest_regressor_model')
    return random_forest_regressor_model


def model_evaluation(model: RandomForestRegressor,
                     X_test: np.ndarray,
                     y_test: np.ndarray) -> np.float64:
    random_forest_regressor_model = load_encoder(
        "random_forest_regressor_model"
    )
    prediction = random_forest_regressor_model.predict(X_test)
    rmsle = compute_rmsle(y_test, prediction)
    return rmsle


def compute_rmsle(y_test: np.ndarray,
                  y_pred: np.ndarray,
                  precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def build_model(df: pd.DataFrame) -> dict[str, str]:
    # Returns a dictionary with the model performances
    # (for example {'rmse': 0.18})
    X_train, X_test, y_train, y_test = split_dataset(df)
    X_train = drop_unrelevent_features(X_train)
    X_test = drop_unrelevent_features(X_test)
    X_train = ordinal_encode_features(X_train)
    X_test = ordinal_encode_features(X_test)
    X_train = one_hot_encode_features(X_train)
    X_test = one_hot_encode_features(X_test)
    X_train = scaling_features(X_train)
    X_test = scaling_features(X_test)
    ml_model = model_training(X_train, y_train)
    rmsle = model_evaluation(ml_model, X_test, y_test)
    return {'rmsle': rmsle}
