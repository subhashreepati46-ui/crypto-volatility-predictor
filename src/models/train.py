import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.config import *


def train_model(df):

    # sanity check
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataframe")

    # sort by time BEFORE split (critical for time series)
    df = df.sort_values("date").reset_index(drop=True)
    # encode crypto identity (critical for multi-asset modeling)
    df["symbol_id"] = df["symbol"].astype("category").cat.codes


    # features and target
    X = df.drop(columns=[TARGET, "date", "symbol"])
    y = df[TARGET]

    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------- TIME SERIES SPLIT ----------
    split_index = int(len(X_scaled) * (1 - TEST_SIZE))

    X_train = X_scaled[:split_index]
    X_test  = X_scaled[split_index:]
    y_train = y[:split_index]
    y_test  = y[split_index:]

    # model
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # train
    model.fit(X_train, y_train)

    # save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return X_test, y_test, df.loc[y_test.index, "symbol"], model


