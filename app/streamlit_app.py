import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

st.title("Crypto Volatility Predictor")

open_p = st.number_input("Open", value=42000.0)
high = st.number_input("High", value=43000.0)
low = st.number_input("Low", value=41000.0)
close = st.number_input("Close", value=42500.0)
volume = st.number_input("Volume", value=35000000000.0)
mcap = st.number_input("Market Cap", value=800000000000.0)

if st.button("Predict"):

    # ---- create fake history so indicators can exist ----
    rows = 30
    base = pd.DataFrame({
        "open": np.linspace(open_p*0.95, open_p, rows),
        "high": np.linspace(high*0.95, high, rows),
        "low": np.linspace(low*0.95, low, rows),
        "close": np.linspace(close*0.95, close, rows),
        "volume": np.linspace(volume*0.8, volume, rows),
        "market_cap": np.linspace(mcap*0.9, mcap, rows),
    })

    # returns & volatility
    base["returns"] = base["close"].pct_change()
    base["volatility"] = base["returns"].rolling(7).std()

    # moving averages
    base["ma_7"] = base["close"].rolling(7).mean()
    base["ma_21"] = base["close"].rolling(21).mean()

    # liquidity
    base["liq_ratio"] = base["volume"] / (base["market_cap"] + 1)

    # simple ATR approximation
    base["atr"] = (base["high"] - base["low"]).rolling(14).mean()

    # bollinger approximation
    rolling_mean = base["close"].rolling(20).mean()
    rolling_std = base["close"].rolling(20).std()
    base["bb_high"] = rolling_mean + 2 * rolling_std
    base["bb_low"] = rolling_mean - 2 * rolling_std

    # volatility clustering
    base["vol_lag_1"] = base["volatility"].shift(1)
    base["vol_lag_3"] = base["volatility"].shift(3)
    base["vol_lag_7"] = base["volatility"].shift(7)
    base["volatility_rolling_mean"] = base["volatility"].rolling(14).mean()
    base["volatility_rolling_std"] = base["volatility"].rolling(14).std()

    # symbol encoding (generic asset)
    base["symbol_id"] = 0

    # last row = current state
    df = base.tail(1)

    # match training columns
    df = df[scaler.feature_names_in_]

    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)

    st.success(f"Predicted Volatility: {pred[0]:.5f}")
