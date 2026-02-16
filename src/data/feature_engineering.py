import pandas as pd
import ta


# ---------- per symbol feature creation ----------
def compute_symbol_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()

    # returns
    group["returns"] = group["close"].pct_change()

    # predict next-week volatility
    group["volatility"] = group["returns"].rolling(7).std().shift(-7)

    # moving averages
    group["ma_7"] = group["close"].rolling(7).mean()
    group["ma_21"] = group["close"].rolling(21).mean()

    # liquidity
    group["liq_ratio"] = group["volume"] / (group["market_cap"] + 1)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=group["close"], window=20)
    group["bb_high"] = bb.bollinger_hband()
    group["bb_low"] = bb.bollinger_lband()

    # ATR
    atr = ta.volatility.AverageTrueRange(
        high=group["high"],
        low=group["low"],
        close=group["close"],
        window=14
    )
    group["atr"] = atr.average_true_range()

    # volatility clustering features
    group["vol_lag_1"] = group["volatility"].shift(1)
    group["vol_lag_3"] = group["volatility"].shift(3)
    group["vol_lag_7"] = group["volatility"].shift(7)

    group["volatility_rolling_mean"] = group["volatility"].rolling(14).mean()
    group["volatility_rolling_std"]  = group["volatility"].rolling(14).std()

    return group


# ---------- main feature pipeline ----------
def add_features(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or len(df) == 0:
        raise ValueError("Dataset empty before feature engineering")

    # keep only symbols with enough history
    min_rows_required = 40
    valid_symbols = df["symbol"].value_counts()
    valid_symbols = valid_symbols[valid_symbols >= min_rows_required].index
    df = df[df["symbol"].isin(valid_symbols)]

    if len(df) == 0:
        raise ValueError("All symbols removed due to insufficient history")

    # group processing WITHOUT keeping group columns inside apply
    groups = []
    for sym, g in df.sort_values(["symbol", "date"]).groupby("symbol"):
        g = compute_symbol_features(g)
        g["symbol"] = sym
        groups.append(g)

    df = pd.concat(groups, ignore_index=True)

    # drop incomplete rows
    df = df.dropna(subset=[
        "volatility",
        "ma_7", "ma_21", "atr",
        "vol_lag_1", "vol_lag_3", "vol_lag_7",
        "volatility_rolling_mean", "volatility_rolling_std"
    ])

    if len(df) == 0:
        raise ValueError("All rows removed after feature creation")

    return df
