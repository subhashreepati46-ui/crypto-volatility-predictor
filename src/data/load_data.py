import pandas as pd
from src.config import DATA_PATH

def normalize(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "")
    )

def load_dataset():
    df = pd.read_csv(DATA_PATH)

    # HARD normalize all column names
    df.columns = [normalize(c) for c in df.columns]

    # remove index column if present
    if "unnamed:_0" in df.columns or "unnamed_0" in df.columns:
        df = df.drop(columns=[c for c in df.columns if "unnamed" in c])

    # auto-detect symbol column
    for possible in ["crypto_name","symbol","coin","name"]:
        if possible in df.columns:
            df = df.rename(columns={possible:"symbol"})
            break

    # auto-detect market cap
    for possible in ["marketcap","market_cap","market_capitalization"]:
        if possible in df.columns:
            df = df.rename(columns={possible:"market_cap"})
            break

    # -------- DATE HANDLING --------
    if "timestamp" in df.columns:
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["date"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.drop(columns=["timestamp"])

    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("No date column found")

    # ensure required columns exist
    required_cols = ["date","symbol","open","high","low","close","volume","market_cap"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # fill symbol blocks
    df["symbol"] = df["symbol"].ffill()

    # drop invalid rows
    df = df.dropna(subset=["date"])

    df = df.sort_values(["symbol","date"]).reset_index(drop=True)

    print("Loaded columns:", df.columns.tolist())
    print("Rows after cleaning:", len(df))

    return df

