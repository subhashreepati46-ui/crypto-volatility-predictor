def clean_data(df):

    # remove duplicates
    df = df.drop_duplicates()

    # correct order for time series
    df = df.sort_values(["symbol", "date"])

    # forward fill numeric columns per symbol safely
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        df[col] = df.groupby("symbol")[col].transform("ffill")

    return df

