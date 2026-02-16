from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd


def evaluate(model, X_test, y_test, symbols=None):

    # predictions
    preds = model.predict(X_test)

    y_test_series = pd.Series(y_test).reset_index(drop=True)
    preds_series = pd.Series(preds).reset_index(drop=True)

    # ---------- model metrics ----------
    rmse = np.sqrt(mean_squared_error(y_test_series, preds_series))
    mae = mean_absolute_error(y_test_series, preds_series)
    r2 = r2_score(y_test_series, preds_series)

    # ---------- persistence baseline ----------
    baseline = y_test_series.shift(1)
    valid_idx = baseline.notna()

    baseline_rmse = np.sqrt(mean_squared_error(y_test_series[valid_idx], baseline[valid_idx]))
    baseline_mae = mean_absolute_error(y_test_series[valid_idx], baseline[valid_idx])
    baseline_r2 = r2_score(y_test_series[valid_idx], baseline[valid_idx])

    improvement = (baseline_rmse - rmse) / baseline_rmse * 100

    # ---------- print summary ----------
    print("\nMODEL PERFORMANCE")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R2:   {r2:.4f}")

    print("\nBASELINE (Persistence)")
    print(f"RMSE: {baseline_rmse:.6f}")
    print(f"MAE:  {baseline_mae:.6f}")
    print(f"R2:   {baseline_r2:.4f}")

    print(f"\nImprovement over baseline: {improvement:.2f}%")

    # ---------- per-asset evaluation ----------
    if symbols is not None:
        print("\nPER-ASSET R2 (min 50 samples):")
        df_eval = pd.DataFrame({
            "y": y_test_series,
            "pred": preds_series,
            "symbol": pd.Series(symbols).reset_index(drop=True)
        })

        for sym, g in df_eval.groupby("symbol"):
            if len(g) >= 50:
                score = r2_score(g["y"], g["pred"])
                print(f"{sym:12s} R2: {score:.3f}")
