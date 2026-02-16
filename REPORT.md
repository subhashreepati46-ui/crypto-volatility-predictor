# Cryptocurrency Volatility Prediction

## 1. Problem Statement
The objective of this project is to predict **future cryptocurrency volatility** using historical OHLC (Open, High, Low, Close), trading volume, and market capitalization data.

Unlike price prediction, volatility forecasting is a risk-modeling task. Financial markets exhibit *volatility clustering* where high volatility periods tend to follow high volatility periods. This project attempts to model that behavior using machine learning.

---

## 2. Dataset
The dataset contains daily historical records of multiple cryptocurrencies including:

- Open
- High
- Low
- Close
- Volume
- Market Cap
- Symbol
- Date

After cleaning:
- Missing values forward-filled per asset
- Time ordering preserved
- Assets with insufficient history removed

Total rows after cleaning: ~72,000+

---

## 3. Feature Engineering

### Target Variable
Future 7-day volatility:
volatility = rolling_std(returns, 7).shift(-7)


### Technical Indicators
- Moving averages (7, 21)
- Bollinger Bands
- ATR (Average True Range)
- Liquidity ratio

### Temporal Features
- Lagged volatility (1, 3, 7 days)
- Rolling volatility mean/std

### Multi-Asset Handling
Each cryptocurrency encoded using categorical identifier to prevent regime mixing.

---

## 4. Model
Model used: **XGBoost Regressor**

Reason:
- Handles nonlinear relationships
- Robust to noisy financial data
- Works well with tabular time-series features

Time-aware split used instead of random split to avoid data leakage.

---

## 5. Evaluation Metrics
- RMSE
- MAE
- R²
- Persistence baseline comparison

---

## 6. Results

Overall Performance:

| Metric | Value |
|------|------|
RMSE | ~0.018
MAE | ~0.007
R² | ~0.85

Baseline Improvement: ~71%

---

## 7. Key Findings

1. Volatility is predictable using past volatility (clustering effect)
2. Price indicators alone are insufficient
3. Multi-asset modeling requires asset identity encoding
4. Stablecoins behave differently due to peg mechanism

---

## 8. Conclusion
The model successfully forecasts future cryptocurrency volatility and significantly outperforms naive persistence methods. The results demonstrate the presence of volatility clustering in crypto markets and the importance of time-aware validation.
