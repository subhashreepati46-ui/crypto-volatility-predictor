# Final Report — Cryptocurrency Volatility Prediction

## Objective
The goal of this project is to forecast future cryptocurrency volatility using historical OHLC prices, trading volume, and market capitalization. Predicting volatility helps traders and institutions manage risk and identify unstable market periods.

---

## Dataset
The dataset contains daily records for 50+ cryptocurrencies with:

- Date
- Symbol
- Open, High, Low, Close
- Volume
- Market Capitalization

Data was cleaned, sorted by time, and missing values were handled using forward filling.

---

## Feature Engineering
The following features were created:

### Volatility Features
- Rolling 7-day volatility (target)
- Lagged volatility (1, 3, 7 days)
- Rolling volatility mean and standard deviation

### Trend Indicators
- 7-day moving average
- 21-day moving average

### Liquidity Indicator
- Volume / Market Cap ratio

### Technical Indicators
- Bollinger Bands
- Average True Range (ATR)

---

## Model
Model Used: **XGBoost Regressor**

Time-based split was used to avoid data leakage.

---

## Evaluation Metrics

Model Performance:

RMSE: 0.0185  
MAE: 0.0074  
R²: 0.858

Baseline (Persistence Model):

RMSE: 0.0643  
MAE: 0.0374  
R²: -0.714

The model improves prediction accuracy by approximately **71% over baseline**.

---

## Observations
- Volatility clustering exists in cryptocurrency markets
- Lagged volatility significantly improves prediction accuracy
- Stablecoins behave differently due to price peg mechanisms

---

## Deployment
The trained model is deployed using Streamlit, allowing users to input market values and obtain predicted volatility.

---

## Conclusion
The project successfully predicts cryptocurrency volatility and significantly outperforms naive forecasting. The model can assist in risk assessment and market stability analysis.

---

## Future Work
- Hyperparameter tuning
- Deep learning time-series models
- Live market API integration

