# Crypto Volatility Prediction using Machine Learning

This project predicts future cryptocurrency volatility using multi-asset historical OHLC, volume, and market cap data.

It includes feature engineering techniques such as:
- Rolling volatility
- Moving averages
- Liquidity ratios
- Bollinger Bands
- Average True Range (ATR)

An XGBoost model is trained and deployed using Streamlit for real-time predictions.

---

## 📂 Project Structure

crypto-volatility-predictor/
│
├── src/ # Core ML pipeline
├── app/ # Streamlit application
├── data/ # Raw and processed datasets
├── reports/ # Final reports and screenshots
├── notebooks/ # EDA notebooks
├── requirements.txt
└── README.md

---