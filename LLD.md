# Low Level Design

## data/load_data.py
Reads dataset, cleans column names, parses timestamps, ensures schema consistency.

## data/preprocess.py
Handles missing values and orders data by symbol and date.

## data/feature_engineering.py
Creates technical indicators and lagged volatility features per asset.

## models/train.py
Splits data using time-series split and trains XGBoost model.

## models/evaluate.py
Calculates metrics and compares with persistence baseline.

## pipeline/run_pipeline.py
Runs full workflow from data loading to evaluation.

## app/streamlit_app.py
Interactive UI allowing user to input OHLC values and predict volatility.

