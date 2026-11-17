# Optiver Realized Volatility Prediction

A practical implementation of realized volatility prediction using publicly available high-frequency OHLCV data, motivated by the [Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction) competition.

## Motivation

I wanted to understand how realized volatility can be constructed from high-frequency returns and whether simple features have predictive power. This project uses publicly available Yahoo Finance data instead of the competition’s order book data, focusing on feature engineering, leakage-aware validation, and model comparison rather than leaderboard ranking.

The work is related to topics in high-frequency financial econometrics and volatility modeling.

## What I Did

- Constructed realized volatility from 5-minute and 60-minute bars (excluding overnight gaps).
- Built features including range estimators, intraday shape, volume/VWAP, overnight gap, and lagged log-RV.
- Compared a persistence baseline, HAR (ridge), and LightGBM under rolling-origin cross-validation.
- Ran ablation studies to see which feature groups actually help.

## Key Results (RMSPE)

| Data       | Persistence | HAR   | LightGBM |
|------------|-------------|-------|----------|
| 5-min bars | 0.3405      | 0.2568| 0.2624   |
| 60-min bars| 0.8278      | 0.5696| 0.5577   |

HAR captures most of the improvement. Additional features help only marginally, and market-wide features slightly hurt performance. Sampling frequency has a larger impact than model choice.

## Repository Structure
