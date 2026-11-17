


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

| Data        | Persistence | HAR    | LightGBM |
|-------------|-------------|--------|----------|
| 5-min bars  | 0.3405      | 0.2568 | 0.2624   |
| 60-min bars | 0.8278      | 0.5696 | 0.5577   |

HAR captures most of the improvement. Additional features help only marginally, and market-wide features slightly hurt performance. Sampling frequency has a larger impact than model choice.

## Repository Structure

```
src/
├── download.py          # Download OHLCV data from Yahoo Finance
├── build_features.py    # Compute RV and features
└── train.py             # Train and evaluate models
```

## How to Run

Python 3.10 or newer. About 15 minutes end to end, most of it in the two downloads.  
All outputs land in `data/`, which the scripts create automatically.

```bash
pip install -r requirements.txt

python src/download.py --interval 5m --range 60d --out bars_5m.csv.gz
python src/download.py --interval 60m --range 730d --out bars_1h.csv.gz

python src/build_features.py --bars bars_5m.csv.gz --min-bars 40 --out dataset_5m.csv.gz
python src/build_features.py --bars bars_1h.csv.gz --min-bars 4 --out dataset_1h.csv.gz

python src/train.py --dataset dataset_5m.csv.gz
python src/train.py --dataset dataset_1h.csv.gz
```

To see the ablation table, add `--ablation`:

```bash
python src/train.py --dataset dataset_1h.csv.gz --ablation
```

**Note**: Re-running the 60-minute download later will shift the sample, because the endpoint returns the most recent 730 trading days. Row counts can change slightly over time.

## Notes

This is a learning project. The main goal was to implement the realized volatility pipeline cleanly and understand which simple features carry signal.

```
