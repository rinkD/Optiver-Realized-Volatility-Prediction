# Optiver Realized Volatility Prediction

# Setup
163 US tickers, Yahoo Finance OHLCV only. Two samplings: 5-min bars (50 days, 78 bars/day, 8,146 rows) and 60-min bars (730 days, 7 bars/day, 116,975 rows). RV = sqrt of summed squared log returns, first return from open (overnight gap excluded). Features include three range estimators, intraday shape, volume/VWAP, overnight gap, 5/22-day log-RV averages, and market-comparison features.

# Models
Persistence baseline, HAR ridge, LightGBM on 29 features. Rolling-origin five-fold by date to avoid leakage.

# Results (RMSPE)
 5-min: persistence 0.3405, HAR 0.2568, LightGBM 0.2624. 60-min: 0.8278, 0.5696, 0.5577. Fold spread 0.02–0.05 exceeds model gaps: 3-feature HAR and 29-feature LightGBM are not separable at 5 min; LightGBM only slightly better at 60 min. Ablation: HAR alone 0.2656 at 5 min; adding range, intraday shape, volume/price, market features gives 0.2633, 0.2602, 0.2606, 0.2624. HAR captures 93%/94% of total gain. Market-wide features hurt.

# Surprises
 Range estimators help only with coarse bars (7 bars/day: +0.0126; Garman–Klass top feature at 32% gain). Market-comparison features hurt both tracks. Sampling rate dominates: 78 bars/day ≈0.26; 7 bars/day ≈0.56 despite 20× history.

# Limits
 One regime, noisy RV from 7 returns, no order book, one horizon, large caps only, cumulative grouping.

## Running it

pip install -r requirements.txt
python src/download.py --interval 5m --range 60d --out bars_5m.csv.gz
python src/download.py --interval 60m --range 730d --out bars_1h.csv.gz
python src/build_features.py --bars bars_5m.csv.gz --min-bars 40 --out dataset_5m.csv.gz
python src/build_features.py --bars bars_1h.csv.gz --min-bars 4 --out dataset_1h.csv.gz
python src/train.py --dataset dataset_5m.csv.gz
python src/train.py --dataset dataset_1h.csv.gz --ablation
