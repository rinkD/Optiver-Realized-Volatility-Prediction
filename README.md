# Realized volatility from 5-minute bars

I wanted to know two things: how much of tomorrow's volatility is already visible in
today's intraday path, and whether it's worth throwing a pile of features at the problem.

Short answer to the second one: no. A three-feature linear model does as well as a
29-feature gradient boosting model. That surprised me, and it's most of what's below.

## The data

163 US tickers from the Yahoo Finance chart endpoint, two frequencies:

- `interval=5m&range=60d` — 60 calendar days, which is 50 usable trading days, 78 bars each
- `interval=60m&range=730d` — that returned 730 *trading* days, roughly three years, 7 bars each

Collapsed into one row per (ticker, day) that's 8,146 rows and 116,975 rows. Two of the
165 tickers I wanted dropped out — one renamed, one delisted.

There's no order book here, so no spread, no book imbalance, no trade-level data. That's
the one thing the Optiver competition actually provided, and it's why nothing below is
comparable to that leaderboard. Everything here is OHLCV.

## What I computed

Per day, from the bars: realized volatility as `RV = sqrt(sum r^2)` on log returns, with
the first return measured from the open rather than from the previous close (otherwise the
overnight gap leaks into the intraday number). Then Parkinson, Garman–Klass and
Rogers–Satchell range estimators and their geometric mean. Then where in the day the
volatility sat — first sixth, last sixth, middle — plus realized skew and kurtosis and the
serial correlation of the 5-minute returns.

On top of that: volume level, volume concentration, VWAP deviation, overnight gap, and the
HAR history (5-day and 22-day averages of `log RV`). Plus two market-wide things known at
the close — the cross-sectional mean of `log RV` and the gap to it — and per-date z-scores
of three features.

Target is next-day `log RV`. I train on the log and score on the level, which are nearly
the same thing: for small errors, `log(pred/true)` and `(pred-true)/true` agree to first
order. On a test batch of five predictions RMSPE came out 0.0738 and log-RMSE 0.0723, so
squared loss on the log is already minimising RMSPE.

Validation is rolling-origin over dates, five folds, each scored once on a window that
starts after its training window ends. Random splits leak badly here — neighbouring days
of the same ticker are near-duplicates.

## Results

RMSPE, mean ± sd across folds.

| | 5-minute (50 days) | 60-minute (718 days) |
|---|---|---|
| persistence (tomorrow = today) | 0.3405 ± 0.0190 | 0.8278 ± 0.0490 |
| HAR ridge, 3 features | **0.2568 ± 0.0233** | 0.5696 ± 0.0203 |
| LightGBM, 29 features | 0.2624 ± 0.0248 | **0.5577 ± 0.0189** |

The spread across folds is 0.02–0.05, wider than the gap between the two models. Worth
keeping in mind before claiming one beats the other.

To see where the signal actually is, I ran a cumulative ablation: start with the three HAR
features, add a group, rescore on the same folds.

| features added | 5-minute | 60-minute |
|---|---|---|
| HAR only | 0.2656 | 0.5698 |
| + range estimators | 0.2633 | 0.5572 |
| + intraday shape | 0.2602 | 0.5551 |
| + volume and price | 0.2606 | 0.5540 |
| + cross-section | 0.2624 | 0.5577 |

Against persistence the total attainable gain is 0.075 at 5 minutes and 0.274 at 60
minutes. HAR takes 0.075 and 0.258 of it — 93% and 94%. The other 25 features split what's
left.

## Four things I didn't expect

**Range estimators matter much more when bars are coarse.** On 5-minute bars they buy
almost nothing. On 60-minute bars they buy 0.0126, more than half of everything left after
HAR. Garman–Klass alone is the top feature at 7 bars a day, at 32% of total gain. Four
prices per bar apparently carry more volatility information than seven squared returns do.

**Cross-sectional features make both tracks worse.** Ranking a stock against the market on
the same day, and standardising within date, are the only additions that hurt. I don't have
a good explanation and I'm not going to invent one.

**Yesterday's volatility is nearly useless on its own.** `log_rv` contributes under 2% of
gain in both tracks while `rv_ma22` contributes 28–46%. What predicts tomorrow is the level
over weeks, not the last print.

**Sampling frequency dominates everything else.** 78 bars a day gets 0.26; 7 bars a day
gets 0.56 with twenty times the history. Extra days don't compensate for a noisy target.

## Running it

```bash
pip install -r requirements.txt

python src/download.py --interval 5m  --range 60d  --out bars_5m.csv.gz
python src/download.py --interval 60m --range 730d --out bars_1h.csv.gz

python src/build_features.py --bars bars_5m.csv.gz --min-bars 40 --out dataset_5m.csv.gz
python src/build_features.py --bars bars_1h.csv.gz  --min-bars 4  --out dataset_1h.csv.gz

python src/train.py --dataset dataset_5m.csv.gz
python src/train.py --dataset dataset_1h.csv.gz --ablation
```

Downloads take about eight minutes, feature building twenty seconds each, training about
three. Raw bars aren't committed — the endpoint refetches them.

## What's wrong with this

- 50 trading days is one regime. The 0.26 is optimistic for a different year, and I can't
  test that with this endpoint.
- The 60-minute track has plenty of days but a bad target: RV from 7 returns carries large
  sampling error, which is exactly why its numbers look so much worse.
- No book, no trades, no pre-market session. The features that carried most of the
  competition's signal are simply absent.
- One horizon only. Predicting further out, or just the overnight session, is untested.
- 163 large caps, so nothing is said about small caps.
- The ablation is cumulative, so it measures "this group given the previous ones" rather
  than each group's standalone value. Cross-section alone might be fine.
- The first fold of the 5-minute track trains on 1,463 rows and scores 0.2294, better than
  later folds. That isn't the model doing better, those nine days were just quiet.

## Reference

Optiver Realized Volatility Prediction, Kaggle, 2021. Same question, better data.
