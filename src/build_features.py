"""Turn intraday bars into one row per (ticker, day): features + next-day RV target.

    python src/build_features.py --bars bars_5m.csv.gz --min-bars 40 --out dataset_5m.csv.gz
    python src/build_features.py --bars bars_1h.csv.gz  --min-bars 4  --out dataset_1h.csv.gz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def day_features(g: pd.DataFrame) -> dict:
    g = g.sort_values("ts")
    o, c = g["open"].to_numpy(), g["close"].to_numpy()
    h, l = g["high"].to_numpy(), g["low"].to_numpy()
    v = g["volume"].to_numpy(dtype=float)

    r = np.diff(np.log(c), prepend=np.log(o[0]))
    n = len(r)
    rv = np.sqrt(np.sum(r ** 2))

    edge = max(1, n // 6)
    rv_first = np.sqrt(np.sum(r[:edge] ** 2))
    rv_last = np.sqrt(np.sum(r[-edge:] ** 2))
    rv_mid = np.sqrt(np.sum(r[edge:-edge] ** 2)) if n > 2 * edge else np.nan

    sd = r.std() + 1e-12
    skew = float(np.mean(((r - r.mean()) / sd) ** 3))
    kurt = float(np.mean(((r - r.mean()) / sd) ** 4))
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if r.std() > 0 else np.nan
    except Exception:
        ac1 = np.nan

    vwap = float(np.sum(c * v) / (np.sum(v) + 1e-9))
    vshare = v / (v.sum() + 1e-9)

    # range-based estimators, per bar then aggregated
    lo_hi = np.log(h / l)
    co = np.log(c / o)
    gk_var = 0.5 * lo_hi ** 2 - (2 * np.log(2) - 1) * co ** 2
    rs_var = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    rv_gk = float(np.sqrt(np.clip(gk_var.sum(), 0, None)))
    rv_rs = float(np.sqrt(np.clip(rs_var.sum(), 0, None)))
    park = float(np.sqrt(np.mean(lo_hi ** 2) / (4 * np.log(2))))
    vol_avg = float(np.exp(np.mean(np.log(np.array([park, rv_gk, rv_rs]) + 1e-12))))

    return {
        "log_rv": np.log(rv + 1e-12),
        "rv_first": rv_first, "rv_last": rv_last, "rv_mid": rv_mid,
        "skew": skew, "kurt": kurt, "ac1": ac1,
        "hl_range": float(np.log(h.max() / l.min())),
        "parkinson": park,
        "rv_gk": rv_gk, "rv_rs": rv_rs, "vol_avg": vol_avg,
        "log_volume": float(np.log(v.sum() + 1.0)),
        "volume_cv": float(v.std() / (v.mean() + 1e-9)),
        "volume_hhi": float(np.sum(vshare ** 2)),
        "vwap_dev": float(np.log(c[-1] / vwap)),
        "ret_day": float(np.log(c[-1] / o[0])),
        "n_bars": n,
        "day_close": float(c[-1]),
        "day_open": float(o[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", default="bars_5m.csv.gz")
    ap.add_argument("--min-bars", type=int, default=40)
    ap.add_argument("--out", default="dataset_5m.csv.gz")
    args = ap.parse_args()

    bars = pd.read_csv(DATA / args.bars, parse_dates=["ts"])
    bars["date"] = bars["ts"].dt.tz_convert("America/New_York").dt.date

    rows = []
    for (ticker, date), g in bars.groupby(["ticker", "date"], sort=True):
        if len(g) < args.min_bars:
            continue
        f = day_features(g)
        f["ticker"], f["date"] = ticker, date
        rows.append(f)
    d = pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)

    d["prev_close"] = d.groupby("ticker")["day_close"].shift(1)
    d["gap"] = np.log(d["day_open"] / d["prev_close"])
    d["target"] = d.groupby("ticker")["log_rv"].shift(-1)

    by = d.groupby("ticker")["log_rv"]
    d["log_rv_lag2"] = by.shift(2)
    d["rv_ma5"] = by.transform(lambda s: s.rolling(5, min_periods=3).mean())
    d["rv_ma22"] = by.transform(lambda s: s.rolling(22, min_periods=10).mean())
    d["rv_std5"] = by.transform(lambda s: s.rolling(5, min_periods=3).std())

    # market-wide context, known at the close of day t
    d["mkt_log_rv"] = d.groupby("date")["log_rv"].transform("mean")
    d["rel_rv"] = d["log_rv"] - d["mkt_log_rv"]
    d["rv_xrank"] = d.groupby("date")["log_rv"].rank(pct=True)

    # cross-sectional standardisation, so a row is comparable across dates
    for col, name in [("log_rv", "z_log_rv"), ("parkinson", "z_parkinson"),
                      ("log_volume", "z_volume")]:
        g = d.groupby("date")[col]
        d[name] = ((d[col] - g.transform("mean"))
                   / (g.transform("std") + 1e-9))

    # the two volatility estimators disagreeing is itself informative
    d["rv_vs_park"] = d["log_rv"] - np.log(d["parkinson"] + 1e-12)

    keep = [
        "ticker", "date", "target",
        "log_rv", "log_rv_lag2", "rv_ma5", "rv_ma22", "rv_std5",
        "mkt_log_rv", "rel_rv", "rv_xrank",
        "z_log_rv", "z_parkinson", "z_volume", "rv_vs_park",
        "rv_first", "rv_last", "rv_mid",
        "skew", "kurt", "ac1", "hl_range", "parkinson",
        "rv_gk", "rv_rs", "vol_avg",
        "log_volume", "volume_cv", "volume_hhi",
        "vwap_dev", "ret_day", "gap", "n_bars",
    ]
    d = d[keep].replace([np.inf, -np.inf], np.nan).dropna()

    d.to_csv(DATA / args.out, index=False, compression="gzip")
    print(f"{args.out}: {len(d):,} rows, {d['ticker'].nunique()} tickers, "
          f"{d['date'].nunique()} days")


if __name__ == "__main__":
    main()
