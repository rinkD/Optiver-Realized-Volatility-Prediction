"""LightGBM + HAR-ridge baseline under rolling-origin validation; RMSPE on the vol scale.

    python src/train.py --dataset dataset_5m.csv.gz
    python src/train.py --dataset dataset_1h.csv.gz
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

FEATURES = [
    "log_rv", "log_rv_lag2", "rv_ma5", "rv_ma22", "rv_std5",
    "mkt_log_rv", "rel_rv", "rv_xrank",
    "z_log_rv", "z_parkinson", "z_volume", "rv_vs_park",
    "rv_first", "rv_last", "rv_mid",
    "skew", "kurt", "ac1", "hl_range", "parkinson",
    "rv_gk", "rv_rs", "vol_avg",
    "log_volume", "volume_cv", "volume_hhi",
    "vwap_dev", "ret_day", "gap",
]

HAR = ["log_rv", "rv_ma5", "rv_ma22"]

# cumulative feature groups, for the ablation
GROUPS = [
    ("HAR only", HAR),
    ("+ range estimators",
     HAR + ["parkinson", "rv_gk", "rv_rs", "vol_avg", "hl_range", "rv_vs_park"]),
    ("+ intraday shape",
     HAR + ["parkinson", "rv_gk", "rv_rs", "vol_avg", "hl_range", "rv_vs_park",
            "rv_first", "rv_last", "rv_mid", "skew", "kurt", "ac1"]),
    ("+ volume and price",
     HAR + ["parkinson", "rv_gk", "rv_rs", "vol_avg", "hl_range", "rv_vs_park",
            "rv_first", "rv_last", "rv_mid", "skew", "kurt", "ac1",
            "log_volume", "volume_cv", "volume_hhi", "vwap_dev", "ret_day", "gap"]),
    ("+ cross-section", FEATURES),
]

PARAMS = dict(
    objective="regression", metric="l2", learning_rate=0.03,
    num_leaves=15, min_data_in_leaf=60, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbosity=-1, seed=7,
)


def rmspe(y_log_true, y_log_pred):
    yt, yp = np.exp(y_log_true), np.exp(y_log_pred)
    return float(np.sqrt(np.mean(((yp - yt) / yt) ** 2)))


def har_ridge_fit(tr, lam=1.0):
    X = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in HAR])
    y = tr["target"].to_numpy()
    A = X.T @ X
    pen = lam * np.eye(A.shape[0])
    pen[0, 0] = 0.0                      # do not penalise the intercept
    return np.linalg.solve(A + pen, X.T @ y)


def har_ridge_pred(beta, df):
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in HAR])
    return X @ beta


def run_ablation(df, blocks):
    """Score each cumulative feature group on identical folds."""
    out = {name: [] for name, _ in GROUPS}
    out["persistence"] = []

    for i in range(1, len(blocks)):
        cut, ev = blocks[i - 1][-1], blocks[i]
        tr = df[df["date"] <= cut]
        va = df[df["date"].isin(ev)]
        if len(tr) < 200 or len(va) < 50:
            continue
        out["persistence"].append(rmspe(va["target"], va["log_rv"]))
        for name, cols in GROUPS:
            dtr = lgb.Dataset(tr[cols], tr["target"])
            dva = lgb.Dataset(va[cols], va["target"], reference=dtr)
            m = lgb.train(PARAMS, dtr, 2000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(80, verbose=False)])
            p = m.predict(va[cols], num_iteration=m.best_iteration)
            out[name].append(rmspe(va["target"], p))

    print(f"\nablation: {len(df):,} rows, {df['date'].nunique()} days, "
          f"{df['ticker'].nunique()} tickers, {len(out['persistence'])} folds")
    base = None
    for name in ["persistence"] + [n for n, _ in GROUPS]:
        vals = np.array(out[name])
        n_feat = 1 if name == "persistence" else dict(GROUPS)[name].__len__()
        line = f"  {name:22s} n_feat={n_feat:2d}  {vals.mean():.4f} ± {vals.std():.4f}"
        if base is not None:
            line += f"   ({vals.mean() - base:+.4f})"
        else:
            base = vals.mean()
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_5m.csv.gz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ablation", action="store_true",
                    help="score each cumulative feature group under the same folds")
    args = ap.parse_args()

    df = pd.read_csv(DATA / args.dataset, parse_dates=["date"])
    blocks = np.array_split(np.sort(df["date"].unique()), args.folds + 1)

    if args.ablation:
        run_ablation(df, blocks)
        return

    rows = []
    for i in range(1, len(blocks)):
        cut, ev = blocks[i - 1][-1], blocks[i]
        tr = df[df["date"] <= cut]
        va = df[df["date"].isin(ev)]
        if len(tr) < 200 or len(va) < 50:
            continue

        dtr = lgb.Dataset(tr[FEATURES], tr["target"])
        dva = lgb.Dataset(va[FEATURES], va["target"], reference=dtr)
        model = lgb.train(PARAMS, dtr, 2000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(80, verbose=False)])
        p_lgb = model.predict(va[FEATURES], num_iteration=model.best_iteration)

        p_har = har_ridge_pred(har_ridge_fit(tr), va)
        p_ens = 0.5 * p_lgb + 0.5 * p_har

        rows.append({
            "fold": i, "train": len(tr), "eval": len(va),
            "persistence": rmspe(va["target"], va["log_rv"]),
            "har_ridge": rmspe(va["target"], p_har),
            "lightgbm": rmspe(va["target"], p_lgb),
            "ensemble": rmspe(va["target"], p_ens),
        })

    res = pd.DataFrame(rows)
    print(f"\n{args.dataset}: {len(df):,} rows, {df['date'].nunique()} days, "
          f"{df['ticker'].nunique()} tickers")
    print(res.round(4).to_string(index=False))
    print()
    for c in ["persistence", "har_ridge", "lightgbm", "ensemble"]:
        print(f"  {c:12s} {res[c].mean():.4f} ± {res[c].std():.4f}")

    full = lgb.train(PARAMS, lgb.Dataset(df[FEATURES], df["target"]), 350)
    gain = pd.Series(full.feature_importance("gain"), index=FEATURES)
    share = (gain / gain.sum() * 100).sort_values(ascending=False)
    print("\ntop features by gain share (%)")
    print(share.head(10).round(1).to_string())


if __name__ == "__main__":
    main()
