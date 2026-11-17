"""Download intraday bars from the Yahoo Finance chart endpoint.

The endpoint caps intraday history: 5-minute bars go back 60 days, 60-minute
bars go back 730 days.

    python src/download.py --interval 5m --range 60d  --out bars_5m.csv.gz
    python src/download.py --interval 60m --range 730d --out bars_1h.csv.gz
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

UNIVERSE = """AAPL MSFT NVDA AMZN META GOOGL GOOG TSLA AVGO AMD INTC MU QCOM TXN ADI AMAT LRCX
JPM BAC WFC C GS MS SCHW BLK AXP V MA PYPL
XOM CVX COP SLB OXY EOG PSX MPC VLO
JNJ PFE MRK ABBV LLY UNH CVS TMO ABT BMY GILD AMGN
WMT COST TGT HD LOW MCD SBUX NKE PG KO PEP MDLZ CL
DIS CMCSA NFLX T VZ TMUS ORCL CRM ADBE NOW INTU IBM ACN CSCO
BA CAT DE GE HON LMT RTX NOC GD EMR ETN ITW PH
SPY QQQ IWM DIA
BKNG MAR ABNB UBER LYFT DASH SHOP SQ COIN PLTR SNOW
PANW CRWD ZS DDOG NET MDB TEAM WDAY VEEV HUBS
MRNA REGN VRTX BIIB ZTS SYK MDT ISRG DHR
RTX NEE DUK SO AEP EXC XEL
LIN DOW DD NEM FCX NUE STLD CLF
TFC USB PNC COF MET PRU AFL ALL TRV
LUV DAL UAL AAL CCL RCL MGM WYNN LVS
YUM CMG DPZ KHC GIS K HSY SYY KR
""".split()

URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
       "?range={rng}&interval={itv}&includePrePost=false")


def fetch(sym, interval, rng):
    req = urllib.request.Request(
        URL.format(sym=sym, itv=interval, rng=rng),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=25) as r:
        js = json.loads(r.read().decode())
    res = js["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "ticker": sym,
        "ts": pd.to_datetime(res["timestamp"], unit="s", utc=True),
        "open": q["open"], "high": q["high"], "low": q["low"],
        "close": q["close"], "volume": q["volume"],
    })
    return df.dropna(subset=["close"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--range", dest="rng", default="60d")
    ap.add_argument("--out", default="bars_5m.csv.gz")
    args = ap.parse_args()

    frames, failed = [], []
    for i, sym in enumerate(UNIVERSE, 1):
        try:
            frames.append(fetch(sym, args.interval, args.rng))
        except Exception:
            failed.append(sym)
        if i % 10 == 0 or i == len(UNIVERSE):
            print(f"  {i:2d}/{len(UNIVERSE)}")
        time.sleep(0.4)

    bars = pd.concat(frames, ignore_index=True)
    bars.to_csv(DATA / args.out, index=False, compression="gzip")
    n_days = (bars["ts"].dt.tz_convert("America/New_York").dt.date.nunique())
    print(f"{args.interval}/{args.rng}: {bars['ticker'].nunique()} tickers, "
          f"{len(bars):,} bars, {n_days} days -> {args.out}")
    if failed:
        print("failed:", " ".join(failed))


if __name__ == "__main__":
    main()
