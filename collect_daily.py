"""
Daily OHLCV collector for Bitcoin, Gold and Ethereum.

Outputs (data/):
  btc_daily.csv   — Bitcoin  (USD), from 2012-01-01
  xau_daily.csv   — Gold     (USD), from 2012-01-01
  eth_daily.csv   — Ethereum (USD), from 2017-11-09

Format (identical for all three):
  Date,Open,High,Low,Close,Volume
  YYYY-MM-DD, comma-separated, floats rounded to 6 decimals.

Auto-update: re-running the script only downloads data newer than the
last row already present in each CSV file.

Sources:
  BTC — local btcusd_1-min_data.csv resampled to 1d,
         then yfinance BTC-USD for dates not covered locally.
  Gold — local XAU_1h_data.csv resampled to 1d,
          then yfinance GC=F for dates not covered locally.
  ETH — yfinance ETH-USD (no local source available).
"""

import os
import pandas as pd
import yfinance as yf
from datetime import date, datetime, timedelta

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
DATE_COL   = "Date"
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
ALL_COLS   = [DATE_COL] + OHLCV_COLS


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _load_daily(path: str) -> pd.DataFrame:
    """Load an existing daily CSV; returns empty DataFrame if absent."""
    if not os.path.isfile(path):
        return pd.DataFrame(columns=ALL_COLS)
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.date
    return df.sort_values(DATE_COL).reset_index(drop=True)


def _last_date(df: pd.DataFrame):
    """Return the last date present in a daily DataFrame, or None."""
    if df.empty:
        return None
    return df[DATE_COL].max()


def _save(df: pd.DataFrame, path: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    df = df.sort_values(DATE_COL).drop_duplicates(DATE_COL).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.strftime("%Y-%m-%d")
    df[OHLCV_COLS] = df[OHLCV_COLS].round(6)
    df[ALL_COLS].to_csv(path, index=False)


def _flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the MultiIndex columns that yfinance ≥ 1.0 returns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def _download_yf(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from yfinance; returns normalised DataFrame."""
    raw = yf.download(ticker, start=start, end=end, interval="1d",
                      auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    df = _flatten_yf(raw).reset_index()

    # Normalise column names (yfinance capitalises them)
    df.columns = [c.capitalize() if c.lower() != "date" else DATE_COL
                  for c in df.columns]

    # Rename index column (yfinance may call it "Date" or "Datetime")
    for alias in ["Date", "Datetime"]:
        if alias in df.columns:
            df = df.rename(columns={alias: DATE_COL})
            break

    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.date
    return df[[DATE_COL] + OHLCV_COLS].dropna()


# ---------------------------------------------------------------------------
# Bitcoin
# ---------------------------------------------------------------------------

def _btc_from_local(after: date | None) -> pd.DataFrame:
    """
    Resample btcusd_1-min_data.csv to daily OHLCV.
    Only processes rows strictly after `after` (or everything if None).
    """
    src = os.path.join(DATA_DIR, "btcusd_1-min_data.csv")
    if not os.path.isfile(src):
        print("  [BTC] local 1-min file not found, skipping.")
        return pd.DataFrame()

    print("  [BTC] reading local 1-min CSV …")
    df = pd.read_csv(src, usecols=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
    df[DATE_COL] = df["Timestamp"].dt.date
    df = df.drop(columns="Timestamp")

    if after is not None:
        df = df[df[DATE_COL] > after]
    if df.empty:
        return pd.DataFrame()

    daily = df.groupby(DATE_COL).agg(
        Open=("Open",   "first"),
        High=("High",   "max"),
        Low=("Low",     "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    ).reset_index()
    print(f"  [BTC] resampled to {len(daily)} daily rows from local file.")
    return daily


def update_btc(verbose: bool = True) -> None:
    path = os.path.join(DATA_DIR, "btc_daily.csv")
    existing = _load_daily(path)
    last = _last_date(existing)

    if verbose:
        print(f"\n{'='*50}")
        print(f"BTC  — last date: {last or 'none (first run)'}")

    # 1. Resample local 1-min data for dates not yet in the CSV
    local_daily = _btc_from_local(after=last)

    # 2. Determine the furthest date now covered by local data
    if not local_daily.empty:
        local_max = local_daily[DATE_COL].max()
    elif last is not None:
        local_max = last
    else:
        local_max = None

    # 3. Fill the gap between local_max and today with yfinance
    today = date.today()
    yf_start = (local_max + timedelta(days=1)).strftime("%Y-%m-%d") if local_max else "2012-01-01"
    yf_end   = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    yf_daily = pd.DataFrame()
    if local_max is None or local_max < today:
        if verbose:
            print(f"  [BTC] downloading yfinance BTC-USD from {yf_start} …")
        yf_daily = _download_yf("BTC-USD", yf_start, yf_end)
        if verbose:
            print(f"  [BTC] yfinance returned {len(yf_daily)} rows.")

    frames = [existing, local_daily, yf_daily]
    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    _save(combined, path)

    result = _load_daily(path)
    if verbose:
        print(f"  [BTC] saved {len(result)} rows → {path}")
        print(f"         from {result[DATE_COL].min()} to {result[DATE_COL].max()}")


# ---------------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------------

def _xau_from_local(after: date | None) -> pd.DataFrame:
    """
    Resample XAU_1h_data.csv to daily OHLCV.
    Only processes rows strictly after `after` (or everything if None).
    """
    src = os.path.join(DATA_DIR, "XAU_1h_data.csv")
    if not os.path.isfile(src):
        print("  [XAU] local 1h file not found, skipping.")
        return pd.DataFrame()

    print("  [XAU] reading local 1h CSV …")
    df = pd.read_csv(src, sep=";")
    # Raw "Date" column is "YYYY.MM.DD HH:MM" → convert to date object in-place
    df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M").dt.date
    df = df.rename(columns={"Date": DATE_COL})

    if after is not None:
        df = df[df[DATE_COL] > after]
    if df.empty:
        return pd.DataFrame()

    daily = df.groupby(DATE_COL).agg(
        Open=("Open",   "first"),
        High=("High",   "max"),
        Low=("Low",     "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    ).reset_index()
    print(f"  [XAU] resampled to {len(daily)} daily rows from local file.")
    return daily


def update_xau(verbose: bool = True) -> None:
    path = os.path.join(DATA_DIR, "xau_daily.csv")
    existing = _load_daily(path)
    last = _last_date(existing)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Gold — last date: {last or 'none (first run)'}")

    # Restrict local resample to 2012-01-01 onwards for consistency
    local_floor = date(2012, 1, 1)
    local_after = max(last, local_floor - timedelta(days=1)) if last else local_floor - timedelta(days=1)
    local_daily = _xau_from_local(after=local_after)

    if not local_daily.empty:
        local_max = local_daily[DATE_COL].max()
    elif last is not None:
        local_max = last
    else:
        local_max = None

    today = date.today()
    yf_start = (local_max + timedelta(days=1)).strftime("%Y-%m-%d") if local_max else "2012-01-01"
    yf_end   = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    yf_daily = pd.DataFrame()
    if local_max is None or local_max < today:
        if verbose:
            print(f"  [XAU] downloading yfinance GC=F from {yf_start} …")
        yf_daily = _download_yf("GC=F", yf_start, yf_end)
        if verbose:
            print(f"  [XAU] yfinance returned {len(yf_daily)} rows.")

    frames = [existing, local_daily, yf_daily]
    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    _save(combined, path)

    result = _load_daily(path)
    if verbose:
        print(f"  [XAU] saved {len(result)} rows → {path}")
        print(f"         from {result[DATE_COL].min()} to {result[DATE_COL].max()}")


# ---------------------------------------------------------------------------
# Ethereum
# ---------------------------------------------------------------------------

def update_eth(verbose: bool = True) -> None:
    path = os.path.join(DATA_DIR, "eth_daily.csv")
    existing = _load_daily(path)
    last = _last_date(existing)

    if verbose:
        print(f"\n{'='*50}")
        print(f"ETH  — last date: {last or 'none (first run)'}")

    today = date.today()
    start = (last + timedelta(days=1)).strftime("%Y-%m-%d") if last else "2017-01-01"
    end   = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    if last and last >= today:
        if verbose:
            print("  [ETH] already up to date.")
        return

    if verbose:
        print(f"  [ETH] downloading yfinance ETH-USD from {start} …")
    new_data = _download_yf("ETH-USD", start, end)
    if verbose:
        print(f"  [ETH] yfinance returned {len(new_data)} rows.")

    combined = pd.concat([existing, new_data], ignore_index=True)
    _save(combined, path)

    result = _load_daily(path)
    if verbose:
        print(f"  [ETH] saved {len(result)} rows → {path}")
        print(f"         from {result[DATE_COL].min()} to {result[DATE_COL].max()}")


# ---------------------------------------------------------------------------
# S&P 500
# ---------------------------------------------------------------------------

def update_snp500(verbose: bool = True) -> None:
    path = os.path.join(DATA_DIR, "snp500_daily.csv")
    existing = _load_daily(path)
    last = _last_date(existing)

    if verbose:
        print(f"\n{'='*50}")
        print(f"S&P500 — last date: {last or 'none (first run)'}")

    today = date.today()
    start = (last + timedelta(days=1)).strftime("%Y-%m-%d") if last else "2012-01-01"
    end   = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    if last and last >= today:
        if verbose:
            print("  [SNP] already up to date.")
        return

    if verbose:
        print(f"  [SNP] downloading yfinance ^GSPC from {start} …")
    new_data = _download_yf("^GSPC", start, end)
    if verbose:
        print(f"  [SNP] yfinance returned {len(new_data)} rows.")

    combined = pd.concat([existing, new_data], ignore_index=True)
    _save(combined, path)

    result = _load_daily(path)
    if verbose:
        print(f"  [SNP] saved {len(result)} rows → {path}")
        print(f"         from {result[DATE_COL].min()} to {result[DATE_COL].max()}")


# ---------------------------------------------------------------------------
# DXY (Dollar Index)
# ---------------------------------------------------------------------------

def update_dxy(verbose: bool = True) -> None:
    path = os.path.join(DATA_DIR, "dxy_daily.csv")
    existing = _load_daily(path)
    last = _last_date(existing)

    if verbose:
        print(f"\n{'='*50}")
        print(f"DXY   — last date: {last or 'none (first run)'}")

    today = date.today()
    start = (last + timedelta(days=1)).strftime("%Y-%m-%d") if last else "2012-01-01"
    end   = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    if last and last >= today:
        if verbose:
            print("  [DXY] already up to date.")
        return

    if verbose:
        print(f"  [DXY] downloading yfinance DX-Y.NYB from {start} …")
    new_data = _download_yf("DX-Y.NYB", start, end)
    if verbose:
        print(f"  [DXY] yfinance returned {len(new_data)} rows.")

    combined = pd.concat([existing, new_data], ignore_index=True)
    _save(combined, path)

    result = _load_daily(path)
    if verbose:
        print(f"  [DXY] saved {len(result)} rows → {path}")
        print(f"         from {result[DATE_COL].min()} to {result[DATE_COL].max()}")


# ---------------------------------------------------------------------------
# VIX (Volatility Index)
# ---------------------------------------------------------------------------

def update_vix(verbose: bool = True) -> None:
    path = os.path.join(DATA_DIR, "vix_daily.csv")
    existing = _load_daily(path)
    last = _last_date(existing)

    if verbose:
        print(f"\n{'='*50}")
        print(f"VIX   — last date: {last or 'none (first run)'}")

    today = date.today()
    start = (last + timedelta(days=1)).strftime("%Y-%m-%d") if last else "2012-01-01"
    end   = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    if last and last >= today:
        if verbose:
            print("  [VIX] already up to date.")
        return

    if verbose:
        print(f"  [VIX] downloading yfinance ^VIX from {start} …")
    new_data = _download_yf("^VIX", start, end)
    if verbose:
        print(f"  [VIX] yfinance returned {len(new_data)} rows.")

    combined = pd.concat([existing, new_data], ignore_index=True)
    _save(combined, path)

    result = _load_daily(path)
    if verbose:
        print(f"  [VIX] saved {len(result)} rows → {path}")
        print(f"         from {result[DATE_COL].min()} to {result[DATE_COL].max()}")


# ---------------------------------------------------------------------------
# Helpers for single-value series (Date, Value)
# ---------------------------------------------------------------------------

def _load_single(path: str) -> pd.DataFrame:
    """Load a Date,Value CSV; returns empty DataFrame if absent."""
    if not os.path.isfile(path):
        return pd.DataFrame(columns=[DATE_COL, "Value"])
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.date
    return df.sort_values(DATE_COL).reset_index(drop=True)


def _save_single(df: pd.DataFrame, path: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    df = df.sort_values(DATE_COL).drop_duplicates(DATE_COL).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.strftime("%Y-%m-%d")
    df["Value"] = df["Value"].round(6)
    df[[DATE_COL, "Value"]].to_csv(path, index=False)


# ---------------------------------------------------------------------------
# US 10-Year Treasury Yield
# ---------------------------------------------------------------------------

def update_us10y(verbose: bool = True) -> None:
    path = os.path.join(DATA_DIR, "us10y_daily.csv")
    existing = _load_daily(path)
    last = _last_date(existing)

    if verbose:
        print(f"\n{'='*50}")
        print(f"US10Y — last date: {last or 'none (first run)'}")

    today = date.today()
    start = (last + timedelta(days=1)).strftime("%Y-%m-%d") if last else "2012-01-01"
    end   = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    if last and last >= today:
        if verbose:
            print("  [US10Y] already up to date.")
        return

    if verbose:
        print(f"  [US10Y] downloading yfinance ^TNX from {start} …")
    new_data = _download_yf("^TNX", start, end)
    if verbose:
        print(f"  [US10Y] yfinance returned {len(new_data)} rows.")

    combined = pd.concat([existing, new_data], ignore_index=True)
    _save(combined, path)

    result = _load_daily(path)
    if verbose:
        print(f"  [US10Y] saved {len(result)} rows → {path}")
        print(f"           from {result[DATE_COL].min()} to {result[DATE_COL].max()}")


# ---------------------------------------------------------------------------
# Fed Funds Rate (FRED — DFF, daily effective rate)
# ---------------------------------------------------------------------------

def update_fedfunds(verbose: bool = True) -> None:
    import requests
    from io import StringIO

    path = os.path.join(DATA_DIR, "fedfunds_daily.csv")
    existing = _load_single(path)
    last = _last_date(existing)

    if verbose:
        print(f"\n{'='*50}")
        print(f"FedFunds — last date: {last or 'none (first run)'}")

    today = date.today()
    if last and last >= today:
        if verbose:
            print("  [FED] already up to date.")
        return

    start = (last + timedelta(days=1)).strftime("%Y-%m-%d") if last else "2012-01-01"

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"
    if verbose:
        print(f"  [FED] fetching FRED DFF from {start} …")
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    df.columns = [DATE_COL, "Value"]
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.date
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df[df[DATE_COL] >= date(2012, 1, 1)].dropna()

    if last is not None:
        df = df[df[DATE_COL] > last]

    if verbose:
        print(f"  [FED] fetched {len(df)} new rows.")

    combined = pd.concat([existing, df], ignore_index=True)
    _save_single(combined, path)

    result = _load_single(path)
    if verbose:
        print(f"  [FED] saved {len(result)} rows → {path}")
        print(f"         from {result[DATE_COL].min()} to {result[DATE_COL].max()}")


# ---------------------------------------------------------------------------
# Oil (WTI Crude — CL=F)
# ---------------------------------------------------------------------------

def update_oil(verbose: bool = True) -> None:
    path = os.path.join(DATA_DIR, "oil_daily.csv")
    existing = _load_daily(path)
    last = _last_date(existing)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Oil   — last date: {last or 'none (first run)'}")

    today = date.today()
    start = (last + timedelta(days=1)).strftime("%Y-%m-%d") if last else "2012-01-01"
    end   = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    if last and last >= today:
        if verbose:
            print("  [OIL] already up to date.")
        return

    if verbose:
        print(f"  [OIL] downloading yfinance CL=F from {start} …")
    new_data = _download_yf("CL=F", start, end)
    if verbose:
        print(f"  [OIL] yfinance returned {len(new_data)} rows.")

    combined = pd.concat([existing, new_data], ignore_index=True)
    _save(combined, path)

    result = _load_daily(path)
    if verbose:
        print(f"  [OIL] saved {len(result)} rows → {path}")
        print(f"         from {result[DATE_COL].min()} to {result[DATE_COL].max()}")


# ---------------------------------------------------------------------------
# Silver (SI=F)
# ---------------------------------------------------------------------------

def update_silver(verbose: bool = True) -> None:
    path = os.path.join(DATA_DIR, "silver_daily.csv")
    existing = _load_daily(path)
    last = _last_date(existing)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Silver — last date: {last or 'none (first run)'}")

    today = date.today()
    start = (last + timedelta(days=1)).strftime("%Y-%m-%d") if last else "2012-01-01"
    end   = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    if last and last >= today:
        if verbose:
            print("  [SLV] already up to date.")
        return

    if verbose:
        print(f"  [SLV] downloading yfinance SI=F from {start} …")
    new_data = _download_yf("SI=F", start, end)
    if verbose:
        print(f"  [SLV] yfinance returned {len(new_data)} rows.")

    combined = pd.concat([existing, new_data], ignore_index=True)
    _save(combined, path)

    result = _load_daily(path)
    if verbose:
        print(f"  [SLV] saved {len(result)} rows → {path}")
        print(f"         from {result[DATE_COL].min()} to {result[DATE_COL].max()}")


# ---------------------------------------------------------------------------
# BTC Funding Rate (Binance BTCUSDT perpetual — daily average)
# Disponible depuis septembre 2019. Format: Date, Value (moyenne journalière).
# ---------------------------------------------------------------------------

def update_funding_rate(verbose: bool = True) -> None:
    import requests as _req
    import time

    path = os.path.join(DATA_DIR, "btc_funding_rate_daily.csv")
    existing = _load_single(path)
    last = _last_date(existing)

    if verbose:
        print(f"\n{'='*50}")
        print(f"BTC Funding Rate — last date: {last or 'none (first run)'}")

    today = date.today()
    if last and last >= today:
        if verbose:
            print("  [FUND] already up to date.")
        return

    start_dt = datetime(last.year, last.month, last.day) + timedelta(days=1) \
               if last else datetime(2019, 9, 10)
    end_dt   = datetime(today.year, today.month, today.day) + timedelta(days=1)

    url    = "https://fapi.binance.com/fapi/v1/fundingRate"
    symbol = "BTCUSDT"
    limit  = 1000
    all_rows = []
    current = start_dt

    if verbose:
        print(f"  [FUND] fetching Binance funding rates from {start_dt.date()} …")

    while current < end_dt:
        window_end = min(current + timedelta(days=90), end_dt)
        params = {
            "symbol":    symbol,
            "startTime": int(current.timestamp() * 1000),
            "endTime":   int(window_end.timestamp() * 1000),
            "limit":     limit,
        }
        r = _req.get(url, params=params, timeout=30)
        r.raise_for_status()
        chunk = r.json()
        if chunk:
            all_rows.extend(chunk)
        # Advance by the window size regardless of chunk size
        current = window_end
        time.sleep(0.1)

    if not all_rows:
        if verbose:
            print("  [FUND] no new data.")
        return

    df = pd.DataFrame(all_rows)
    df[DATE_COL] = pd.to_datetime(df["fundingTime"], unit="ms").dt.date
    df["Value"]  = df["fundingRate"].astype(float)
    # Average the three daily funding rates (00:00, 08:00, 16:00 UTC)
    daily = df.groupby(DATE_COL)["Value"].mean().reset_index()
    daily = daily[daily[DATE_COL] < today]

    if verbose:
        print(f"  [FUND] {len(daily)} daily rows after aggregation.")

    combined = pd.concat([existing, daily], ignore_index=True)
    _save_single(combined, path)

    result = _load_single(path)
    if verbose:
        print(f"  [FUND] saved {len(result)} rows → {path}")
        print(f"          from {result[DATE_COL].min()} to {result[DATE_COL].max()}")


# ---------------------------------------------------------------------------
# BTC Hash Rate (Blockchain.info — TH/s, daily)
# ---------------------------------------------------------------------------

def update_hashrate(verbose: bool = True) -> None:
    import requests as _req

    path = os.path.join(DATA_DIR, "btc_hashrate_daily.csv")
    existing = _load_single(path)
    last = _last_date(existing)

    if verbose:
        print(f"\n{'='*50}")
        print(f"BTC HashRate — last date: {last or 'none (first run)'}")

    today = date.today()
    if last and last >= today:
        if verbose:
            print("  [HASH] already up to date.")
        return

    if verbose:
        print("  [HASH] fetching blockchain.info hash-rate (all history) …")

    url = "https://api.blockchain.info/charts/hash-rate?timespan=all&format=json&sampled=false"
    r = _req.get(url, timeout=60)
    r.raise_for_status()
    values = r.json()["values"]

    df = pd.DataFrame(values)          # columns: x (unix ts), y (hash rate)
    df[DATE_COL] = pd.to_datetime(df["x"], unit="s").dt.normalize()
    df = df.rename(columns={"y": "Value"}).drop(columns="x")
    df = df[df[DATE_COL] >= pd.Timestamp("2012-01-01")]
    df[DATE_COL] = df[DATE_COL].dt.date

    if last is not None:
        df = df[df[DATE_COL] > last]

    if verbose:
        print(f"  [HASH] {len(df)} new rows.")

    combined = pd.concat([existing, df], ignore_index=True)
    _save_single(combined, path)

    result = _load_single(path)
    if verbose:
        print(f"  [HASH] saved {len(result)} rows → {path}")
        print(f"          from {result[DATE_COL].min()} to {result[DATE_COL].max()}")


# ---------------------------------------------------------------------------
# MVRV & NUPL (CoinMetrics community API)
# NUPL is derived from MVRV: NUPL = 1 - (1 / MVRV)
# ---------------------------------------------------------------------------

def _fetch_coinmetrics(metric: str, start: str, verbose: bool) -> pd.DataFrame:
    """Paginate through CoinMetrics community API for a single BTC metric."""
    import requests as _req

    url    = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    rows   = []
    params = {
        "assets":     "btc",
        "metrics":    metric,
        "start_time": start,
        "page_size":  1000,
    }

    while True:
        r = _req.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        rows.extend(data.get("data", []))
        next_token = data.get("next_page_token")
        if not next_token:
            break
        # Keep original params and add the pagination token
        params = {**params, "next_page_token": next_token}
        params.pop("start_time", None)  # start_time conflicts with next_page_token

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df[DATE_COL] = pd.to_datetime(df["time"]).dt.date
    df["Value"]  = pd.to_numeric(df[metric], errors="coerce")
    return df[[DATE_COL, "Value"]].dropna()


def update_mvrv_nupl(verbose: bool = True) -> None:
    mvrv_path = os.path.join(DATA_DIR, "btc_mvrv_daily.csv")
    nupl_path = os.path.join(DATA_DIR, "btc_nupl_daily.csv")

    existing_mvrv = _load_single(mvrv_path)
    last = _last_date(existing_mvrv)

    if verbose:
        print(f"\n{'='*50}")
        print(f"MVRV/NUPL — last date: {last or 'none (first run)'}")

    today = date.today()
    if last and last >= today:
        if verbose:
            print("  [MVRV] already up to date.")
        return

    start = (last + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z") if last else "2012-01-01T00:00:00Z"

    if verbose:
        print(f"  [MVRV] fetching CoinMetrics CapMVRVCur from {start[:10]} …")

    new_mvrv = _fetch_coinmetrics("CapMVRVCur", start, verbose)
    if verbose:
        print(f"  [MVRV] {len(new_mvrv)} new rows.")

    # MVRV
    combined_mvrv = pd.concat([existing_mvrv, new_mvrv], ignore_index=True)
    _save_single(combined_mvrv, mvrv_path)

    # NUPL = 1 - (1 / MVRV)
    all_mvrv = _load_single(mvrv_path)
    nupl_df  = all_mvrv.copy()
    nupl_df["Value"] = (1 - (1 / nupl_df["Value"])).round(6)
    nupl_df.to_csv(nupl_path, index=False)

    result = _load_single(mvrv_path)
    if verbose:
        print(f"  [MVRV] saved {len(result)} rows → {mvrv_path}")
        print(f"  [NUPL] saved {len(result)} rows → {nupl_path}")
        print(f"          from {result[DATE_COL].min()} to {result[DATE_COL].max()}")


# ---------------------------------------------------------------------------
# Google Trends "bitcoin" (weekly, rescaled across overlapping windows)
# Granularity: 1-year windows → weekly data (7-day intervals).
# Consecutive windows share a 3-month overlap used to rescale new data
# onto the existing scale.
# Incremental: on subsequent runs only the last 1–2 windows are fetched.
# ---------------------------------------------------------------------------

def _fetch_trends_window(pt, ws: date, we: date, retries: int = 3) -> pd.DataFrame:
    """Fetch a single Google Trends window with retry on rate-limit."""
    import time
    timeframe = f"{ws.strftime('%Y-%m-%d')} {we.strftime('%Y-%m-%d')}"
    for attempt in range(retries):
        try:
            pt.build_payload(["bitcoin"], timeframe=timeframe)
            time.sleep(2)
            df = pt.interest_over_time()
            if df.empty:
                return pd.DataFrame()
            df = df[["bitcoin"]].rename(columns={"bitcoin": "Value"})
            df.index = pd.to_datetime(df.index).date
            df.index.name = DATE_COL
            return df.astype(float)
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"    window {ws}→{we} attempt {attempt+1} failed: {e}. Retrying in {wait}s …")
            time.sleep(wait)
    return pd.DataFrame()


def update_google_trends(verbose: bool = True) -> None:
    """Fetch Google Trends 'bitcoin' interest as a single request covering 2012→today.

    For periods > 5 years, pytrends returns monthly data normalised to the true
    global peak (2021 ATH = 100).  A single request avoids window-stitching
    entirely: any multi-window approach fails for Bitcoin because interest grew
    ~100× from 2012 to 2021, so the rescaling ratio between adjacent windows
    always amplifies future values beyond 100 and clip() saturates everything.

    Granularity: monthly (one row per month).  Forward-fill to daily in the
    ML preprocessing pipeline.
    Update cadence: re-fetches the full history once per month.
    """
    import time

    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("  [TRENDS] pytrends not installed. Run: pip install pytrends")
        return

    path = os.path.join(DATA_DIR, "google_trends_bitcoin.csv")
    existing = _load_single(path)
    last = _last_date(existing)

    today = date.today()

    if verbose:
        print(f"\n{'='*50}")
        print(f"Google Trends — last date: {last or 'none (first run)'}")

    # Monthly data: refresh once a month is enough
    if last and (today - last).days < 30:
        if verbose:
            print("  [TRENDS] already up to date (updated within last 30 days).")
        return

    pt = TrendReq(hl="en-US", tz=0)

    # Single request: 2012-01-01 → today.
    # pytrends returns monthly granularity for periods > 5 years, with all
    # values correctly normalised relative to the global maximum in the range.
    timeframe = f"2012-01-01 {today.strftime('%Y-%m-%d')}"
    if verbose:
        print(f"  [TRENDS] fetching full history 2012→{today} (monthly, single request) …")

    df = pd.DataFrame()
    for attempt in range(3):
        try:
            pt.build_payload(["bitcoin"], timeframe=timeframe)
            time.sleep(2)
            df = pt.interest_over_time()
            if not df.empty:
                break
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"    attempt {attempt + 1} failed: {e}. Retrying in {wait}s …")
            time.sleep(wait)

    if df.empty:
        print("  [TRENDS] fetch failed — no data returned.")
        return

    df = df[["bitcoin"]].rename(columns={"bitcoin": "Value"}).astype(float)
    df.index = pd.to_datetime(df.index).date
    df.index.name = DATE_COL

    result_df = df.reset_index()
    result_df.columns = [DATE_COL, "Value"]
    result_df["Value"] = result_df["Value"].clip(0, 100).round(2)
    result_df[DATE_COL] = pd.to_datetime(result_df[DATE_COL]).dt.date

    _save_single(result_df, path)

    result = _load_single(path)
    if verbose:
        print(f"  [TRENDS] saved {len(result)} rows → {path}")
        print(f"            from {result[DATE_COL].min()} to {result[DATE_COL].max()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    update_btc()
    update_xau()
    update_eth()
    update_snp500()
    update_dxy()
    update_vix()
    update_us10y()
    update_fedfunds()
    update_oil()
    update_silver()
    update_funding_rate()
    update_hashrate()
    update_mvrv_nupl()
    update_google_trends()
    print("\nDone.")
