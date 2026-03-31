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
from datetime import date, timedelta

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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    update_btc()
    update_xau()
    update_eth()
    update_snp500()
    print("\nDone.")
