"""
S&P 500 OHLCV data collector — hourly format.

Strategy:
  - Last 730 days      → real hourly bars (^GSPC, 1h interval via yfinance)
  - Before 730 days    → daily bars (1d interval), one entry per trading day
  - Auto-update        → if data/snp500_1h_data.csv already exists, only
                         downloads data after the last recorded timestamp.

Output: data/snp500_1h_data.csv
Format: semicolon-separated, matching XAU_1h_data.csv
  Timestamp;Open;High;Low;Close;Volume   (YYYY.MM.DD HH:MM, UTC)
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone

TICKER      = "^GSPC"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "snp500_1h_data.csv")
START_DATE  = "2012-01-01"
DATE_FMT    = "%Y.%m.%d %H:%M"
# yfinance only keeps hourly bars for the last 730 days
HOURLY_LOOKBACK_DAYS = 729


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the multi-level columns yfinance ≥ 1.0 returns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def _to_utc_naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert a tz-aware or tz-naive DatetimeIndex to UTC-naive."""
    if index.tzinfo is not None or (
        hasattr(index, "tz") and index.tz is not None
    ):
        return index.tz_convert("UTC").tz_localize(None)
    return index


def _download(start: str, end: str, interval: str) -> pd.DataFrame:
    """Download data from yfinance and return a clean OHLCV DataFrame."""
    raw = yf.download(
        TICKER,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        return pd.DataFrame()

    df = _flatten(raw)
    df.index = _to_utc_naive(df.index)

    # Keep only needed columns (yfinance names may vary slightly)
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "open":   col_map[c] = "Open"
        elif cl == "high": col_map[c] = "High"
        elif cl == "low":  col_map[c] = "Low"
        elif cl == "close":col_map[c] = "Close"
        elif cl == "volume": col_map[c] = "Volume"
    df = df.rename(columns=col_map)[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Timestamp"
    return df


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_existing() -> pd.DataFrame:
    """Load existing CSV and return a DataFrame with a DatetimeIndex."""
    if not os.path.isfile(OUTPUT_FILE):
        return pd.DataFrame()
    df = pd.read_csv(OUTPUT_FILE, sep=";", parse_dates=["Timestamp"],
                     date_format=DATE_FMT)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format=DATE_FMT)
    df = df.set_index("Timestamp")
    return df


def collect(verbose: bool = True) -> None:
    existing = load_existing()

    now_utc   = datetime.now(timezone.utc).replace(tzinfo=None)
    today_str = now_utc.strftime("%Y-%m-%d")

    # Boundary between daily and hourly zones
    hourly_start_dt = now_utc - timedelta(days=HOURLY_LOOKBACK_DAYS)
    hourly_start    = hourly_start_dt.strftime("%Y-%m-%d")

    if existing.empty:
        daily_start = START_DATE
        if verbose:
            print(f"No existing file — downloading from {START_DATE}.")
    else:
        last_ts     = existing.index.max()
        daily_start = (last_ts + timedelta(days=1)).strftime("%Y-%m-%d")
        if verbose:
            print(f"Existing data up to {last_ts.strftime(DATE_FMT)} — "
                  f"downloading from {daily_start}.")

    frames = []

    # --- Daily portion (2012 → hourly_start) ---
    if daily_start < hourly_start:
        if verbose:
            print(f"  Downloading daily bars  {daily_start} → {hourly_start} …")
        df_daily = _download(daily_start, hourly_start, "1d")
        if not df_daily.empty:
            # Set time component to 14:30 UTC (NYSE open in winter / EST)
            df_daily.index = df_daily.index.normalize() + pd.Timedelta(hours=14, minutes=30)
            frames.append(df_daily)
            if verbose:
                print(f"    → {len(df_daily)} daily rows")

    # --- Hourly portion (hourly_start → today) ---
    h_start = max(daily_start, hourly_start)
    if h_start <= today_str:
        if verbose:
            print(f"  Downloading hourly bars  {h_start} → {today_str} …")
        df_hourly = _download(h_start, today_str, "1h")
        if not df_hourly.empty:
            frames.append(df_hourly)
            if verbose:
                print(f"    → {len(df_hourly)} hourly rows")

    if not frames:
        if verbose:
            print("Nothing new to download.")
        return

    new_data = pd.concat(frames).sort_index()
    new_data = new_data[~new_data.index.duplicated(keep="last")]

    # Merge with existing
    if not existing.empty:
        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_data

    # Remove any rows that are fully NaN
    combined.dropna(how="all", inplace=True)

    # Format and save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    combined.index = combined.index.strftime(DATE_FMT)
    combined.index.name = "Timestamp"
    combined.to_csv(OUTPUT_FILE, sep=";", float_format="%.6f")

    if verbose:
        print(f"\nSaved {len(combined)} rows to {OUTPUT_FILE}")
        print(f"  From : {combined.index[0]}")
        print(f"  To   : {combined.index[-1]}")


if __name__ == "__main__":
    collect()
