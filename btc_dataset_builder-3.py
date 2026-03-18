"""
=============================================================
  BTC Price Prediction — Dataset Builder  v2.0
=============================================================
  Pas de temps : 1 minute (Binance klines, pagination auto)
  Sources :
    1. OHLCV 1m          → Binance REST API (BTCUSDT)
    2. Indicateurs tech.  → RSI, MACD, BB, ATR, EMA, VWAP…
    3. Actifs macro       → yfinance 1d  (SP500, Gold, DXY, ETH, NASDAQ, VIX)
                            + resampling / forward-fill sur le calendrier 1m
    4. On-chain / marché  → CoinGecko API (market cap, dominance, volume global)
    5. Fear & Greed       → alternative.me (journalier, ffill)
    6. Actualités NLP     → Yahoo Finance News
                            + Google News RSS (20 requêtes)
                            + CryptoPanic RSS
                            → sentiment VADER agrégé par heure, ffill sur 1m
    7. Labels cibles      → return_5m / 15m / 1h (régression + direction)

  Dépendances :
    pip install requests pandas numpy yfinance feedparser vaderSentiment tqdm
=============================================================
"""

import time
import datetime
import warnings
import requests
import feedparser
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


def _to_naive(idx: pd.Index) -> pd.Index:
    """Convertit tout DatetimeIndex en tz-naive. Opération idempotente."""
    idx = pd.to_datetime(idx)
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def _naive_ts(ts) -> pd.Timestamp:
    """Convertit n'importe quelle valeur de date/timestamp en Timestamp tz-naive."""
    try:
        t = pd.Timestamp(ts)
    except Exception:
        return pd.Timestamp.utcnow().tz_localize(None)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════
START_DATE   = "2010-01-01"       # Binance 1m → recommandé < 2 ans (fichiers volumineux)
END_DATE     = datetime.date.today().strftime("%Y-%m-%d")
OUTPUT_FILE  = "btc_1m_training_dataset.csv"

BINANCE_SYMBOL   = "BTCUSDT"
BINANCE_INTERVAL = "1h"
BINANCE_LIMIT    = 1000           # max par requête (limite Binance)

CORRELATED_TICKERS = {
    "SP500"  : "^GSPC",
    "GOLD"   : "GC=F",
    "DXY"    : "DX-Y.NYB",
    "ETH"    : "ETH-USD",
    "NASDAQ" : "^IXIC",
    "VIX"    : "^VIX",
}

# ── Requêtes RSS pour le scraping large
RSS_QUERIES = [
    # Crypto direct
    "Bitcoin",
    "BTC price",
    "Bitcoin ETF",
    "crypto regulation SEC",
    "Bitcoin halving",
    "crypto exchange hack",
    "stablecoin USDT USDC",
    "Ethereum upgrade",
    # Macro & géopolitique
    "Federal Reserve interest rate",
    "US inflation CPI PPI",
    "US dollar index DXY",
    "China economy GDP",
    "geopolitical risk war sanctions",
    "oil price OPEC",
    "stock market crash",
    "banking crisis",
    # Institutionnel
    "BlackRock Bitcoin",
    "MicroStrategy Bitcoin",
    "crypto whale",
    "Bitcoin mining hashrate",
]

GOOGLE_NEWS_RSS  = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
CRYPTOPANIC_RSS  = "https://cryptopanic.com/news/rss/"


# ═══════════════════════════════════════════════════════
# 1. OHLCV 1 MINUTE — Binance REST API (paginé)
# ═══════════════════════════════════════════════════════
def _ts_ms(dt: datetime.datetime) -> int:
    """datetime → timestamp millisecondes UTC."""
    return int(dt.timestamp() * 1000)


def fetch_binance_1m(start: str, end: str) -> pd.DataFrame:
    """
    Télécharge les klines 1m BTCUSDT depuis Binance en paginant
    automatiquement par blocs de 1000 bougies (~16h40 par requête).
    Pas de clé API requise pour les données publiques.
    """
    print("\n[1/6] Téléchargement OHLCV 1m — Binance (BTCUSDT)...")
    url      = "https://api.binance.com/api/v3/klines"
    start_dt = datetime.datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.datetime.strptime(end,   "%Y-%m-%d")

    all_rows = []
    current  = start_dt
    total_days = (end_dt - start_dt).days
    pbar = tqdm(total=total_days, unit="jours", desc="  Binance klines")

    while current < end_dt:
        params = {
            "symbol"    : BINANCE_SYMBOL,
            "interval"  : BINANCE_INTERVAL,
            "startTime" : _ts_ms(current),
            "endTime"   : _ts_ms(end_dt),
            "limit"     : BINANCE_LIMIT,
        }
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"\n  ⚠ Erreur Binance : {e} — retry dans 5s")
            time.sleep(5)
            continue

        if not data:
            break

        all_rows.extend(data)
        last_ts = data[-1][0]
        next_dt = datetime.datetime.fromtimestamp(last_ts / 1000) + datetime.timedelta(minutes=1)
        pbar.n = min((next_dt - start_dt).days, total_days)
        pbar.refresh()
        current = next_dt
        time.sleep(0.08)

    pbar.close()

    if not all_rows:
        raise ValueError("Binance : aucune donnée retournée. Vérifiez la connexion réseau.")

    cols = ["open_time","open","high","low","close","volume",
            "close_time","quote_volume","trades",
            "taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(all_rows, columns=cols)

    for c in ["open","high","low","close","volume","quote_volume",
              "taker_buy_base","taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)

    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index("date").sort_index()
    df = df.drop(columns=["open_time","close_time","ignore"])
    df = df[~df.index.duplicated(keep="first")]

    # Rendements multi-horizons
    df["return_1m"]   = df["close"].pct_change(1)
    df["return_5m"]   = df["close"].pct_change(5)
    df["return_15m"]  = df["close"].pct_change(15)
    df["return_1h"]   = df["close"].pct_change(60)
    df["log_return"]  = np.log(df["close"] / df["close"].shift(1))
    df["range_pct"]   = (df["high"] - df["low"]) / df["close"]

    # Pression acheteuse (taker buy)
    df["buy_pressure"] = df["taker_buy_base"] / (df["volume"] + 1e-9)

    print(f"  → {len(df):,} bougies 1m  ({df.index[0]}  →  {df.index[-1]})")
    return df


# ═══════════════════════════════════════════════════════
# 2. INDICATEURS TECHNIQUES
# ═══════════════════════════════════════════════════════
def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/6] Calcul des indicateurs techniques...")
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # EMA
    for p in [9, 21, 50, 200]:
        df[f"ema_{p}"] = c.ewm(span=p, adjust=False).mean()

    # Croisements EMA
    df["ema_cross_9_21"]  = (df["ema_9"]  > df["ema_21"]).astype(int)
    df["ema_cross_21_50"] = (df["ema_21"] > df["ema_50"]).astype(int)

    # RSI
    def rsi(series, p=14):
        d  = series.diff()
        g  = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
        lo = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
        return 100 - 100 / (1 + g / (lo + 1e-9))

    df["rsi_14"] = rsi(c, 14)
    df["rsi_7"]  = rsi(c, 7)

    # MACD
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = e12 - e26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Bollinger Bands (20, 2σ)
    bm = c.rolling(20).mean()
    bs = c.rolling(20).std()
    df["bb_upper"]    = bm + 2*bs
    df["bb_lower"]    = bm - 2*bs
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / (bm + 1e-9)
    df["bb_position"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # ATR (14)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["atr_14"]  = tr.ewm(span=14, adjust=False).mean()
    df["atr_pct"] = df["atr_14"] / (c + 1e-9)

    # VWAP journalier
    df["vwap"] = (
        (c * v).groupby(df.index.date).cumsum()
        / v.groupby(df.index.date).cumsum()
    )
    df["vwap_dist"] = (c - df["vwap"]) / (df["vwap"] + 1e-9)

    # Stochastique (14, 3)
    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Williams %R (14)
    df["williams_r"] = -100 * (high14 - c) / (high14 - low14 + 1e-9)

    # Volume
    df["vol_ema_20"] = v.ewm(span=20, adjust=False).mean()
    df["vol_ratio"]  = v / (df["vol_ema_20"] + 1e-9)

    # OBV
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv"]     = obv
    df["obv_ema"] = obv.ewm(span=20, adjust=False).mean()

    # Momentum / ROC
    df["momentum_14"] = c - c.shift(14)
    df["roc_10"]      = c.pct_change(10)

    # Features temporelles
    df["hour"]        = df.index.hour
    df["minute"]      = df.index.minute
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    print("  → 30+ indicateurs calculés")
    return df


# ═══════════════════════════════════════════════════════
# 3. ACTIFS MACRO (yfinance 1d → forward-fill sur index 1m)
# ═══════════════════════════════════════════════════════
def fetch_macro_daily(start: str, end: str) -> pd.DataFrame:
    print("\n[3/6] Téléchargement actifs macro (yfinance 1d)...")
    frames = []
    for name, ticker in CORRELATED_TICKERS.items():
        try:
            data = yf.download(ticker, start=start, end=end,
                               interval="1d", auto_adjust=True, progress=False)
            if data.empty:
                print(f"  ⚠ Vide : {name}")
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            close = data["Close"].squeeze()
            close.index = _to_naive(close.index)
            ret = close.pct_change(1)
            s = pd.concat([close.rename(f"{name.lower()}_close"),
                           ret.rename(f"{name.lower()}_ret1d")], axis=1)
            frames.append(s)
            print(f"  ✓ {name}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  ✗ {name} : {e}")

    if not frames:
        return pd.DataFrame()
    macro = pd.concat(frames, axis=1)
    macro.index.name = "date"
    return macro


def merge_macro_on_1m(df_1m: pd.DataFrame, macro_daily: pd.DataFrame) -> pd.DataFrame:
    if macro_daily.empty:
        return df_1m
    macro_daily.index = _to_naive(macro_daily.index)
    macro_ri = macro_daily.reindex(df_1m.index, method="ffill")
    return df_1m.join(macro_ri, how="left")


# ═══════════════════════════════════════════════════════
# 4. COINGECKO — Market cap, volume global, dominance
# ═══════════════════════════════════════════════════════
def fetch_coingecko_daily(start: str, end: str) -> pd.DataFrame:
    print("\n[4/6] Téléchargement CoinGecko (market cap, dominance)...")
    start_ts = int(datetime.datetime.strptime(start, "%Y-%m-%d").timestamp())
    end_ts   = int(datetime.datetime.strptime(end,   "%Y-%m-%d").timestamp())
    records  = []

    try:
        url  = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        resp = requests.get(url, params={"vs_currency":"usd","from":start_ts,"to":end_ts}, timeout=20)
        data = resp.json()
        mc  = pd.DataFrame(data["market_caps"],   columns=["ts","btc_market_cap"])
        vol = pd.DataFrame(data["total_volumes"], columns=["ts","btc_cg_volume"])
        mc["date"]  = pd.to_datetime(mc["ts"],  unit="ms").dt.normalize()
        vol["date"] = pd.to_datetime(vol["ts"], unit="ms").dt.normalize()
        cg = mc[["date","btc_market_cap"]].merge(vol[["date","btc_cg_volume"]], on="date")
        cg = cg.set_index("date")
        cg["btc_mcap_change"] = cg["btc_market_cap"].pct_change(1)
        records.append(cg)
        print("  ✓ BTC market cap + volume")
        time.sleep(1.5)
    except Exception as e:
        print(f"  ⚠ BTC market cap : {e}")

    if not records:
        return pd.DataFrame()
    result = pd.concat(records, axis=1)
    result.index = _to_naive(result.index)
    result.index.name = "date"
    return result


# ═══════════════════════════════════════════════════════
# 5. FEAR & GREED INDEX
# ═══════════════════════════════════════════════════════
def fetch_fear_greed(limit: int = 2000) -> pd.DataFrame:
    print("\n[5/6] Téléchargement Fear & Greed Index...")
    try:
        resp = requests.get(
            f"https://api.alternative.me/fng/?limit={limit}&format=json", timeout=15)
        data = resp.json()["data"]
        df   = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
        df["fear_greed_value"] = df["value"].astype(float)
        mapping = {"Extreme Fear":0,"Fear":1,"Neutral":2,"Greed":3,"Extreme Greed":4}
        df["fear_greed_ord"] = df["value_classification"].map(mapping)
        df = df[["date","fear_greed_value","fear_greed_ord"]].set_index("date").sort_index()
        df.index = _to_naive(df.index)
        print(f"  → {len(df)} jours")
        return df
    except Exception as e:
        print(f"  ⚠ Fear & Greed : {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════
# 6. SCRAPING LARGE — Actualités + Sentiment NLP
# ═══════════════════════════════════════════════════════
def fetch_yahoo_news() -> list:
    items = []
    try:
        for item in (yf.Ticker("BTC-USD").news or []):
            ct    = item.get("content", {})
            title = ct.get("title") or item.get("title", "")
            raw   = ct.get("pubDate") or item.get("providerPublishTime")
            dt    = _naive_ts(raw) if raw else pd.Timestamp.utcnow().tz_localize(None)
            if title:
                items.append({"dt": dt, "title": title, "source": "yahoo"})
    except Exception as e:
        print(f"  ⚠ Yahoo News : {e}")
    return items


def fetch_google_news_rss(queries: list) -> list:
    items = []
    for q in tqdm(queries, desc="  Google News RSS"):
        try:
            feed = feedparser.parse(GOOGLE_NEWS_RSS.format(query=q.replace(" ", "+")))
            for e in feed.entries:
                # published_parsed est un time.struct_time en UTC — on reconstruit proprement
                try:
                    dt = _naive_ts(datetime.datetime(*e.published_parsed[:6]))
                except Exception:
                    dt = pd.Timestamp.utcnow().tz_localize(None)
                items.append({"dt": dt, "title": e.get("title",""), "source":"google"})
            time.sleep(0.4)
        except Exception:
            pass
    return items


def fetch_cryptopanic_rss() -> list:
    items = []
    try:
        feed = feedparser.parse(CRYPTOPANIC_RSS)
        for e in feed.entries:
            try:
                dt = _naive_ts(datetime.datetime(*e.published_parsed[:6]))
            except Exception:
                dt = pd.Timestamp.utcnow().tz_localize(None)
            items.append({"dt": dt, "title": e.get("title",""), "source":"cryptopanic"})
        print(f"  ✓ CryptoPanic RSS : {len(items)} articles")
    except Exception as e:
        print(f"  ⚠ CryptoPanic : {e}")
    return items


def compute_sentiment_hourly(news_items: list) -> pd.DataFrame:
    """Agrège le sentiment VADER par heure (ffill ensuite sur les minutes)."""
    analyzer = SentimentIntensityAnalyzer()
    records  = []
    for item in news_items:
        if not item.get("title"):
            continue
        s = analyzer.polarity_scores(item["title"])
        records.append({
            "hour"    : _naive_ts(item["dt"]).floor("h"),
            "compound": s["compound"],
            "pos"     : s["pos"],
            "neg"     : s["neg"],
        })
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    agg = df.groupby("hour").agg(
        sentiment_mean      = ("compound","mean"),
        sentiment_std       = ("compound","std"),
        sentiment_pos_ratio = ("pos",     "mean"),
        sentiment_neg_ratio = ("neg",     "mean"),
        news_count          = ("compound","count"),
    ).fillna(0)
    agg["sentiment_bull_bear"] = (
        (agg["sentiment_pos_ratio"] - agg["sentiment_neg_ratio"])
        / (agg["sentiment_pos_ratio"] + agg["sentiment_neg_ratio"] + 1e-9)
    )
    agg.index = _to_naive(agg.index)
    agg.index.name = "hour"
    return agg


def build_sentiment_features() -> pd.DataFrame:
    print("\n[6/6] Collecte des actualités & analyse sentiment NLP...")
    news = []
    yf_news = fetch_yahoo_news()
    news += yf_news
    print(f"  → Yahoo Finance : {len(yf_news)} articles")
    gn_news = fetch_google_news_rss(RSS_QUERIES)
    news += gn_news
    print(f"  → Google News RSS : {len(gn_news)} articles")
    cp_news = fetch_cryptopanic_rss()
    news += cp_news
    print(f"  Total : {len(news)} articles | Calcul sentiment horaire...")
    sent = compute_sentiment_hourly(news)
    print(f"  → Sentiment agrégé sur {len(sent)} heures")
    return sent


# ═══════════════════════════════════════════════════════
# 7. LABELS
# ═══════════════════════════════════════════════════════
def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["label_return_5m"]  = df["close"].pct_change(5).shift(-5)
    df["label_return_15m"] = df["close"].pct_change(15).shift(-15)
    df["label_return_1h"]  = df["close"].pct_change(60).shift(-60)
    df["label_dir_5m"]     = (df["label_return_5m"]  > 0).astype(int)
    df["label_dir_15m"]    = (df["label_return_15m"] > 0).astype(int)
    df["label_dir_1h"]     = (df["label_return_1h"]  > 0).astype(int)
    return df


# ═══════════════════════════════════════════════════════
# ASSEMBLAGE FINAL
# ═══════════════════════════════════════════════════════
def build_dataset() -> pd.DataFrame:

    btc  = fetch_binance_1m(START_DATE, END_DATE)
    # ── Garantie absolue : index principal toujours tz-naive
    btc.index = _to_naive(btc.index)
    btc  = compute_technical_indicators(btc)

    macro = fetch_macro_daily(START_DATE, END_DATE)
    btc   = merge_macro_on_1m(btc, macro)

    cg = fetch_coingecko_daily(START_DATE, END_DATE)
    if not cg.empty:
        btc = btc.join(cg.reindex(btc.index, method="ffill"), how="left")

    fg = fetch_fear_greed()
    if not fg.empty:
        btc = btc.join(fg.reindex(btc.index, method="ffill"), how="left")

    sent = build_sentiment_features()
    if not sent.empty:
        hour_idx = btc.index.floor("h")
        sv = sent.reindex(hour_idx, method="ffill")
        sv.index = btc.index
        btc = btc.join(sv, how="left")

    btc = add_labels(btc)

    # Nettoyage
    btc = btc.iloc[200:]          # warm-up indicateurs
    ffill_cols = [c for c in btc.columns if any(
        c.startswith(k) for k in ["sp500","gold","dxy","eth","nasdaq","vix",
                                   "btc_market","btc_cg","btc_dom",
                                   "fear_greed","sentiment","news_count"])]
    btc[ffill_cols] = btc[ffill_cols].ffill()
    sent_cols = [c for c in btc.columns if "sentiment" in c or "news_count" in c]
    btc[sent_cols] = btc[sent_cols].fillna(0)
    btc = btc.dropna(subset=["label_return_1h"])

    print(f"\n{'='*60}")
    print(f"  Dataset final : {btc.shape[0]:,} lignes × {btc.shape[1]} colonnes")
    print(f"  Période       : {btc.index[0]}  →  {btc.index[-1]}")
    print(f"  Mémoire       : {btc.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"{'='*60}\n")
    for i, col in enumerate(btc.columns, 1):
        print(f"  {i:>3}. {col}")

    return btc


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    df = build_dataset()
    print(f"\n💾 Sauvegarde → {OUTPUT_FILE}  ({len(df):,} lignes)")
    df.to_csv(OUTPUT_FILE)
    print("✅  Terminé.")
    print("\n── Aperçu (3 premières lignes) ──")
    print(df.head(3).to_string())
    print("\n── Valeurs manquantes restantes ──")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    print("  Aucune." if miss.empty else miss.to_string())
