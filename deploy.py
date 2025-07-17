#!/usr/bin/env python3
"""
deploy.py

Intraday deployment script (1-minute bars) that:
1. Fetches recent OHLCV bars from Alpaca
2. Computes technicals + CEEMDAN IMFs (fixed count)
3. Builds the same SEQ_LEN-window features you used in training
4. Runs CNN–LSTM to get last-step features + scaled price prediction
5. Feeds that into your PPO agent to pick 0/hold,1/buy,2/sell
6. Submits market orders on Alpaca paper API, handling buying-power errors
"""

import time
import numpy as np
import pandas as pd

from scipy.signal            import detrend
from PyEMD                    import CEEMDAN
from ta.momentum             import RSIIndicator
from ta.trend                import MACD
from ta.volatility           import BollingerBands
from sklearn.preprocessing   import MinMaxScaler
from tensorflow.keras.models import load_model
from stable_baselines3       import PPO
from alpaca_trade_api.rest   import REST, TimeFrame, APIError

# ─── 0) USER CONFIG ─────────────────────────────────────────────────────────────
TICKERS         = ["AAPL","JPM","AMZN","TSLA","MSFT"]
SEQ_LEN         = 60
INITIAL_BALANCE = 10_000
TRAIN_YEAR      = 2022

ALPACA_API_KEY    = "PK4NLA1IF8I62FZVEJMT"
ALPACA_SECRET_KEY = "U9bNYdnf4nWRGxZ3bjUVSAimfSP8d3bwHNa5r3S9"
ALPACA_BASE_URL   = "https://paper-api.alpaca.markets"

TIMEFRAME      = TimeFrame.Minute
LOOKBACK_BARS  = SEQ_LEN + 20
SLEEP_INTERVAL = 60    # seconds between loops

# ─── 1) LOAD MODELS & RECONSTRUCT SCALERS ────────────────────────────────────────
cnn_lstm = load_model("models/final_multi_stock_cnn_lstm.h5", compile=False)
agent    = PPO.load("cnn_lstm_multi_stock_ppo.zip")

raw_X     = {}
dates_idx = {}
scaler_X  = {}
for ticker in TICKERS:
    tl     = ticker.lower()
    X_full = np.load(f"preprocessed_data/{tl}_raw_X.npy")
    df_tech = pd.read_csv(f"preprocessed_data/{tl}_tech_scaled_features.csv",
                          index_col=0, parse_dates=True)
    idx     = df_tech.index
    mask    = idx.year <= TRAIN_YEAR

    scaler_X[ticker] = MinMaxScaler().fit(X_full[mask])
    raw_X[ticker]    = X_full
    dates_idx[ticker] = idx

# tech columns always = 6; used to compute how many IMFs you trained on
N_TECH = 6
# pad-to width for the CNN–LSTM input
max_feat_dim = max(v.shape[1] for v in raw_X.values())

# ─── 2) ALPACA CLIENT SETUP ─────────────────────────────────────────────────────
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# ─── 3) HELPERS ─────────────────────────────────────────────────────────────────
def compute_technical(df: pd.DataFrame) -> pd.DataFrame:
    c, v = df["close"], df["volume"]
    rsi    = RSIIndicator(c, window=14, fillna=True).rsi()
    macd_o = MACD(c, window_slow=26, window_fast=12, window_sign=9)
    bb     = BollingerBands(c, window=20, window_dev=2)

    tech = pd.concat([
        v.rename("volume"),
        rsi.rename("rsi"),
        macd_o.macd().rename("macd"),
        macd_o.macd_signal().rename("macd_signal"),
        bb.bollinger_hband().rename("bb_high"),
        bb.bollinger_lband().rename("bb_low"),
    ], axis=1).ffill().bfill().fillna(0.0)

    return tech

def compute_ceemdan_imfs(close: pd.Series) -> (np.ndarray, np.ndarray):
    sig    = close.to_numpy(dtype=np.float64)
    sig_dt = detrend(sig)
    ceemd  = CEEMDAN(trials=100, noise_width=0.05, max_imf=6, parallel=True, n_jobs=-1)
    imfs   = ceemd.ceemdan(sig_dt, np.arange(len(sig_dt), dtype=float))

    # ── sanitize any NaNs/Infs
    imfs = np.nan_to_num(imfs, nan=0.0, posinf=0.0, neginf=0.0)

    # compute per-IMF energies and correlations
    energies = (imfs**2).sum(axis=1)
    cors     = np.array([np.corrcoef(imf, sig_dt)[0,1] for imf in imfs])

    # select relevant IMFs
    keep = (energies/energies.sum() >= 0.02) & (np.abs(cors) >= 0.1)
    
    # RETURN both the filtered IMFs *and* their energies
    return imfs[keep], energies[keep]



def extract_features_intraday(ticker: str, df: pd.DataFrame) -> np.ndarray:
    tech = compute_technical(df)                      # (T,6)
    imfs_keep, energies = compute_ceemdan_imfs(df["close"])  # (n_sel,T),(n_sel,)

    total_feats  = scaler_X[ticker].scale_.shape[0]
    n_imfs_train = total_feats - N_TECH

    # pick top‐energy IMFs (or pad with zero‐rows)
    if imfs_keep.shape[0] >= n_imfs_train:
        order = np.argsort(energies)[::-1]
        imfs_sel = imfs_keep[order[:n_imfs_train]]
    else:
        pad_rows  = np.zeros((n_imfs_train - imfs_keep.shape[0], imfs_keep.shape[1]))
        imfs_sel  = np.vstack([imfs_keep, pad_rows])

    X_raw = np.hstack([imfs_sel.T, tech.values])
    if X_raw.shape[1] != total_feats:
        raise ValueError(f"[{ticker}] got {X_raw.shape[1]} feats vs {total_feats}")

    X_scaled = scaler_X[ticker].transform(X_raw)
    if X_scaled.shape[1] < max_feat_dim:
        pad = np.zeros((X_scaled.shape[0], max_feat_dim - X_scaled.shape[1]))
        X_scaled = np.hstack([X_scaled, pad])

    X_seq  = X_scaled[-SEQ_LEN:]
    y_pred = float(cnn_lstm.predict(X_seq[np.newaxis,:,:]).flatten()[0])
    last_f = X_seq[-1]

    return np.hstack([last_f, y_pred])

def get_recent_bars(sym: str) -> pd.DataFrame:
    bars = api.get_bars(symbol=sym, timeframe=TIMEFRAME,
                        limit=LOOKBACK_BARS, adjustment="raw")
    data = [{"t": b.t, "open": b.o, "high": b.h,
             "low": b.l, "close": b.c, "volume": b.v} for b in bars]
    return pd.DataFrame(data).set_index("t")

# ─── 4) PORTFOLIO TRACKING ─────────────────────────────────────────────────────
portfolio = {s: {"shares": 0} for s in TICKERS}

from time import perf_counter

if __name__ == "__main__":
    print("▶️  Starting intraday live deploy. Ctrl+C to exit.", flush=True)
    try:
        while True:
            loop_start = perf_counter()
            now = pd.Timestamp.now()
            print(f"\n🔄 Loop start: {now.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

            # 1) Check market status
            t0 = perf_counter()
            clock = api.get_clock()
            market_open = clock.is_open
            market_msg = (f"Market open, next close at {clock.next_close}"
                          if market_open
                          else f"Market closed, next open at {clock.next_open}")
            print(f"  ⏱ Market check ({perf_counter()-t0:.3f}s): {market_msg}", flush=True)
            if not market_open:
                time.sleep(SLEEP_INTERVAL)
                continue

            # 2) Refresh buying power
            t1 = perf_counter()
            account = api.get_account()
            cash_avail = float(account.cash)
            if not np.isfinite(cash_avail) or cash_avail < 0:
                print(f"  ⚠️  Invalid cash_avail ({cash_avail}), resetting to 0", flush=True)
                cash_avail = 0.0
            print(f"  💰 Cash avail ({perf_counter()-t1:.3f}s): ${cash_avail:.2f}", flush=True)

            # 3) Per‐ticker processing
            for sym in TICKERS:
                print(f"  ▶ {sym}", flush=True)

                # 3a) Fetch bars
                t2 = perf_counter()
                df_bar = get_recent_bars(sym)
                if df_bar.empty:
                    print(f"    ⚠️  No bars fetched ({perf_counter()-t2:.3f}s), skipping", flush=True)
                    continue
                price = float(df_bar["close"].iloc[-1])
                print(f"    📊 Fetch+Price ({perf_counter()-t2:.3f}s): price={price:.2f}", flush=True)

                # 3b) Feature extraction
                t3 = perf_counter()
                try:
                    feats = extract_features_intraday(sym, df_bar)
                except Exception as e:
                    print(f"    ⚠️  Feature error ({perf_counter()-t3:.3f}s): {e}", flush=True)
                    continue
                print(f"    🛠 Features ({perf_counter()-t3:.3f}s)", flush=True)

                # 3c) Build observation
                obs = np.concatenate([feats, [price, cash_avail, portfolio[sym]["shares"]]]).astype(np.float32)
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

                # 3d) Action selection
                t4 = perf_counter()
                action, _ = agent.predict(obs, deterministic=True)
                print(f"    🤖 Action ({perf_counter()-t4:.3f}s): {['Hold','Buy','Sell'][action]}", flush=True)

                # 3e) Order execution
                t5 = perf_counter()
                if action == 1 and cash_avail >= price:
                    qty = int(cash_avail // price)
                    try:
                        api.submit_order(symbol=sym, qty=qty,
                                         side="buy", type="market", time_in_force="gtc")
                        portfolio[sym]["shares"] += qty
                        cash_avail -= qty * price
                        print(f"    ✅ BUY  {qty}@{price:.2f}", flush=True)
                    except APIError as e:
                        print(f"    ❌ BUY failed: {e}", flush=True)

                elif action == 2 and portfolio[sym]["shares"] > 0:
                    qty = portfolio[sym]["shares"]
                    try:
                        api.submit_order(symbol=sym, qty=qty,
                                         side="sell", type="market", time_in_force="gtc")
                        portfolio[sym]["shares"] = 0
                        cash_avail += qty * price
                        print(f"    ✅ SELL {qty}@{price:.2f}", flush=True)
                    except APIError as e:
                        print(f"    ❌ SELL failed: {e}", flush=True)

                print(f"    ⏳ Order exec ({perf_counter()-t5:.3f}s)", flush=True)

            # 4) Loop summary & sleep
            loop_time = perf_counter() - loop_start
            next_run = now + pd.Timedelta(seconds=SLEEP_INTERVAL)
            print(f"✅ Loop done in {loop_time:.2f}s. Next run at {next_run.strftime('%H:%M:%S')}\n", flush=True)

            time.sleep(SLEEP_INTERVAL)

    except KeyboardInterrupt:
        print("🛑  Stopped by user", flush=True)

    except Exception as e:
        print("⚠️  Error in loop:", e, flush=True)
        time.sleep(5)