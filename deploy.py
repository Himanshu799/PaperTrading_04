#!/usr/bin/env python3
import os
import time
from time import perf_counter
from typing import Tuple, Dict

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from alpaca_trade_api.rest import REST, TimeFrame, APIError

# === Extra libs to mirror training ===
import joblib
from tensorflow.keras.models import load_model
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from scipy.signal import detrend
from PyEMD import CEEMDAN

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — must match training choices
# ──────────────────────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "JPM", "AMZN", "TSLA", "MSFT"]  # same order as training

# CNN-LSTM sequence length used in training
SEQ_LEN = int(os.environ.get("SEQ_LEN", 60))

# PPO model path (SB3 appends .zip internally; we strip it to avoid .zip.zip bugs)
MODEL_PATH = os.environ.get("MODEL_PATH", "ppo_ceemd_cnnlstm_rl.zip")
MODEL_PATH = MODEL_PATH[:-4] if MODEL_PATH.endswith(".zip") else MODEL_PATH

# CNN-LSTM weights path (from your training script)
CNN_MODEL_PATH = os.environ.get("CNN_MODEL_PATH", "models/final_multi_stock_cnn_lstm.h5")

# Preprocessed training artifacts
DATA_DIR = os.environ.get("DATA_DIR", "processed_data")  # contains *_scaler_X.pkl and max_feat_dim.npy

# Trading
INITIAL_CASH   = float(os.environ.get("INITIAL_CASH", 10_000))
SLEEP_INTERVAL = int(os.environ.get("SLEEP_INTERVAL", 60))  # seconds between loops

# Alpaca
ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Live data history needed to compute CEEMD + indicators robustly
HIST_BARS = int(os.environ.get("HIST_BARS", max(600, SEQ_LEN + 300)))  # be generous; CEEMD needs history

# CEEMDAN params (match training)
CE_TRIALS     = 100
CE_NOISE      = 0.05
CE_MAX_IMF    = 6
CE_PARALLEL   = True
CE_N_JOBS     = -1

# Technical Indicators (match training)
RSI_WIN       = 14
MACD_FAST     = 12
MACD_SLOW     = 26
MACD_SIG      = 9
BB_WIN        = 20
BB_DEV        = 2

# ──────────────────────────────────────────────────────────────────────────────
# Load artifacts
# ──────────────────────────────────────────────────────────────────────────────
def _const(v):
    return lambda _progress: v

def load_artifacts() -> Tuple[Dict[str, object], int, object, object]:
    # per-ticker feature scaler_X
    scalers_X = {}
    for t in TICKERS:
        tl = t.lower()
        p = os.path.join(DATA_DIR, f"{tl}_scaler_X.pkl")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing scaler for {t}: {p}\n"
                f"Re-run your CNN-LSTM training with saving scalers_X per ticker."
            )
        scalers_X[t] = joblib.load(p)

    # max feature dimension used during training concatenation/padding
    mpath = os.path.join(DATA_DIR, "max_feat_dim.npy")
    if not os.path.exists(mpath):
        raise FileNotFoundError(
            f"Missing max_feat_dim: {mpath}\n"
            f"Save it during training (np.save('processed_data/max_feat_dim.npy', [max_feat_dim]))."
        )
    max_feat_dim = int(np.load(mpath)[0])

    # load CNN-LSTM
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(
            f"Missing CNN-LSTM model weights: {CNN_MODEL_PATH}\n"
            f"Train saved it as 'models/final_multi_stock_cnn_lstm.h5'."
        )
    cnn = load_model(CNN_MODEL_PATH)

    # load PPO (with robust deserializers)
    agent = PPO.load(
        MODEL_PATH,
        custom_objects={
            # set to your training values if different
            "lr_schedule": _const(2.5e-4),
            "clip_range": _const(0.2),
        },
    )
    return scalers_X, max_feat_dim, cnn, agent

# ──────────────────────────────────────────────────────────────────────────────
# Alpaca client
# ──────────────────────────────────────────────────────────────────────────────
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")

def get_recent_bars(sym: str, limit: int) -> pd.DataFrame:
    """
    Fetch recent minute bars. Ensure lowercase columns and keep needed fields.
    """
    bars = api.get_bars(symbol=sym, timeframe=TimeFrame.Minute, limit=limit).df
    if bars.empty:
        return pd.DataFrame()
    bars = bars.reset_index()
    bars.columns = [str(c).lower() for c in bars.columns]
    cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in bars.columns]
    return bars[cols].copy()

# ──────────────────────────────────────────────────────────────────────────────
# Feature builder to mirror training: CEEMD + TA → pad → scale → seq → CNN pred
# Returns per-ticker state vector and latest price
# ──────────────────────────────────────────────────────────────────────────────
def build_state_block(
    sym: str,
    scalers_X: Dict[str, object],
    max_feat_dim: int,
    cnn_model
) -> Tuple[np.ndarray, float]:
    """
    Returns:
      state_vector: 1D array length (max_feat_dim + 1) = [last scaled features, cnn_pred]
      latest_price: float
    """
    df = get_recent_bars(sym, limit=HIST_BARS)
    if df.empty or len(df) < SEQ_LEN + 5:
        raise ValueError(f"Not enough bars for {sym} (have {len(df)}, need >= {SEQ_LEN + 5})")

    close = df["close"].astype(float)
    vol   = df["volume"].astype(float)

    # --- TA (match training) ---
    rsi      = RSIIndicator(close=close, window=RSI_WIN, fillna=True).rsi()
    macd_obj = MACD(close=close, window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIG)
    bb_obj   = BollingerBands(close=close, window=BB_WIN, window_dev=BB_DEV)

    tech = pd.concat([
        vol.rename("Volume"),
        rsi.rename("RSI"),
        macd_obj.macd().rename("MACD"),
        macd_obj.macd_signal().rename("MACD_Signal"),
        bb_obj.bollinger_hband().rename("BB_High"),
        bb_obj.bollinger_lband().rename("BB_Low"),
    ], axis=1).ffill().bfill()

    # --- CEEMD on detrended close (match training) ---
    sig    = close.to_numpy(dtype=np.float64)
    sig_dt = detrend(sig)
    ce     = CEEMDAN(trials=CE_TRIALS, noise_width=CE_NOISE, max_imf=CE_MAX_IMF,
                     parallel=CE_PARALLEL, n_jobs=CE_N_JOBS)
    tarr   = np.arange(len(sig_dt), dtype=np.float64)
    imfs   = ce.ceemdan(sig_dt, tarr)  # (n_imfs, T)

    # Select IMFs (same rule)
    energies      = (imfs ** 2).sum(axis=1)
    energy_ratios = energies / energies.sum()
    cors          = np.array([np.corrcoef(imf, sig_dt)[0, 1] for imf in imfs])
    keep          = (energy_ratios >= 0.02) & (np.abs(cors) >= 0.1)
    sel_imfs      = imfs[keep]
    imf_chan      = sel_imfs.T  # (T, n_sel)

    # Build features in the same order: [IMFs..., tech]
    X_raw = np.hstack([imf_chan, tech.values])  # (T, F_current)

    # Pad/trim to max_feat_dim used in training concatenation
    if X_raw.shape[1] < max_feat_dim:
        pad = np.zeros((X_raw.shape[0], max_feat_dim - X_raw.shape[1]), dtype=X_raw.dtype)
        X_raw = np.hstack([X_raw, pad])
    elif X_raw.shape[1] > max_feat_dim:
        X_raw = X_raw[:, :max_feat_dim]

    # Take last SEQ_LEN rows and scale with per‑ticker scaler_X
    if X_raw.shape[0] < SEQ_LEN:
        raise ValueError(f"{sym}: have {X_raw.shape[0]} feature rows < SEQ_LEN={SEQ_LEN}")
    X_seq_raw = X_raw[-SEQ_LEN:, :]  # (SEQ_LEN, max_feat_dim)
    scaler_X  = scalers_X[sym]
    X_seq     = scaler_X.transform(X_seq_raw)

    # CNN-LSTM inference
    seq3d  = X_seq[np.newaxis, :, :]            # (1, SEQ_LEN, max_feat_dim)
    y_pred = float(cnn_model.predict(seq3d, verbose=0).flatten()[0])

    # State vector (last-step scaled features + cnn prediction)
    X_base = X_seq[-1, :]                        # (max_feat_dim,)
    state_vector = np.hstack([X_base, [y_pred]]).astype(np.float32)  # (max_feat_dim + 1,)

    latest_price = float(close.iloc[-1])
    return state_vector, latest_price

# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────
def softmax(weights: np.ndarray) -> np.ndarray:
    w = weights.astype(np.float64).ravel()
    w = w - np.max(w)
    e = np.exp(w)
    return e / (np.sum(e) + 1e-12)

# ──────────────────────────────────────────────────────────────────────────────
# Main trading loop
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("▶️  Starting RL Alpaca trading loop. Ctrl+C to exit.", flush=True)

    # Load artifacts (scalers, max_feat_dim, CNN, PPO)
    scalers_X, max_feat_dim, cnn, agent = load_artifacts()

    # Validate obs size against the PPO model's expectation
    state_dim = max_feat_dim + 1   # per ticker
    expected_obs = int(np.prod(agent.observation_space.shape))
    current_obs  = len(TICKERS) * state_dim + 1 + len(TICKERS)  # states + cash_ratio + positions
    if expected_obs != current_obs:
        raise ValueError(
            f"Observation mismatch: model expects {expected_obs}, but deploy builds {current_obs}. "
            f"K={len(TICKERS)}, state_dim={state_dim} (max_feat_dim={max_feat_dim} + 1 pred). "
            f"Ensure TICKERS order/count matches training."
        )

    # Portfolio tracker (local)
    portfolio = {sym: {"shares": 0} for sym in TICKERS}

    try:
        while True:
            loop_start = perf_counter()
            now = pd.Timestamp.now(tz="UTC").tz_convert("US/Eastern")
            print(f"\n🔄 Loop start: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)

            # 1) Market check
            clock = api.get_clock()
            if not clock.is_open:
                print(f"  ❌ Market closed, next open at {clock.next_open}")
                time.sleep(SLEEP_INTERVAL)
                continue
            print(f"  ⏱ Market open, next close at {clock.next_close}")

            # 2) Refresh account & cash
            account = api.get_account()
            cash = float(account.cash)

            # 3) Build observation from CEEMD+CNN state vectors
            blocks = []
            latest_prices = []
            for sym in TICKERS:
                try:
                    state_vec, px = build_state_block(sym, scalers_X, max_feat_dim, cnn)
                    blocks.append(state_vec)
                    latest_prices.append(px)
                except Exception as e:
                    print(f"  ⚠️  {sym}: {e}")
                    # if any ticker fails, skip this loop to avoid wrong obs shape
                    blocks = []
                    break

            if not blocks or len(blocks) != len(TICKERS):
                print("  ⚠️  Missing state(s), skipping loop.")
                time.sleep(SLEEP_INTERVAL)
                continue

            per_asset = np.concatenate(blocks, axis=0).astype(np.float32)  # K * (max_feat_dim+1)
            latest_prices = np.array(latest_prices, dtype=np.float64)

            # Local portfolio net worth
            shares_vec = np.array([portfolio[s]["shares"] for s in TICKERS], dtype=np.float64)
            pos_val = float(np.sum(shares_vec * latest_prices))
            net_worth = cash + pos_val
            if net_worth <= 0:
                print("  ❌ Net worth non-positive, skipping.")
                time.sleep(SLEEP_INTERVAL)
                continue

            cash_ratio = np.array([cash / net_worth], dtype=np.float64)  # (1,)
            obs = np.concatenate([per_asset, cash_ratio, shares_vec], axis=0).astype(np.float32).reshape(1, -1)

            # Final safety check (should be redundant)
            if obs.shape[1] != expected_obs:
                print(f"  ❌ Obs shape {obs.shape[1]} != expected {expected_obs}. Skipping.")
                time.sleep(SLEEP_INTERVAL)
                continue

            # 4) Policy -> weights (long-only, sum ≤ 1)
            raw_action, _ = agent.predict(obs, deterministic=True)
            weights = np.clip(raw_action.reshape(-1), 0.0, 1.0)
            total_w = float(weights.sum())
            if total_w > 1.0:
                weights /= total_w
            # Alternatively, always allocate all via softmax:
            # weights = softmax(raw_action) * 0.98

            # 5) Target allocations
            investable_value = net_worth
            target_values = weights * investable_value
            current_values = shares_vec * latest_prices
            deltas = target_values - current_values  # $ to add/remove per asset

            # 6) Rebalance with market orders (minimal sanity checks)
            for i, sym in enumerate(TICKERS):
                px = latest_prices[i]
                # integral shares
                target_shares = int(target_values[i] // max(px, 1e-12))
                cur_sh = portfolio[sym]["shares"]
                to_trade = target_shares - cur_sh
                if to_trade == 0:
                    continue

                if to_trade > 0:
                    est_cost = to_trade * px
                    if est_cost > cash * 0.99:   # leave a little cash headroom
                        to_trade = int((cash * 0.99) // max(px, 1e-12))
                    if to_trade <= 0:
                        continue
                    try:
                        api.submit_order(symbol=sym, qty=to_trade, side="buy", type="market", time_in_force="day")
                        portfolio[sym]["shares"] += to_trade
                        cash -= to_trade * px
                        print(f"  ✅ BUY  {to_trade:4d} {sym} @ {px:.2f}")
                    except APIError as e:
                        print(f"  ❌ BUY {sym}: {e}")
                else:
                    sell_qty = min(-to_trade, cur_sh)
                    if sell_qty <= 0:
                        continue
                    try:
                        api.submit_order(symbol=sym, qty=sell_qty, side="sell", type="market", time_in_force="day")
                        portfolio[sym]["shares"] -= sell_qty
                        cash += sell_qty * px
                        print(f"  ✅ SELL {sell_qty:4d} {sym} @ {px:.2f}")
                    except APIError as e:
                        print(f"  ❌ SELL {sym}: {e}")

            # 7) Loop summary & sleep
            loop_time = perf_counter() - loop_start
            next_run = now + pd.Timedelta(seconds=SLEEP_INTERVAL)
            print(f"✅ Loop done in {loop_time:.2f}s. Next run ~ {next_run.strftime('%H:%M:%S %Z')}", flush=True)
            time.sleep(SLEEP_INTERVAL)

    except KeyboardInterrupt:
        print("🛑  Stopped by user", flush=True)
    except Exception as e:
        print(f"⚠️  Error in loop: {e}", flush=True)
        time.sleep(5)
