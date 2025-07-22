#!/usr/bin/env python3

import os
import time
import numpy as np
import pandas as pd
from time import perf_counter

from stable_baselines3 import PPO
from alpaca_trade_api.rest import REST, TimeFrame, APIError

# ─── CONFIG ──────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "JPM", "AMZN", "TSLA", "MSFT"]
PERFORMANCE_WEIGHTS = {
    "AAPL": 0.10,
    "JPM":  0.12,
    "AMZN": 0.26,
    "TSLA": 0.18,
    "MSFT": 0.34
}
assert abs(sum(PERFORMANCE_WEIGHTS.values()) - 1) < 1e-3, "Weights must sum to 1"
SEQ_LEN = 60
INITIAL_BALANCE = 10_000
SLEEP_INTERVAL = 60  # seconds between loops

# Alpaca credentials from environment or your config
ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# ─── RL AGENT LOAD ────────────────────────────────────────────────────────
agent = PPO.load("cnn_lstm_multi_stock_ppo.zip")

# ─── ALPACA CLIENT ───────────────────────────────────────────────────────
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# ─── PORTFOLIO STATE ─────────────────────────────────────────────────────
portfolio = {s: {"shares": 0} for s in TICKERS}

# ─── FEATURE/BAR HELPERS (provide your actual versions) ──────────────────
def get_recent_bars(sym: str) -> pd.DataFrame:
    bars = api.get_bars(symbol=sym, timeframe=TimeFrame.Minute, limit=SEQ_LEN+20)
    data = [{"t": b.t, "open": b.o, "high": b.h, "low": b.l, "close": b.c, "volume": b.v} for b in bars]
    return pd.DataFrame(data).set_index("t")

def extract_features_intraday(sym: str, df: pd.DataFrame) -> np.ndarray:
    # IMPORTANT: Replace this with your real feature extractor!
    # This dummy version assumes 11 features per stock (adjust if you use more/less).
    return np.zeros(11)  # Change 11 to your actual per-stock feature count.

# ─── MAIN TRADING LOOP ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("▶️  Starting RL Alpaca trading loop. Ctrl+C to exit.", flush=True)
    try:
        while True:
            loop_start = perf_counter()
            now = pd.Timestamp.now()
            print(f"\n🔄 Loop start: {now.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

            # 1) Market check
            clock = api.get_clock()
            if not clock.is_open:
                print(f"  ❌ Market closed, next open at {clock.next_open}")
                time.sleep(SLEEP_INTERVAL)
                continue
            print(f"  ⏱ Market open, next close at {clock.next_close}")

            # 2) Refresh account & cash
            account = api.get_account()
            cash_avail = float(account.cash)
            print(f"  💰 Cash available: ${cash_avail:.2f}")

            # 3) Batch feature and price collection
            features_list, prices_list, skip_idx = [], [], []
            for i, sym in enumerate(TICKERS):
                try:
                    df_bar = get_recent_bars(sym)
                    if df_bar.empty:
                        skip_idx.append(i)
                        print(f"  ⚠️  No bars for {sym}")
                        continue
                    price = float(df_bar["close"].iloc[-1])
                    feats = extract_features_intraday(sym, df_bar)
                    features_list.append(feats)
                    prices_list.append(price)
                except Exception as e:
                    skip_idx.append(i)
                    print(f"  ⚠️  Error for {sym}: {e}")

            if len(features_list) != len(TICKERS):
                print("  ⚠️  Missing data for one or more tickers, skipping loop.")
                time.sleep(SLEEP_INTERVAL)
                continue

            obs_features = np.concatenate(features_list)        # e.g. (5*11,) = (55,)
            obs_prices = np.array(prices_list)                  # (5,)
            portfolio_shares = np.array([portfolio[sym]["shares"] for sym in TICKERS])  # (5,)
            obs = np.concatenate([
                obs_features,
                obs_prices,
                [cash_avail],
                portfolio_shares
            ]).astype(np.float32)                               # e.g. (55+5+1+5,) = (66,)

            # 4) RL agent allocation weights
            # IMPORTANT: Do not reshape! SB3 expects (N_FEATURES,) not (1, N_FEATURES)
            weights, _ = agent.predict(obs, deterministic=True)
            weights = np.clip(weights, 0, 1)
            total_weight = weights.sum()
            if total_weight > 1:
                weights /= total_weight

            # 5) Cap by performance weights
            total_equity = cash_avail + np.sum(portfolio_shares * obs_prices)
            perf_alloc = np.array([PERFORMANCE_WEIGHTS[sym] for sym in TICKERS]) * total_equity
            alloc_dollars = np.minimum(weights * total_equity, perf_alloc)

            # 6) Rebalance via Alpaca orders
            for i, sym in enumerate(TICKERS):
                price = obs_prices[i]
                target_val = alloc_dollars[i]
                curr_val = portfolio[sym]["shares"] * price
                diff_val = target_val - curr_val
                if abs(diff_val) < price:
                    continue  # Ignore if < 1 share
                shares_to_trade = int(abs(diff_val) // price)
                if shares_to_trade == 0:
                    continue

                if diff_val > 0 and cash_avail >= shares_to_trade * price:
                    # BUY
                    try:
                        api.submit_order(symbol=sym, qty=shares_to_trade,
                                         side="buy", type="market", time_in_force="gtc")
                        portfolio[sym]["shares"] += shares_to_trade
                        cash_avail -= shares_to_trade * price
                        print(f"  ✅ BUY  {shares_to_trade} {sym} @ {price:.2f}")
                    except APIError as e:
                        print(f"  ❌ BUY failed for {sym}: {e}")
                elif diff_val < 0 and portfolio[sym]["shares"] >= shares_to_trade:
                    # SELL
                    try:
                        api.submit_order(symbol=sym, qty=shares_to_trade,
                                         side="sell", type="market", time_in_force="gtc")
                        portfolio[sym]["shares"] -= shares_to_trade
                        cash_avail += shares_to_trade * price
                        print(f"  ✅ SELL {shares_to_trade} {sym} @ {price:.2f}")
                    except APIError as e:
                        print(f"  ❌ SELL failed for {sym}: {e}")

            # 7) Loop summary & sleep
            loop_time = perf_counter() - loop_start
            next_run = now + pd.Timedelta(seconds=SLEEP_INTERVAL)
            print(f"✅ Loop done in {loop_time:.2f}s. Next run at {next_run.strftime('%H:%M:%S')}\n", flush=True)

            time.sleep(SLEEP_INTERVAL)

    except KeyboardInterrupt:
        print("🛑  Stopped by user", flush=True)
    except Exception as e:
        print("⚠️  Error in loop:", e, flush=True)
        time.sleep(5)
