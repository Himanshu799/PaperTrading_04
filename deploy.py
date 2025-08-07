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
SEQ_LEN = 10  # Must match your RL window_size
INITIAL_BALANCE = 10_000
SLEEP_INTERVAL = 60  # seconds between loops

ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# ─── RL AGENT LOAD ────────────────────────────────────────────────────────
agent = PPO.load("ppo_multistock_rl")  # Your trained model

# ─── ALPACA CLIENT ───────────────────────────────────────────────────────
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# ─── PORTFOLIO STATE ─────────────────────────────────────────────────────
portfolio = {s: {"shares": 0} for s in TICKERS}

# ─── FEATURE/BAR HELPERS ─────────────────────────────────────────────────
def get_recent_bars(sym: str) -> pd.DataFrame:
    bars = api.get_bars(symbol=sym, timeframe=TimeFrame.Minute, limit=SEQ_LEN+1).df
    if bars.empty:
        return pd.DataFrame()
    bars = bars.tail(SEQ_LEN)  # Ensure correct length
    bars = bars.reset_index()
    return bars

def get_normalized_close_window(sym: str) -> np.ndarray:
    df = get_recent_bars(sym)
    if df.empty or len(df) < SEQ_LEN:
        raise ValueError(f"Not enough bars for {sym}")
    closes = df['close'].to_numpy()
    return closes / closes[0]  # normalize to first value

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

            # 3) Build RL observation (normalized price windows + cash ratio + shares)
            price_windows, latest_prices, skip_idx = [], [], []
            for sym in TICKERS:
                try:
                    norm_window = get_normalized_close_window(sym)
                    df_bar = get_recent_bars(sym)
                    price = float(df_bar['close'].iloc[-1])
                    price_windows.append(norm_window)
                    latest_prices.append(price)
                except Exception as e:
                    skip_idx.append(sym)
                    print(f"  ⚠️  Data error for {sym}: {e}")

            if len(price_windows) != len(TICKERS):
                print("  ⚠️  Missing data for one or more tickers, skipping loop.")
                time.sleep(SLEEP_INTERVAL)
                continue

            price_windows = np.array(price_windows).flatten()
            latest_prices = np.array(latest_prices)
            portfolio_shares = np.array([portfolio[sym]["shares"] for sym in TICKERS], dtype=np.float32)
            cash_ratio = cash_avail / INITIAL_BALANCE
            obs = np.concatenate([price_windows, [cash_ratio], portfolio_shares]).astype(np.float32).reshape(1, -1)

            # 4) RL agent output (allocation weights)
            weights, _ = agent.predict(obs, deterministic=True)
            weights = np.clip(weights, 0, 1)
            total_weight = weights.sum()
            if total_weight > 1:
                weights /= total_weight

            # 5) Compute target dollar allocations
            portfolio_value = cash_avail + np.sum(portfolio_shares * latest_prices)
            target_alloc_dollars = weights * portfolio_value

            # 6) Rebalance by trading to match allocations
            for i, sym in enumerate(TICKERS):
                price = latest_prices[i]
                target_shares = int(target_alloc_dollars[i] // price)
                current_shares = portfolio[sym]["shares"]
                shares_to_trade = target_shares - current_shares
                if abs(shares_to_trade) < 1:
                    continue

                if shares_to_trade > 0 and cash_avail >= shares_to_trade * price:
                    # BUY
                    try:
                        api.submit_order(symbol=sym, qty=shares_to_trade, side="buy",
                                         type="market", time_in_force="gtc")
                        portfolio[sym]["shares"] += shares_to_trade
                        cash_avail -= shares_to_trade * price
                        print(f"  ✅ BUY  {shares_to_trade} {sym} @ {price:.2f}")
                    except APIError as e:
                        print(f"  ❌ BUY failed for {sym}: {e}")
                elif shares_to_trade < 0 and current_shares >= abs(shares_to_trade):
                    # SELL
                    try:
                        api.submit_order(symbol=sym, qty=abs(shares_to_trade), side="sell",
                                         type="market", time_in_force="gtc")
                        portfolio[sym]["shares"] -= abs(shares_to_trade)
                        cash_avail += abs(shares_to_trade) * price
                        print(f"  ✅ SELL {abs(shares_to_trade)} {sym} @ {price:.2f}")
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

