"""
Phase 4.1 — Strategy 1: Straddle IV Reversion

Per ticker, per 15-min bar:
  - ATM straddle IV = (IV_call + IV_put) / 2 at nearest ATM strike, 14-30 day expiry
  - Rolling 10-bar z-score of straddle IV
  - Signal: z > +1.5 → sell straddle (short vol); z < -1.5 → buy straddle (long vol)
  - Gate: bid-ask spread above rolling median confirms MM inventory pressure

Output columns: timestamp, ticker, direction (+1/-1/0), z_score,
                straddle_iv, spread_flag, atm_strike, expiry_date
"""
import numpy as np
import pandas as pd

from config import S1_ROLLING_WINDOW, S1_ZSCORE_ENTRY, S1_MIN_TTE_DAYS, S1_MAX_TTE_DAYS


def compute_s1_signals(
    greeks_df: pd.DataFrame,
    iv_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Strategy 1 signals for all tickers.

    Parameters
    ----------
    greeks_df : intraday_greeks — has ticker, timestamp, strike, tte, type, iv
    iv_df     : iv_data with bid/ask (for spread check)

    Returns
    -------
    DataFrame with one row per (ticker, timestamp) that has a signal.
    """
    results = []

    for ticker, grp in greeks_df.groupby("ticker"):
        sig = _compute_ticker_s1(ticker, grp, iv_df)
        if not sig.empty:
            results.append(sig)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True).sort_values(["ticker", "timestamp"])


def _compute_ticker_s1(
    ticker: str,
    greeks: pd.DataFrame,
    iv_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute S1 signal time-series for a single ticker."""
    greeks = greeks.sort_values("timestamp").copy()

    tte_min = S1_MIN_TTE_DAYS / 365.25
    tte_max = S1_MAX_TTE_DAYS / 365.25

    # Filter to qualifying expiry window
    qual = greeks[(greeks["tte"] >= tte_min) & (greeks["tte"] <= tte_max)].copy()
    if qual.empty:
        return pd.DataFrame()

    records = []

    for ts, bar in qual.groupby("timestamp"):
        # Get spot for this bar
        spot = float(bar["spot"].iloc[0]) if "spot" in bar.columns else None
        if spot is None:
            continue

        # Find nearest expiry slice
        tte_per_expiry = bar.groupby("tte")["tte"].first().sort_values()
        if tte_per_expiry.empty:
            continue
        tte_nearest = float(tte_per_expiry.iloc[0])

        slice_df = bar[bar["tte"] == tte_nearest]

        # ATM strike: argmin |K - S|
        slice_df = slice_df.copy()
        slice_df["abs_moneyness"] = (slice_df["strike"] - spot).abs()
        atm_strike = float(slice_df.loc[slice_df["abs_moneyness"].idxmin(), "strike"])

        call_iv = slice_df[(slice_df["strike"] == atm_strike) & (slice_df["type"] == "c")]["iv"]
        put_iv  = slice_df[(slice_df["strike"] == atm_strike) & (slice_df["type"] == "p")]["iv"]

        if call_iv.empty or put_iv.empty:
            continue

        straddle_iv = (float(call_iv.iloc[0]) + float(put_iv.iloc[0])) / 2.0

        # Straddle bid-ask spread
        iv_bar = iv_df[
            (iv_df["ticker"] == ticker) &
            (iv_df["report_time"] == ts) &
            (iv_df["strike"] == atm_strike)
        ] if iv_df is not None and not iv_df.empty else pd.DataFrame()

        straddle_spread = None
        if not iv_bar.empty:
            call_row = iv_bar[iv_bar["type"] == "c"]
            put_row  = iv_bar[iv_bar["type"] == "p"]
            if not call_row.empty and not put_row.empty:
                c_ask = float(call_row["ask"].iloc[0])
                c_bid = float(call_row["bid"].iloc[0])
                p_ask = float(put_row["ask"].iloc[0])
                p_bid = float(put_row["bid"].iloc[0])
                straddle_spread = (c_ask + p_ask) - (c_bid + p_bid)

        expiry_col = "expiry_date" if "expiry_date" in slice_df.columns else None
        expiry = slice_df[expiry_col].iloc[0] if expiry_col else None

        records.append({
            "timestamp":    ts,
            "ticker":       ticker,
            "atm_strike":   atm_strike,
            "tte":          tte_nearest,
            "expiry_date":  expiry,
            "straddle_iv":  straddle_iv,
            "straddle_spread": straddle_spread,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("timestamp")

    # ── Rolling statistics ─────────────────────────────────────────────────
    # Use .shift(1) to ensure we never use current-bar data in the window
    iv_series = df["straddle_iv"]
    roll_mean = iv_series.shift(1).rolling(S1_ROLLING_WINDOW, min_periods=S1_ROLLING_WINDOW // 2).mean()
    roll_std  = iv_series.shift(1).rolling(S1_ROLLING_WINDOW, min_periods=S1_ROLLING_WINDOW // 2).std()

    # Floor rolling std at 0.3 % IV absolute to prevent extreme z-scores when IV
    # is near-constant (roll_std → 0 gives z = ΔIV / 1e-8 → ±thousands).
    roll_std  = roll_std.clip(lower=3e-3)

    df["rolling_mean_iv"] = roll_mean
    df["rolling_std_iv"]  = roll_std
    df["z_score"] = (df["straddle_iv"] - roll_mean) / roll_std

    # ── Spread gate ────────────────────────────────────────────────────────
    # spread_flag = True when current spread > rolling median (wider than usual).
    # Short straddle: wide spread confirms MM inventory pressure (elevated IV).
    # Long  straddle: narrow spread (NOT wide) confirms genuinely compressed IV
    #                 and gives a cheaper entry — use ~spread_flag.
    if "straddle_spread" in df.columns and df["straddle_spread"].notna().any():
        spread_series = df["straddle_spread"].ffill()
        spread_median = spread_series.shift(1).rolling(S1_ROLLING_WINDOW, min_periods=3).median()
        df["spread_flag"] = df["straddle_spread"] > spread_median
    else:
        df["spread_flag"] = True   # no spread data → gate always open

    # ── Signal generation ─────────────────────────────────────────────────
    # Short: IV spike (z > +threshold) AND wide spread (confirms elevated premium)
    # Long : IV compression (z < −threshold) AND narrow spread (confirms low IV,
    #        cheaper to enter; use ~spread_flag)
    sell_signal = (df["z_score"] >  S1_ZSCORE_ENTRY) &  df["spread_flag"]
    buy_signal  = (df["z_score"] < -S1_ZSCORE_ENTRY) & ~df["spread_flag"]

    df["direction"] = 0
    df.loc[sell_signal, "direction"] = -1   # -1 = sell straddle (short vol)
    df.loc[buy_signal,  "direction"] = +1   # +1 = buy straddle (long vol)

    # Only return rows with a signal or needed for context
    return df[["timestamp", "ticker", "direction", "z_score",
               "straddle_iv", "spread_flag", "atm_strike",
               "tte", "expiry_date", "rolling_mean_iv", "rolling_std_iv"]]


if __name__ == "__main__":
    from config import DATA_DIR

    print("=== Phase 4.1: Strategy 1 — Straddle IV Reversion Signals ===")

    print("  Loading intraday_greeks.parquet …")
    greeks_df = pd.read_parquet(DATA_DIR / "intraday_greeks.parquet")

    print("  Loading iv_data.parquet …")
    iv_df = pd.read_parquet(DATA_DIR / "iv_data.parquet")

    signals = compute_s1_signals(greeks_df, iv_df)

    if signals.empty:
        print("  [WARN] No S1 signals generated.")
    else:
        out_path = DATA_DIR / "signals_s1.parquet"
        signals.to_parquet(out_path, index=False)
        n_fired = int((signals["direction"] != 0).sum())
        print(f"  Saved {len(signals):,} rows → {out_path}")
        print(f"  Signals fired  : {n_fired} ({n_fired/len(signals)*100:.1f}%)")
        print(f"  Sell straddle  : {int((signals['direction'] == -1).sum())}")
        print(f"  Buy straddle   : {int((signals['direction'] == +1).sum())}")
