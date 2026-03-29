"""
Phase 4.3 — Strategy 4: PCR Opening Signal

Opening window only (bars 1–2: 9:30–10:00):
  - Near-ATM volume: strikes within ±5% of spot, nearest 2 expiries
  - PCR = put_volume / call_volume over bars 1-2
  - Rolling 20-day mean/std of opening PCR
  - Signal: PCR_z > +1.5 → bearish; PCR_z < -1.5 → bullish
  - GEX corroboration: only fire if S2 regime aligns with direction

Output columns: date, ticker, direction (+1/-1/0), pcr_z,
                pcr_raw, gex_corroboration
"""
import numpy as np
import pandas as pd

from config import (
    S4_OPENING_BARS, S4_MONEYNESS_BAND,
    S4_NEAREST_EXPIRIES, S4_PCR_WINDOW_DAYS,
    S4_ZSCORE_THRESHOLD,
)


def compute_s4_signals(
    iv_df: pd.DataFrame,
    s2_signals: pd.DataFrame,
    stock_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Strategy 4 (PCR Opening) signals for all tickers.

    Parameters
    ----------
    iv_df      : option chain with volume, bid, ask, strike, tte, type, report_time
    s2_signals : output of signal_s2 — used for GEX corroboration check
    stock_df   : stock bars for spot lookup

    Returns
    -------
    One row per (ticker, date) with a signal.
    """
    results = []

    iv_df    = iv_df.copy()
    iv_df["report_time"] = pd.to_datetime(iv_df["report_time"])
    stock_df = stock_df.sort_values(["ticker", "timestamp"])

    for ticker, grp in iv_df.groupby("ticker"):
        sig = _compute_ticker_s4(ticker, grp, s2_signals, stock_df)
        if not sig.empty:
            results.append(sig)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True).sort_values(["ticker", "date"])


def _compute_ticker_s4(
    ticker: str,
    iv_grp: pd.DataFrame,
    s2_signals: pd.DataFrame,
    stock_df: pd.DataFrame,
) -> pd.DataFrame:
    stock_t = stock_df[stock_df["ticker"] == ticker].sort_values("timestamp")
    s2_t    = s2_signals[s2_signals["ticker"] == ticker] if not s2_signals.empty else pd.DataFrame()

    # Group by date, keep only opening bars (first S4_OPENING_BARS per day)
    iv_grp = iv_grp.sort_values("report_time").copy()
    iv_grp["date"] = iv_grp["report_time"].dt.date

    daily_records = []

    for date_key, day_grp in iv_grp.groupby("date"):
        day_bars = day_grp.sort_values("report_time")

        # Opening bars: first S4_OPENING_BARS (9:30 and 9:45)
        open_bars = day_bars.head(S4_OPENING_BARS)

        # Spot price at first bar
        open_ts = pd.Timestamp(open_bars["report_time"].iloc[0])
        idx = stock_t["timestamp"].searchsorted(open_ts, side="right") - 1
        if idx < 0:
            continue
        spot = float(stock_t.iloc[idx]["close"])

        # Near-ATM filter: ±5% of spot
        mono_lo = spot * (1 - S4_MONEYNESS_BAND)
        mono_hi = spot * (1 + S4_MONEYNESS_BAND)
        near_atm = open_bars[
            (open_bars["strike"] >= mono_lo) &
            (open_bars["strike"] <= mono_hi)
        ].copy()

        # Nearest S4_NEAREST_EXPIRIES expiries
        if near_atm.empty:
            continue

        expiry_col = "expiry_date" if "expiry_date" in near_atm.columns else "tte"
        sorted_expiries = sorted(near_atm[expiry_col].unique())[:S4_NEAREST_EXPIRIES]
        near_atm = near_atm[near_atm[expiry_col].isin(sorted_expiries)]

        # PCR = sum(put volume) / sum(call volume) over opening bars
        call_vol = near_atm[near_atm["type"] == "c"]["volume"].fillna(0).sum()
        put_vol  = near_atm[near_atm["type"] == "p"]["volume"].fillna(0).sum()

        if call_vol <= 0:
            continue

        pcr = put_vol / call_vol

        daily_records.append({"date": date_key, "ticker": ticker, "pcr_raw": pcr})

    if not daily_records:
        return pd.DataFrame()

    df = pd.DataFrame(daily_records).sort_values("date")

    # ── Rolling z-score ────────────────────────────────────────────────────
    pcr_series = df["pcr_raw"]
    roll_mean  = pcr_series.shift(1).rolling(S4_PCR_WINDOW_DAYS, min_periods=S4_PCR_WINDOW_DAYS // 2).mean()
    roll_std   = pcr_series.shift(1).rolling(S4_PCR_WINDOW_DAYS, min_periods=S4_PCR_WINDOW_DAYS // 2).std()

    # Guard against near-zero std (insufficient PCR variation → no signal)
    # Threshold: at least 5% PCR std before a z-score is meaningful
    MIN_PCR_STD = 0.05
    valid_std = roll_std >= MIN_PCR_STD
    df["pcr_z"] = np.where(
        valid_std & roll_mean.notna(),
        (pcr_series - roll_mean) / roll_std,
        0.0,   # zero z-score when std is too small → no signal fires
    )

    # ── Signal direction ───────────────────────────────────────────────────
    df["direction"] = 0
    df.loc[df["pcr_z"] > +S4_ZSCORE_THRESHOLD, "direction"] = -1   # bearish
    df.loc[df["pcr_z"] < -S4_ZSCORE_THRESHOLD, "direction"] = +1   # bullish

    # ── GEX corroboration ─────────────────────────────────────────────────
    # Check S2 regime at 9:30 bar; only keep signal if S2 regime corroborates
    df["gex_corroboration"] = False

    if not s2_t.empty:
        s2_t = s2_t.copy()
        s2_t["date"] = pd.to_datetime(s2_t["timestamp"]).dt.date

        # Use first bar of day (9:30) regime
        s2_open = s2_t.sort_values("timestamp").groupby("date").first().reset_index()
        s2_open = s2_open[["date", "regime"]]

        df = df.merge(s2_open, on="date", how="left")

        # Corroboration logic:
        # bullish (+1) corroborated by momentum regime (GEX negative → price likely moves)
        # bearish (-1) corroborated by momentum regime
        # mean-reversion regime corroborates neither directional bet
        df["gex_corroboration"] = (
            ((df["direction"] == +1) & (df["regime"] == "momentum")) |
            ((df["direction"] == -1) & (df["regime"] == "momentum"))
        )
        # Only keep signals where GEX corroborates
        df.loc[df["direction"] != 0, "direction"] = df.loc[df["direction"] != 0].apply(
            lambda row: row["direction"] if row["gex_corroboration"] else 0, axis=1
        )

    keep_cols = ["date", "ticker", "direction", "pcr_z", "pcr_raw", "gex_corroboration"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


if __name__ == "__main__":
    from config import DATA_DIR

    print("=== Phase 4.3: Strategy 4 — PCR Opening Signals ===")

    print("  Loading iv_data.parquet …")
    iv_df = pd.read_parquet(DATA_DIR / "iv_data.parquet")

    print("  Loading signals_s2.parquet …")
    s2_df = pd.read_parquet(DATA_DIR / "signals_s2.parquet")

    print("  Loading stock_bars.parquet …")
    stock_df = pd.read_parquet(DATA_DIR / "stock_bars.parquet")

    signals = compute_s4_signals(iv_df, s2_df, stock_df)

    if signals.empty:
        print("  [WARN] No S4 signals generated.")
    else:
        out_path = DATA_DIR / "signals_s4.parquet"
        signals.to_parquet(out_path, index=False)
        n_fired = int((signals["direction"] != 0).sum())
        print(f"  Saved {len(signals):,} rows → {out_path}")
        print(f"  Signals fired  : {n_fired} ({n_fired/len(signals)*100:.1f}% of trading days)")
        print(f"  Bullish (+1)   : {int((signals['direction'] == +1).sum())}")
        print(f"  Bearish (-1)   : {int((signals['direction'] == -1).sum())}")
        corr = int(signals['gex_corroboration'].sum()) if 'gex_corroboration' in signals.columns else 0
        print(f"  GEX corroboration: {corr} days")
