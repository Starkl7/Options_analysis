"""
Phase 4.2 — Strategy 2: Net Gamma Exposure (GEX)

Per ticker, per 15-min bar:
  - GEX per contract = OI × BS_gamma × 100 × S² × 0.01
    (calls positive, puts negative — dealer-convention)
  - Net GEX = Σ(call GEX) - Σ(put GEX)
  - Normalise: GEX_z = (GEX - rolling_20d_mean) / rolling_20d_std
  - Regime:
      GEX_z < -1 → momentum
      GEX_z > +1 → mean-reversion
      else        → neutral (no trade)

OI is refreshed once at market open (9:30 bar); held constant intraday.
Gamma updates every 15 min from live IV.

Output columns: timestamp, ticker, regime, gex_raw, gex_z
"""
import numpy as np
import pandas as pd

from config import (
    S2_GEX_WINDOW_DAYS, S2_REGIME_THRESHOLD,
    CONTRACT_MULTIPLIER, BARS_PER_DAY,
)
from pipeline.utils.greeks import gamma as bs_gamma_fn


_ROLL_BARS = S2_GEX_WINDOW_DAYS * BARS_PER_DAY


def _gex_per_contract(row, spot: float) -> float:
    """
    GEX = OI × gamma × multiplier × S² × 0.01
    Sign: +1 for calls, -1 for puts (dealer short convention).
    """
    try:
        oi    = float(row.get("oi", 0) or 0)
        iv    = float(row.get("iv", 0.2) or 0.2)
        K     = float(row["strike"])
        T     = float(row["tte"])
        r     = float(row.get("rf_rate", 0.04))
        q     = float(row.get("div_yield", 0.0))

        g = bs_gamma_fn(spot, K, T, r, q, iv)
        gex = oi * g * CONTRACT_MULTIPLIER * (spot**2) * 0.01

        return gex if row["type"] == "c" else -gex
    except Exception:
        return 0.0


def compute_s2_signals(
    iv_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    external_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Strategy 2 (Net GEX) signals for all tickers.

    Returns one row per (ticker, timestamp) with gex_raw, gex_z, regime.
    """
    results = []

    external_df = external_df.copy()
    external_df["date"] = pd.to_datetime(external_df["date"]).dt.date
    stock_df = stock_df.sort_values(["ticker", "timestamp"])
    iv_df["report_time"] = pd.to_datetime(iv_df["report_time"])

    for ticker, grp in iv_df.groupby("ticker"):
        sig = _compute_ticker_s2(ticker, grp, stock_df, external_df)
        if not sig.empty:
            results.append(sig)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True).sort_values(["ticker", "timestamp"])


def _compute_ticker_s2(
    ticker: str,
    iv_grp: pd.DataFrame,
    stock_df: pd.DataFrame,
    external_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute GEX time-series for a single ticker."""
    iv_grp = iv_grp.sort_values("report_time").copy()
    stock_t = stock_df[stock_df["ticker"] == ticker].sort_values("timestamp")

    records = []
    # OI cache: refreshed at each 9:30 bar, held constant intraday
    oi_cache = {}

    timestamps = sorted(iv_grp["report_time"].unique())

    for ts in timestamps:
        ts = pd.Timestamp(ts)
        bar = iv_grp[iv_grp["report_time"] == ts].copy()
        date_key = ts.date()

        # Spot price
        idx = stock_t["timestamp"].searchsorted(ts, side="right") - 1
        if idx < 0:
            continue
        spot = float(stock_t.iloc[idx]["close"])

        # r, q
        ext_row = external_df[external_df["date"] == date_key]
        r = float(ext_row["rf_rate"].iloc[0]) if not ext_row.empty else 0.04
        q_col = f"div_yield_{ticker}"
        q = float(ext_row[q_col].iloc[0]) if not ext_row.empty and q_col in ext_row.columns else 0.0

        bar["rf_rate"]   = r
        bar["div_yield"] = q

        # OI refresh: use market-open OI snapshot, hold constant intraday
        is_open_bar = (ts.hour == 9 and ts.minute == 30)

        if is_open_bar or not oi_cache:
            # Update OI cache from current bar
            for _, row in bar.iterrows():
                key = (float(row["strike"]), str(row.get("expiry_date", row["tte"])), row["type"])
                oi_cache[key] = float(row.get("oi", 0) or 0)

        # Apply cached OI to current bar (intraday OI doesn't change)
        def get_oi(row):
            key = (float(row["strike"]), str(row.get("expiry_date", row["tte"])), row["type"])
            return oi_cache.get(key, float(row.get("oi", 0) or 0))

        bar["oi_frozen"] = bar.apply(get_oi, axis=1)
        bar_oi = bar.copy()
        bar_oi["oi"] = bar_oi["oi_frozen"]

        # Compute per-contract GEX and sum
        bar_oi["gex"] = bar_oi.apply(lambda r_: _gex_per_contract(r_, spot), axis=1)

        net_gex = float(bar_oi["gex"].sum())

        records.append({"timestamp": ts, "ticker": ticker, "gex_raw": net_gex})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("timestamp")

    # ── Rolling normalisation ──────────────────────────────────────────────
    gex_series = df["gex_raw"]
    roll_mean = gex_series.shift(1).rolling(_ROLL_BARS, min_periods=_ROLL_BARS // 4).mean()
    roll_std  = gex_series.shift(1).rolling(_ROLL_BARS, min_periods=_ROLL_BARS // 4).std()

    df["gex_z"] = (gex_series - roll_mean) / (roll_std + 1e-8)

    # ── Regime classification ──────────────────────────────────────────────
    conditions = [
        df["gex_z"] < -S2_REGIME_THRESHOLD,
        df["gex_z"] > +S2_REGIME_THRESHOLD,
    ]
    choices = ["momentum", "mean_reversion"]
    df["regime"] = np.select(conditions, choices, default="neutral")

    return df[["timestamp", "ticker", "regime", "gex_raw", "gex_z"]]


if __name__ == "__main__":
    from config import DATA_DIR

    print("=== Phase 4.2: Strategy 2 — Net GEX Regime Signals ===")

    print("  Loading iv_data.parquet …")
    iv_df = pd.read_parquet(DATA_DIR / "iv_data.parquet")

    print("  Loading stock_bars.parquet …")
    stock_df = pd.read_parquet(DATA_DIR / "stock_bars.parquet")

    print("  Loading external_data.parquet …")
    external_df = pd.read_parquet(DATA_DIR / "external_data.parquet")

    signals = compute_s2_signals(iv_df, stock_df, external_df)

    if signals.empty:
        print("  [WARN] No S2 signals generated.")
    else:
        out_path = DATA_DIR / "signals_s2.parquet"
        signals.to_parquet(out_path, index=False)
        regime_counts = signals["regime"].value_counts()
        print(f"  Saved {len(signals):,} rows → {out_path}")
        for regime, count in regime_counts.items():
            print(f"    {regime:<18s}: {count:,} bars ({count/len(signals)*100:.1f}%)")
