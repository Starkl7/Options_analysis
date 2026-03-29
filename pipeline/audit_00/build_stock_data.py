"""
Phase 0.2 — Underlying Price Data

Loads all 15-min stock bar parquets, restricts to market hours,
aligns to option-chain timestamps, computes log returns and
20-day rolling realised vol.

Output: data/stock_bars.parquet
  columns: ticker, timestamp, open, high, low, close, volume,
           log_return, realized_vol_20d

Run:
    python -m pipeline.00_audit.build_stock_data
"""
import numpy as np
import pandas as pd

from config import (
    TICKERS, DATA_DIR,
    BACKTEST_START, BACKTEST_END,
    BARS_PER_DAY,
)
from pipeline.utils.data_loader import load_all_stock

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Rolling vol window: 20 trading days × 26 bars/day (theoretical full-coverage target).
# In practice yfinance 15-min data silently truncates to the last ~10 trading days for
# longer date ranges, so the full 520-bar window will never fill.  Use min_periods=2 so
# the rolling std grows from whatever data is available rather than staying all-NaN.
_ROLL_BARS  = 20 * BARS_PER_DAY   # 520 — used as the look-back once data is abundant
_MIN_BARS   = 2                    # absolute minimum to compute a meaningful std


def build_stock_bars(
    tickers: list[str] = TICKERS,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    output_path=DATA_DIR / "stock_bars.parquet",
) -> pd.DataFrame:
    """
    Load, clean, align, and enrich stock bar data for all tickers.
    Returns and saves the unified DataFrame.
    """
    print("=== Phase 0.2: Building stock bars ===")

    df = load_all_stock(start, end, tickers)

    if df.empty:
        print("  [WARN] No stock data found. Returning empty DataFrame.")
        return df

    print(f"  Loaded {len(df):,} raw bar rows across {df['ticker'].nunique()} tickers")

    # Ensure timestamp is tz-naive (drop tz info for consistent merges)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    # Sort
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    # ── Log returns ────────────────────────────────────────────────────────
    # Compute within each ticker; first bar of each day has no prior bar —
    # leave as NaN (correct behaviour, not a bug).
    df["log_return"] = (
        df.groupby("ticker")["close"]
        .transform(lambda s: np.log(s / s.shift(1)))
    )

    # ── Rolling 20-day realised vol ────────────────────────────────────────
    # Annualise: σ_annual = σ_bar × √(252 × 26)
    ann_factor = np.sqrt(252 * BARS_PER_DAY)

    df["realized_vol_20d"] = (
        df.groupby("ticker")["log_return"]
        .transform(lambda s: s.rolling(_ROLL_BARS, min_periods=_MIN_BARS).std() * ann_factor)
    )

    # ── Select & reorder columns ───────────────────────────────────────────
    keep_cols = ["ticker", "timestamp", "open", "high", "low", "close", "volume",
                 "log_return", "realized_vol_20d"]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available]

    df.to_parquet(output_path, index=False)
    print(f"  Saved {len(df):,} rows → {output_path}")

    # Quick diagnostics
    print("\n  Coverage per ticker:")
    for ticker, grp in df.groupby("ticker"):
        n_days = grp["timestamp"].dt.date.nunique()
        n_bars = len(grp)
        nan_rv = grp["realized_vol_20d"].isna().sum()
        print(f"    {ticker:6s}: {n_days} days, {n_bars:5d} bars, "
              f"realised vol NaN: {nan_rv} ({nan_rv/n_bars*100:.1f}%)")

    return df


if __name__ == "__main__":
    build_stock_bars()
