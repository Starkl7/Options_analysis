"""
Phase 1.3 — In-House IV Computation

Vectorised IV solver using Jäckel's Let's Be Rational algorithm.

Main entry point
----------------
    from pipeline.cleaning_01.compute_iv import compute_iv_for_snapshot

    df_with_iv = compute_iv_for_snapshot(filtered_df, spot, r, q)
"""
import warnings

import numpy as np
import pandas as pd
from py_vollib_vectorized import vectorized_implied_volatility as _viv
from config import IV_MIN, IV_MAX


def compute_iv_for_snapshot(
    df: pd.DataFrame,
    spot: float,
    r: float,
    q: float,
) -> pd.DataFrame:
    """
    Compute implied volatility for all contracts in a single snapshot using
    Jäckel's 'Let's Be Rational' algorithm (py_vollib_vectorized).

    Replaces row-by-row Brent's method with a single vectorised call —
    same result, ~10-100× faster.

    Expects columns: mid, strike, tte, type  ('c' or 'p')
    Adds column:     iv

    Drops contracts where IV solver fails (returns 0.0) or IV falls outside
    [IV_MIN, IV_MAX].

    Parameters
    ----------
    df   : filtered option chain snapshot (output of filter_snapshot)
    spot : underlying spot price
    r    : risk-free rate (continuous)
    q    : dividend yield (continuous)

    Returns
    -------
    DataFrame with 'iv' column; rows with invalid IV removed.
    """
    df = df.copy()

    if df.empty:
        df["iv"] = pd.Series(dtype=float)
        return df

    # Single vectorised IV solve across the entire snapshot
    ivs = _viv(
        price    = df["mid"].to_numpy(dtype=float),
        S        = float(spot),
        K        = df["strike"].to_numpy(dtype=float),
        t        = df["tte"].to_numpy(dtype=float),
        r        = float(r),
        flag     = df["type"].to_numpy(dtype=str),
        q        = float(q),
        on_error = "ignore",           # returns 0.0 on failure instead of raising
        model    = "black_scholes_merton",
        return_as = "numpy",
    )
    df["iv"] = ivs

    # Post-solve bounds filter
    n_before = len(df)
    df = df[df["iv"].between(IV_MIN, IV_MAX, inclusive="both")].copy()
    n_dropped_bounds = n_before - len(df)

    if n_dropped_bounds > 0:
        pct = n_dropped_bounds / n_before * 100 if n_before else 0
        warnings.warn(
            f"IV bounds filter dropped {n_dropped_bounds} contracts ({pct:.1f}%)",
            stacklevel=2,
        )

    return df



def compute_iv_batch(
    options_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    external_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process a multi-day, multi-ticker options DataFrame.

    Parameters
    ----------
    options_df  : filtered + mid-priced option chains with columns:
                  ticker, report_time, strike, tte, type, mid, bid, ask, oi, volume
    stock_df    : from build_stock_data — has ticker, timestamp, close
    external_df : from fetch_external   — has date, rf_rate, div_yield_{TICKER}

    Returns
    -------
    options_df with added column: iv
    """
    from pipeline.cleaning_01.filter_contracts import filter_snapshot

    results = []

    external_df = external_df.copy()
    external_df["date"] = pd.to_datetime(external_df["date"]).dt.date

    # Align stock bar to option report_time using merge_asof
    stock_df   = stock_df.sort_values(["ticker", "timestamp"])
    options_df = options_df.sort_values(["ticker", "report_time"])

    for (ticker, ts), grp in options_df.groupby(["ticker", "report_time"]):
        # Look up spot price
        stock_ticker = stock_df[stock_df["ticker"] == ticker]
        if stock_ticker.empty:
            continue

        idx = stock_ticker["timestamp"].searchsorted(ts, side="right") - 1
        if idx < 0:
            continue
        spot = float(stock_ticker.iloc[idx]["close"])

        # Look up r, q
        date_key = pd.Timestamp(ts).date()
        ext_row  = external_df[external_df["date"] == date_key]
        r = float(ext_row["rf_rate"].iloc[0]) if not ext_row.empty else 0.04
        q_col = f"div_yield_{ticker}"
        q = float(ext_row[q_col].iloc[0]) if not ext_row.empty and q_col in ext_row.columns else 0.0

        # Filter then compute IV
        filtered = filter_snapshot(grp, spot)
        if filtered.empty:
            continue

        iv_df = compute_iv_for_snapshot(filtered, spot, r, q)
        if iv_df.empty:
            continue

        iv_df["spot"]    = spot
        iv_df["rf_rate"] = r
        iv_df["div_yield"] = q
        results.append(iv_df)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    from config import TICKERS, DATA_DIR, BACKTEST_START, BACKTEST_END
    from pipeline.utils.data_loader import load_all_options

    print("=== Phase 1.3: Computing Implied Volatilities ===")

    print("  Loading raw option chain snapshots from drive …")
    options_df = load_all_options(BACKTEST_START, BACKTEST_END, TICKERS)
    if options_df.empty:
        print("  [ERROR] No option chain data found. Run the drive audit first.")
        raise SystemExit(1)
    print(f"  Loaded {len(options_df):,} raw option rows across "
          f"{options_df['ticker'].nunique()} tickers")

    print("  Loading stock_bars.parquet …")
    stock_df = pd.read_parquet(DATA_DIR / "stock_bars.parquet")

    print("  Loading external_data.parquet …")
    external_df = pd.read_parquet(DATA_DIR / "external_data.parquet")

    print("  Running filter + IV solver + put-call parity check …")
    iv_df = compute_iv_batch(options_df, stock_df, external_df)

    if iv_df.empty:
        print("  [WARN] No IV data produced — check that stock_bars and "
              "external_data timestamps overlap the option chain dates.")
        raise SystemExit(1)

    output_path = DATA_DIR / "iv_data.parquet"
    iv_df.to_parquet(output_path, index=False)
    print(f"\n  Saved {len(iv_df):,} rows → {output_path}")

    # Diagnostics
    n_snap  = iv_df["report_time"].nunique() if "report_time" in iv_df.columns else "?"
    n_tick  = iv_df["ticker"].nunique()
    iv_lo   = iv_df["iv"].min()
    iv_hi   = iv_df["iv"].max()
    print(f"  Tickers   : {n_tick}")
    print(f"  Snapshots : {n_snap}")
    print(f"  IV range  : {iv_lo:.3f} – {iv_hi:.3f}")
