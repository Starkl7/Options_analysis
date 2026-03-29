"""
Phase 3.1-3.2 — Heston Calibration Pipeline

Runs daily Heston calibrations for all tickers using the 9:30 snapshot.
Two-stage optimizer: differential_evolution → Levenberg-Marquardt.

Outputs: data/heston_params.parquet
  columns: date, ticker, kappa, theta, xi, rho, v0,
           ivrmse, converged, n_contracts, feller_ok

Run:
    python -m pipeline.02_heston.run_calibration
"""
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import numpy as np

from config import (
    TICKERS, DATA_DIR,
    BACKTEST_START, BACKTEST_END,
    HESTON_IVRMSE_THRESHOLD,
)
from pipeline.heston_02.calibration import (
    select_calibration_contracts,
    calibrate_heston,
)

warnings.filterwarnings("ignore")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Market-open timestamp suffix (9:30)
OPEN_SUFFIX = "0930"
PARAM_JUMP_THRESHOLD = 0.50   # flag if any param jumps > 50% day-over-day


# ── Load pre-computed IV data ──────────────────────────────────────────────

def load_iv_data(path: Path = DATA_DIR / "iv_data.parquet") -> pd.DataFrame:
    """Load the cleaned + IV-computed option chain data."""
    if not path.exists():
        raise FileNotFoundError(
            f"IV data not found at {path}. "
            "Run pipeline.01_cleaning.compute_iv first."
        )
    return pd.read_parquet(path)


def load_external(path: Path = DATA_DIR / "external_data.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ── Stability check ────────────────────────────────────────────────────────

def check_param_stability(
    new_params: dict,
    prev_params: dict,
    threshold: float = PARAM_JUMP_THRESHOLD,
) -> list[str]:
    """
    Return list of parameter names that jumped > threshold fraction from prev.
    """
    flagged = []
    for param in ["kappa", "theta", "xi", "rho", "v0"]:
        prev = prev_params.get(param)
        curr = new_params.get(param)
        if prev is None or curr is None or abs(prev) < 1e-8:
            continue
        rel_jump = abs(curr - prev) / abs(prev)
        if rel_jump > threshold:
            flagged.append(f"{param}: {prev:.4f} → {curr:.4f} ({rel_jump*100:.1f}%)")
    return flagged


# ── Single-day single-ticker calibration ─────────────────────────────────

def calibrate_one(
    iv_df: pd.DataFrame,
    ticker: str,
    date: str,
    r: float,
    q: float,
    prev_params: dict | None = None,
    n_retry: int = 3,
) -> dict:
    """
    Calibrate Heston for one (ticker, date) using the 9:30 snapshot.

    Retries up to n_retry times with different random initialisations
    if parameters jump suspiciously.
    """
    # Select 9:30 snapshot (or nearest open bar)
    df = iv_df[
        (iv_df["ticker"] == ticker) &
        (pd.to_datetime(iv_df["report_time"]).dt.date == pd.Timestamp(date).date())
    ].copy()

    if df.empty:
        return {"ticker": ticker, "date": date, "converged": False, "error": "no data"}

    # Take earliest bar of the day as market-open snapshot
    df = df.sort_values("report_time")
    open_ts = df["report_time"].iloc[0]
    df = df[df["report_time"] == open_ts]

    spot = float(df["spot"].iloc[0])

    # Select calibration universe
    contracts = select_calibration_contracts(df, spot)
    if len(contracts) < 8:
        return {"ticker": ticker, "date": date, "converged": False,
                "error": f"only {len(contracts)} contracts after filtering"}

    best_result = None

    for attempt in range(n_retry):
        # On retry, don't use prev_params as warm-start (force fresh search)
        warm = prev_params if attempt == 0 else None

        result = calibrate_heston(contracts, S=spot, r=r, q=q, prev_params=warm)
        result["ticker"] = ticker
        result["date"]   = date

        if not result.get("converged"):
            best_result = result
            continue

        # Check stability if we have prev params
        if prev_params is not None:
            jumps = check_param_stability(result, prev_params)
            if jumps:
                warnings.warn(
                    f"[{ticker} {date}] Param jumps detected (attempt {attempt+1}): {jumps}"
                )
                if attempt < n_retry - 1:
                    continue   # retry

        best_result = result
        break

    if best_result is None:
        best_result = {"ticker": ticker, "date": date, "converged": False}

    return best_result


# ── Full pipeline run ─────────────────────────────────────────────────────

def run_calibration(
    start: str = BACKTEST_START,
    end: str   = BACKTEST_END,
    tickers: list[str] = TICKERS,
    output_path: Path = DATA_DIR / "heston_params.parquet",
    iv_data_path: Path = DATA_DIR / "iv_data.parquet",
    n_threads: int = 4,
) -> pd.DataFrame:
    """
    Run Heston calibration for all tickers and dates.

    Calibrates once per trading day at market open.
    Uses prior-day parameters as warm-start.
    Tickers within each date are calibrated in parallel (ThreadPoolExecutor).
    Saves incremental checkpoints every 5 dates.
    """
    print("=== Phase 3: Heston Calibration ===")

    iv_df    = load_iv_data(iv_data_path)
    external = load_external()

    iv_df["report_time"] = pd.to_datetime(iv_df["report_time"])
    iv_df["date_key"]    = iv_df["report_time"].dt.date

    # Pre-split DataFrame by ticker to avoid full-df copies in threads
    ticker_dfs = {t: iv_df[iv_df["ticker"] == t].copy() for t in tickers}

    # Get sorted unique trading dates
    all_dates = sorted(iv_df["date_key"].unique())
    all_dates = [d for d in all_dates
                 if pd.Timestamp(start).date() <= d <= pd.Timestamp(end).date()]

    print(f"  Calibrating {len(tickers)} tickers × {len(all_dates)} dates  "
          f"[{n_threads} threads]")

    records     = []
    prev_params = {t: None for t in tickers}   # warm-start store

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        for i, date in enumerate(all_dates):
            date_str = str(date)

            # Fetch r for this date
            ext_row = external[external["date"] == date]
            r = float(ext_row["rf_rate"].iloc[0]) if not ext_row.empty else 0.04

            # Submit all tickers for this date in parallel
            future_to_ticker = {}
            for ticker in tickers:
                q_col = f"div_yield_{ticker}"
                q_val = (float(ext_row[q_col].iloc[0])
                         if not ext_row.empty and q_col in ext_row.columns
                         else 0.0)
                fut = pool.submit(
                    calibrate_one,
                    iv_df       = ticker_dfs[ticker],
                    ticker      = ticker,
                    date        = date_str,
                    r           = r,
                    q           = q_val,
                    prev_params = prev_params[ticker],
                )
                future_to_ticker[fut] = ticker

            # Collect results (preserves warm-start ordering per ticker)
            date_results = {}
            for fut in as_completed(future_to_ticker):
                t      = future_to_ticker[fut]
                result = fut.result()
                date_results[t] = result

            for ticker in tickers:
                result = date_results[ticker]
                if result.get("converged"):
                    prev_params[ticker] = {
                        k: result[k] for k in ["kappa", "theta", "xi", "rho", "v0"]
                    }
                records.append(result)

            # Progress every 5 days
            if (i + 1) % 5 == 0:
                n_conv = sum(1 for rec in records if rec.get("converged"))
                print(f"  [{i+1}/{len(all_dates)}] {date_str}  |  "
                      f"converged so far: {n_conv}/{len(records)}")
                # Incremental save so a crash doesn't lose everything
                _checkpoint = pd.DataFrame(records)
                _checkpoint.to_parquet(str(output_path) + ".tmp", index=False)

    df = pd.DataFrame(records)

    # Summary stats
    if not df.empty and "converged" in df.columns:
        n_total = len(df)
        n_conv  = df["converged"].sum()
        n_high_err = (df["ivrmse"] > HESTON_IVRMSE_THRESHOLD).sum() if "ivrmse" in df.columns else 0
        print(f"\n  Results: {n_conv}/{n_total} converged "
              f"({n_conv/n_total*100:.1f}%)")
        print(f"  High-error (IVRMSE > {HESTON_IVRMSE_THRESHOLD*100:.0f}bps): {n_high_err}")
        if "ivrmse" in df.columns:
            print(f"  Median IVRMSE: {df['ivrmse'].median()*100:.1f}bps")

    df.to_parquet(output_path, index=False)
    # Remove temp checkpoint file if present
    _tmp = Path(str(output_path) + ".tmp")
    if _tmp.exists():
        _tmp.unlink()
    print(f"\n  Saved {len(df)} rows → {output_path}")

    return df


if __name__ == "__main__":
    run_calibration()
