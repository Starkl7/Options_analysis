"""
Phase 0.1 — Data Inventory & Audit

Scans the Seagate drive and produces data/audit_report.json containing:
  - snapshot completeness per (date, ticker)
  - field presence check
  - volume semantics flag (interval vs cumulative)
  - timestamp alignment between options and stock bars
  - missing data patterns (heatmap-ready matrix)

Run:
    python -m pipeline.00_audit.audit_data
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    SEAGATE_ROOT, TICKERS, DATA_DIR,
    BACKTEST_START, BACKTEST_END,
    BARS_PER_DAY,
)
from pipeline.utils.data_loader import (
    list_available_dates,
    list_snapshots_for_date,
    list_stock_snapshots_for_date,
    load_options_snapshot,
    load_stock_snapshot,
)

warnings.filterwarnings("ignore")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Expected columns in raw option parquets
EXPECTED_OPTION_COLS = {
    "bid", "ask", "strike", "volume", "type",
    "contract_symbol", "report_time",
}
# 3 tickers used for volume semantics check
VOLUME_CHECK_TICKERS = TICKERS[:3]


# ── 1. Snapshot completeness ───────────────────────────────────────────────

def build_completeness_matrix(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Returns a (date × ticker) DataFrame where each cell is the number of
    option chain snapshots found that day.  BARS_PER_DAY is the expected max.
    """
    dates = [d for d in list_available_dates() if start <= d <= end]
    records = []
    for date in dates:
        row = {"date": date}
        for ticker in tickers:
            row[ticker] = len(list_snapshots_for_date(date, ticker))
        records.append(row)

    if not records:
        return pd.DataFrame(columns=["date"] + tickers)
    return pd.DataFrame(records).set_index("date")


def completeness_summary(matrix: pd.DataFrame) -> dict:
    """Fraction of expected snapshots present per ticker."""
    summary = {}
    for col in matrix.columns:
        filled = (matrix[col] > 0).sum()
        total = len(matrix)
        avg_per_day = matrix[col][matrix[col] > 0].mean() if filled > 0 else 0
        summary[col] = {
            "trading_days_with_data": int(filled),
            "total_trading_days":     int(total),
            "completeness_pct":       round(filled / total * 100, 1) if total else 0,
            "avg_snapshots_per_day":  round(float(avg_per_day), 1),
            "expected_per_day":       BARS_PER_DAY,
        }
    return summary


# ── 2. Field presence check ────────────────────────────────────────────────

def check_fields(tickers: list[str], dates: list[str]) -> dict:
    """
    Sample one snapshot per ticker (most recent available date) and verify
    expected fields are present.
    """
    results = {}
    for ticker in tickers:
        found = False
        for date in reversed(dates):
            paths = list_snapshots_for_date(date, ticker)
            if paths:
                try:
                    df = load_options_snapshot(paths[0])
                    cols = set(df.columns)
                    missing = sorted(EXPECTED_OPTION_COLS - cols)
                    extra   = sorted(cols - EXPECTED_OPTION_COLS - {"market_iv", "oi", "last_price", "date", "ticker"})
                    results[ticker] = {
                        "columns":         sorted(cols),
                        "missing_expected": missing,
                        "extra_fields":    extra,
                        "row_count_sample": len(df),
                    }
                    found = True
                    break
                except Exception as e:
                    results[ticker] = {"error": str(e)}
                    found = True
                    break
        if not found:
            results[ticker] = {"error": "no data found"}
    return results


# ── 3. Volume semantics check ──────────────────────────────────────────────

def check_volume_semantics(
    tickers: list[str] = VOLUME_CHECK_TICKERS,
    sample_date: str | None = None,
) -> dict:
    """
    For each ticker:
      a) Sum intraday 15-min option volumes across all snapshots on sample_date.
      b) Fetch yfinance daily volume for that date.
      c) Ratio > 1 strongly suggests cumulative (not interval) volumes.
    """
    dates = list_available_dates()
    if not dates:
        return {"error": "no data on drive"}

    results = {}
    for ticker in tickers:
        # Find a date with full or near-full data
        target_date = sample_date
        if target_date is None:
            for d in reversed(dates):
                if len(list_snapshots_for_date(d, ticker)) >= BARS_PER_DAY // 2:
                    target_date = d
                    break
        if target_date is None:
            results[ticker] = {"error": "no suitable date found"}
            continue

        paths = list_snapshots_for_date(target_date, ticker)
        if not paths:
            results[ticker] = {"error": f"no snapshots for {target_date}"}
            continue

        intraday_vols = []
        for p in paths:
            try:
                df = load_options_snapshot(p)
                if "volume" in df.columns:
                    intraday_vols.append(df["volume"].fillna(0).sum())
            except Exception:
                pass

        if not intraday_vols:
            results[ticker] = {"error": "could not read volumes"}
            continue

        # yfinance daily total volume for the underlying on that date
        try:
            hist = yf.Ticker(ticker).history(start=target_date, end=target_date, interval="1d")
            daily_volume = int(hist["Volume"].iloc[0]) if not hist.empty else None
        except Exception:
            daily_volume = None

        max_intraday = max(intraday_vols)
        sum_intraday = sum(intraday_vols)

        results[ticker] = {
            "date":           target_date,
            "n_snapshots":    len(intraday_vols),
            "max_snapshot_volume_sum":  int(max_intraday),
            "sum_of_snapshots_volume":  int(sum_intraday),
            "yfinance_daily_volume":    daily_volume,
            "ratio_sum_to_daily":       round(sum_intraday / daily_volume, 3) if daily_volume else None,
            "likely_cumulative":        (sum_intraday / daily_volume > 1.5) if daily_volume else None,
        }

    return results


# ── 4. Timestamp alignment ─────────────────────────────────────────────────

def check_timestamp_alignment(
    ticker: str,
    date: str,
) -> dict:
    """
    Load one day of option + stock snapshots for ticker.
    Compute the median lag (seconds) between option report_time and nearest stock bar.
    """
    opt_paths   = list_snapshots_for_date(date, ticker)
    stock_paths = list_stock_snapshots_for_date(date, ticker)

    if not opt_paths or not stock_paths:
        return {"error": "missing data for alignment check"}

    # Collect option report times
    opt_times = []
    for p in opt_paths:
        try:
            df = load_options_snapshot(p)
            if "report_time" in df.columns:
                t = pd.to_datetime(df["report_time"].iloc[0])
                opt_times.append(t)
        except Exception:
            pass

    # Collect stock bar times
    stock_times = []
    for p in stock_paths:
        try:
            df = load_stock_snapshot(p)
            ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
            t = pd.to_datetime(df[ts_col].iloc[0])
            stock_times.append(t)
        except Exception:
            pass

    if not opt_times or not stock_times:
        return {"error": "could not parse timestamps"}

    opt_ts   = pd.DatetimeIndex(opt_times)
    stock_ts = pd.DatetimeIndex(stock_times)

    # For each option snapshot, find the nearest stock bar
    lags_sec = []
    for ot in opt_ts:
        nearest = stock_ts[abs(stock_ts - ot).argmin()]
        lags_sec.append(abs((ot - nearest).total_seconds()))

    return {
        "ticker":             ticker,
        "date":               date,
        "n_option_snapshots": len(opt_times),
        "n_stock_snapshots":  len(stock_times),
        "median_lag_sec":     round(float(np.median(lags_sec)), 1),
        "max_lag_sec":        round(float(np.max(lags_sec)), 1),
        "pct_within_1min":    round(float(np.mean(np.array(lags_sec) <= 60) * 100), 1),
    }


# ── 5. Missing data patterns ───────────────────────────────────────────────

def build_missing_pattern(matrix: pd.DataFrame) -> dict:
    """
    Identify tickers / dates with zero or very sparse coverage.
    Returns dict with:
      - 'sparse_tickers': tickers with completeness < 50%
      - 'sparse_dates':   dates where > 50% of tickers have no data
    """
    ticker_completeness = (matrix > 0).mean(axis=0)
    date_completeness   = (matrix > 0).mean(axis=1)

    sparse_tickers = ticker_completeness[ticker_completeness < 0.5].index.tolist()
    sparse_dates   = date_completeness[date_completeness < 0.5].index.tolist()

    return {
        "sparse_tickers":         sparse_tickers,
        "sparse_dates":           sparse_dates,
        "ticker_completeness_pct": {
            t: round(v * 100, 1) for t, v in ticker_completeness.items()
        },
    }


# ── Main runner ────────────────────────────────────────────────────────────

def run_audit(
    start: str = BACKTEST_START,
    end:   str = BACKTEST_END,
    tickers: list[str] = TICKERS,
    output_path: Path = DATA_DIR / "audit_report.json",
) -> dict:
    print("=== Phase 0.1: Data Audit ===")

    dates = [d for d in list_available_dates() if start <= d <= end]
    print(f"  Found {len(dates)} date folders between {start} and {end}")

    print("  Building completeness matrix …")
    matrix = build_completeness_matrix(tickers, start, end)
    completeness = completeness_summary(matrix)

    print("  Checking field presence …")
    fields = check_fields(tickers, dates)

    print("  Checking volume semantics …")
    volume = check_volume_semantics()

    print("  Checking timestamp alignment …")
    alignment = {}
    sample_ticker = tickers[0]
    sample_date   = dates[-1] if dates else None
    if sample_date:
        alignment = check_timestamp_alignment(sample_ticker, sample_date)

    print("  Building missing data patterns …")
    missing = build_missing_pattern(matrix) if not matrix.empty else {}

    report = {
        "audit_date":       pd.Timestamp.now().isoformat(),
        "backtest_window":  {"start": start, "end": end},
        "n_trading_days":   len(dates),
        "tickers":          tickers,
        "completeness":     completeness,
        "field_check":      fields,
        "volume_semantics": volume,
        "timestamp_alignment": alignment,
        "missing_patterns": missing,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Audit report saved → {output_path}")

    # Print quick summary
    print("\n  Ticker completeness summary:")
    for t, stats in completeness.items():
        flag = "✓" if stats["completeness_pct"] >= 80 else "⚠"
        print(f"    {flag} {t:6s}: {stats['completeness_pct']:5.1f}%  "
              f"({stats['trading_days_with_data']}/{stats['total_trading_days']} days, "
              f"avg {stats['avg_snapshots_per_day']:.1f} snaps/day)")

    return report


if __name__ == "__main__":
    run_audit()
