"""
Data loading utilities.

Handles the two distinct file format phases found on the Seagate drive:

  Phase 1  (2026-01-16 → 2026-02-12)
    Options: {TICKER}_options_{YYYY-MM-DD}_{HHMMSS}.parquet   ← daily EOD only
    Stock  : NOT present
    Schema : no expiry_date column → must parse from contractSymbol

  Phase 2  (2026-02-13 → 2026-03-24)
    Options: {TICKER}_options_{HHMM}.parquet                  ← intraday 15-min
    Stock  : NOT present
    Schema : expiry_date column added

  Phase 3  (2026-03-25 → present)
    Options: same as Phase 2
    Stock  : {TICKER}_stock_data_{HHMMSS}.parquet  (saved with index=True)
    Stock schema: tz-aware 'Datetime' index → timestamp column; Title-Case OHLCV

Stock data source split (STOCK_PARQUET_CUTOVER = 2026-03-28):
  Pre-cutover  (< 2026-03-28) : yfinance 15-min API only
                                 (drive files had index=False → timestamps lost)
  Post-cutover (≥ 2026-03-28) : Seagate 1-min parquets
                                 Each file covers a rolling 5-day window → deduplicate
                                 on timestamp after concatenation.

Key fixes applied vs original version:
  ✓ report_time parsed with explicit format='%Y-%m-%d-%H%M%S'
  ✓ expiry_date back-filled from contractSymbol for Phase 1 files
  ✓ openInterest rename works correctly (columns lowercased first, then renamed)
  ✓ Stock snapshot: tz-aware Datetime column stripped and renamed to timestamp
  ✓ Stock snapshot: ticker injected from filename prefix
  ✓ list_snapshots_for_date covers BOTH filename patterns; skips _calls_/_puts_ files
  ✓ Cutover-aware stock loading: yfinance pre-cutover, drive 1-min post-cutover
"""
import re
from datetime import datetime, time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config import (
    SEAGATE_ROOT, TICKERS,
    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
    STOCK_PARQUET_CUTOVER,
)

# report_time is stored as the string "YYYY-MM-DD-HHMMSS" by the cron job
_REPORT_TIME_FMT = "%Y-%m-%d-%H%M%S"


# ── File discovery ─────────────────────────────────────────────────────────

def list_available_dates(root: Path = SEAGATE_ROOT) -> list[str]:
    """Return sorted list of YYYY-MM-DD date folders on the drive."""
    if not root.exists():
        return []
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name)
    )


def list_snapshots_for_date(date: str, ticker: str, root: Path = SEAGATE_ROOT) -> list[Path]:
    """
    Return sorted option-chain parquet paths for (date, ticker).

    Handles both naming patterns:
      Phase 1: {TICKER}_options_{YYYY-MM-DD}_{HHMMSS}.parquet
      Phase 2: {TICKER}_options_{HHMM}.parquet

    Explicitly excludes _calls_ and _puts_ files (per-expiry splits from
    the very early collection period).
    """
    date_dir = root / date
    if not date_dir.exists():
        return []

    candidates = sorted(date_dir.glob(f"{ticker}_options_*.parquet"))
    # drop per-expiry split files that snuck in during early collection
    return [p for p in candidates if "_calls_" not in p.name and "_puts_" not in p.name]


def list_stock_snapshots_for_date(date: str, ticker: str, root: Path = SEAGATE_ROOT) -> list[Path]:
    """Return sorted stock bar parquet paths for (date, ticker)."""
    date_dir = root / date
    if not date_dir.exists():
        return []
    return sorted(date_dir.glob(f"{ticker}_stock_data_*.parquet"))


# ── Options snapshot loader ────────────────────────────────────────────────

def load_options_snapshot(path: Path) -> pd.DataFrame:
    """
    Load a single options parquet, normalise column names and types.

    Handles both Phase 1 (no expiry_date) and Phase 2+ (expiry_date present).
    """
    df = pd.read_parquet(path)

    # ── 1. Lowercase all column names first ─────────────────────────────
    df.columns = [c.lower().strip() for c in df.columns]

    # ── 2. Rename known aliases → pipeline-canonical names ──────────────
    rename = {
        "impliedvolatility": "market_iv",
        "openinterest":      "oi",
        "contractsymbol":    "contract_symbol",
        "lastprice":         "last_price",
        "lasttradedate":     "last_trade_date",
        "inthemoney":        "in_the_money",
        "contractsize":      "contract_size",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    # ── 2b. Fix boolean-valued columns stored as float/object ───────────
    # yfinance serialises booleans as Python bool, but after cross-file
    # pd.concat the column often becomes object with mixed 1.0/0.0/True/False.
    # Cast to proper bool so pyarrow can serialise it without errors.
    if "in_the_money" in df.columns:
        df["in_the_money"] = df["in_the_money"].fillna(False).astype(bool)

    # ── 3. Standardise option type to 'c' / 'p' ─────────────────────────
    if "type" in df.columns:
        df["type"] = df["type"].str.strip().str.lower().map(
            {"call": "c", "put": "p", "c": "c", "p": "p"}
        )

    # ── 4. Parse report_time from "YYYY-MM-DD-HHMMSS" string ────────────
    if "report_time" in df.columns and df["report_time"].dtype == object:
        df["report_time"] = pd.to_datetime(
            df["report_time"], format=_REPORT_TIME_FMT, errors="coerce"
        )
        # Drop any rows where parsing failed (malformed timestamps)
        df = df.dropna(subset=["report_time"])

    # ── 5. Back-fill expiry_date from contractSymbol for Phase 1 files ──
    if "expiry_date" not in df.columns:
        sym_col = "contract_symbol" if "contract_symbol" in df.columns else None
        if sym_col:
            df["expiry_date"] = df[sym_col].apply(_expiry_from_symbol)
        else:
            df["expiry_date"] = pd.NaT
    else:
        # Ensure it's a datetime; Phase 2 stores it as a YYYY-MM-DD string
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
        # Standard convention: options expire at 16:00 ET
        df["expiry_date"] = df["expiry_date"].apply(
            lambda d: d.replace(hour=16) if pd.notna(d) else d
        )

    # ── 6. Compute TTE in years (calendar days / 365.25) ─────────────────
    if "report_time" in df.columns and "expiry_date" in df.columns:
        df["tte"] = (
            (df["expiry_date"] - df["report_time"]).dt.total_seconds()
            / (365.25 * 86400)
        ).clip(lower=0)

    return df


def _expiry_from_symbol(symbol: str) -> Optional[pd.Timestamp]:
    """
    Parse expiry date from OCC contract symbol: {TICKER}{YYMMDD}{C/P}{STRIKE}
    e.g. AAPL260116C00150000 → 2026-01-16 16:00:00
    """
    if not isinstance(symbol, str):
        return pd.NaT
    m = re.search(r"(\d{6})[CP]", symbol.upper())
    if not m:
        return pd.NaT
    try:
        return pd.to_datetime(m.group(1), format="%y%m%d").replace(hour=16)
    except ValueError:
        return pd.NaT


# ── Stock snapshot loader ──────────────────────────────────────────────────

def load_stock_snapshot(path: Path) -> pd.DataFrame:
    """
    Load a single post-cutover stock bar parquet (saved with index=True).

    Post-cutover format (cron_stock.py fixed 2026-03-28):
      - Index is a tz-aware DatetimeIndex saved as column 'Datetime' (or 'datetime')
      - OHLCV columns are Title-Case: Open, High, Low, Close, Volume
      - Each file contains ~1950 rows (5 rolling trading days × 390 1-min bars)
      - No explicit ticker column → injected from filename prefix

    Parameters
    ----------
    path : full path to the parquet, e.g. …/2026-03-28/AAPL_stock_data_093001.parquet
    """
    df = pd.read_parquet(path)

    # Lowercase all column names and normalise spaces
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # The tz-aware DatetimeIndex is saved as 'datetime' (or 'date') by index=True
    ts_col = next(
        (c for c in df.columns if c in ("datetime", "date", "timestamp")),
        None,
    )
    if ts_col:
        parsed = pd.to_datetime(df[ts_col])
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_convert(None)   # strip tz without shifting (already ET)
        df["timestamp"] = parsed
        if ts_col != "timestamp":
            df = df.drop(columns=[ts_col])
    else:
        # Fallback: index itself may carry timestamps (if reset_index was not called)
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            df["timestamp"] = idx.tz_convert(None) if idx.tz else idx
            df = df.reset_index(drop=True)
        else:
            df["timestamp"] = pd.NaT

    # Inject ticker from filename prefix: "AAPL_stock_data_093001.parquet" → "AAPL"
    df["ticker"] = path.stem.split("_")[0]

    return df


# ── Batch loaders ──────────────────────────────────────────────────────────

def load_options_for_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    root: Path = SEAGATE_ROOT,
    phase2_only: bool = True,
) -> pd.DataFrame:
    """
    Load all option chain snapshots for *ticker* between start_date and
    end_date (inclusive, YYYY-MM-DD strings).

    Parameters
    ----------
    phase2_only : if True (default) skip Phase 1 dates that only have EOD
                  snapshots — not usable for 15-min intraday backtesting.
    """
    dates = [d for d in list_available_dates(root) if start_date <= d <= end_date]
    # Phase 2 starts 2026-02-13
    if phase2_only:
        dates = [d for d in dates if d >= "2026-02-13"]

    frames = []
    for date in dates:
        paths = list_snapshots_for_date(date, ticker, root)
        for p in paths:
            try:
                df = load_options_snapshot(p)
                df["date"]   = date
                df["ticker"] = ticker
                frames.append(df)
            except Exception as e:
                print(f"[WARN] Could not load {p}: {e}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_stock_for_ticker_from_drive(
    ticker: str,
    start_date: str,
    end_date: str,
    root: Path = SEAGATE_ROOT,
) -> pd.DataFrame:
    """
    Load post-cutover 1-min stock bars from Seagate for *ticker*.

    Only loads date folders >= STOCK_PARQUET_CUTOVER (2026-03-28), because
    earlier files were saved with index=False and have no timestamps.

    Each parquet covers a rolling 5-day window, so the same 1-min bar can
    appear in multiple files. After concatenation, rows are deduplicated on
    (ticker, timestamp) and then filtered to [start_date, end_date].
    """
    # Only load post-cutover folders — pre-cutover files lack timestamps
    effective_start = max(start_date, STOCK_PARQUET_CUTOVER)
    dates = [
        d for d in list_available_dates(root)
        if effective_start <= d <= end_date
    ]

    frames = []
    for date in dates:
        paths = list_stock_snapshots_for_date(date, ticker, root)
        for p in paths:
            try:
                df = load_stock_snapshot(p)
                frames.append(df)
            except Exception as e:
                print(f"[WARN] Could not load {p}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined["timestamp"] = ts.dt.tz_convert(None) if ts.dt.tz is not None else ts
    combined = combined.dropna(subset=["timestamp"])

    if combined.empty:
        return combined

    # cron_stock.py runs once per Friday → each file covers a distinct 5-day
    # window with no overlap between runs; no deduplication needed.
    # Filter to the requested window (each file spans the 5 days up to that Friday,
    # so bars outside [start_date, end_date] may be present)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    mask = (
        (combined["timestamp"] >= start_ts)
        & (combined["timestamp"] < end_ts)
    )
    return combined[mask].sort_values("timestamp").reset_index(drop=True)


def _yfinance_stock_pull(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "15m",
) -> pd.DataFrame:
    """
    Pull OHLCV bars from yfinance; return tz-naive DataFrame or empty.

    Yahoo Finance caps each sub-daily history request at 7 calendar days.
    A single call spanning the full backtest window silently returns only
    the last 7-day chunk.  This function chunks the range into 7-day windows
    and stitches the results together to cover the full date range.
    """
    _CHUNK_DAYS = 7  # max calendar days Yahoo Finance returns per sub-daily request

    chunk_start = pd.Timestamp(start_date)
    # yfinance `end` is exclusive, so advance one day past end_date
    request_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    raw_frames = []
    while chunk_start < request_end:
        chunk_end = min(chunk_start + pd.Timedelta(days=_CHUNK_DAYS), request_end)
        try:
            hist = yf.Ticker(ticker).history(
                start=chunk_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
            )
            if not hist.empty:
                raw_frames.append(hist.reset_index())
        except Exception as e:
            print(
                f"[WARN] yfinance {interval} chunk "
                f"{chunk_start.date()} → {chunk_end.date()} "
                f"failed for {ticker}: {e}"
            )
        chunk_start = chunk_end   # next chunk starts exactly where this one ended

    if not raw_frames:
        return pd.DataFrame()

    hist = pd.concat(raw_frames, ignore_index=True)

    # Normalise column names
    hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]
    ts_col = next(
        (c for c in hist.columns if "datetime" in c or c == "date"),
        None,
    )
    if ts_col:
        hist.rename(columns={ts_col: "timestamp"}, inplace=True)

    hist["ticker"] = ticker
    # Strip timezone keeping ET wall-clock time.
    # tz_localize(None) removes tzinfo without shifting the clock hands,
    # so "09:30-05:00" (EST) → "09:30" tz-naive.
    # tz_convert(None) would first convert to UTC → "14:30" tz-naive, which
    # causes _filter_market_hours to discard most of the trading session.
    ts = pd.to_datetime(hist["timestamp"])
    hist["timestamp"] = ts.dt.tz_localize(None) if ts.dt.tz is not None else ts

    # Drop duplicates that can appear at chunk boundaries
    hist = hist.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return hist


def resample_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a 1-min OHLCV DataFrame to 15-min bars.

    Expects columns: timestamp, open, high, low, close, volume, ticker.
    Groups by ticker so a multi-ticker DataFrame is handled correctly.
    """
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    agg_map = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    # Keep only columns that are actually present
    agg_map = {k: v for k, v in agg_map.items() if k in df.columns}

    out_frames = []
    for ticker, grp in df.groupby("ticker"):
        resampled = (
            grp.set_index("timestamp")
            .resample("15min")[list(agg_map.keys())]
            .agg(agg_map)
            .dropna(how="all")
            .reset_index()
        )
        resampled["ticker"] = ticker
        out_frames.append(resampled)

    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()


def load_stock_for_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    root: Path = SEAGATE_ROOT,
    market_hours_only: bool = True,
    resample_1min_to_15min: bool = True,
) -> pd.DataFrame:
    """
    Return OHLCV bars for *ticker* across the full date range, using the
    appropriate source for each portion of the window.

    Source split at STOCK_PARQUET_CUTOVER (2026-03-28):
      Pre-cutover  (< 2026-03-28) : yfinance 15-min API
                                     Earlier Seagate files had index=False →
                                     timestamps were not saved.
      Post-cutover (≥ 2026-03-28) : Seagate 1-min parquets (index=True, fixed)
                                     Optionally resampled to 15-min for the
                                     pipeline (resample_1min_to_15min=True).

    Parameters
    ----------
    resample_1min_to_15min : if True (default), resample post-cutover 1-min
                              bars to 15-min so the returned DataFrame has a
                              uniform bar frequency matching the pre-cutover
                              yfinance data.
    """
    frames = []

    # ── Pre-cutover: yfinance 15-min ─────────────────────────────────────
    pre_end = min(end_date, "2026-03-27")   # day before cutover
    if start_date <= pre_end:
        yf_df = _yfinance_stock_pull(ticker, start_date, pre_end, interval="15m")
        if not yf_df.empty:
            frames.append(yf_df)

    # ── Post-cutover: Seagate 1-min ───────────────────────────────────────
    post_start = max(start_date, STOCK_PARQUET_CUTOVER)
    if post_start <= end_date:
        drive_df = load_stock_for_ticker_from_drive(ticker, post_start, end_date, root)
        if not drive_df.empty:
            if resample_1min_to_15min:
                drive_df = resample_to_15min(drive_df)
            frames.append(drive_df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["ticker", "timestamp"], keep="first")

    if market_hours_only and "timestamp" in out.columns:
        out = _filter_market_hours(out, "timestamp")

    return out.sort_values("timestamp").reset_index(drop=True)


def load_all_options(
    start_date: str,
    end_date: str,
    tickers: list[str] = TICKERS,
    root: Path = SEAGATE_ROOT,
    phase2_only: bool = True,
) -> pd.DataFrame:
    """Load option chains for all tickers in the universe."""
    frames = [
        load_options_for_ticker(t, start_date, end_date, root, phase2_only)
        for t in tickers
    ]
    non_empty = [f for f in frames if not f.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def load_all_stock(
    start_date: str,
    end_date: str,
    tickers: list[str] = TICKERS,
    root: Path = SEAGATE_ROOT,
) -> pd.DataFrame:
    """Load stock bars for all tickers (yfinance + Seagate supplement)."""
    frames = [load_stock_for_ticker(t, start_date, end_date, root) for t in tickers]
    non_empty = [f for f in frames if not f.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


# ── Helpers ────────────────────────────────────────────────────────────────

def _filter_market_hours(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Keep only bars whose timestamp falls within 9:30–16:00."""
    ts = df[ts_col]
    open_time  = time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
    close_time = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)
    mask = ts.dt.time.between(open_time, close_time)
    return df[mask].copy()


def tte_years(report_time: datetime, expiry: datetime) -> float:
    """Time-to-expiry in years (calendar days / 365.25)."""
    delta = (expiry - report_time).total_seconds() / (365.25 * 86400)
    return max(delta, 0.0)
