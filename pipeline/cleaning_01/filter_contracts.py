"""
Phase 1.1-1.2 — Contract Filtering & Mid-Price Computation

Applies the full filter waterfall to a raw option chain DataFrame:
  1. Drop bid == 0 or ask == 0
  2. Drop crossed markets (ask < bid)
  3. Drop mid < intrinsic value
  4. Drop volume == 0 AND OI == 0
  5. Drop moneyness outside [0.85, 1.15]
  6. Drop TTE outside [5, 90] calendar days
  7. Compute mid = (bid + ask) / 2

Logs drop rates per filter; flags snapshots where any single filter
removes > 30% of contracts.

Usage
-----
    from pipeline.cleaning_01.filter_contracts import filter_snapshot

    filtered_df = filter_snapshot(raw_df, spot_price, log=True)
"""
import math
import logging

import pandas as pd
import numpy as np

from config import MONEYNESS_MIN, MONEYNESS_MAX, TTE_MIN_DAYS, TTE_MAX_DAYS

logger = logging.getLogger(__name__)

# Flag threshold: if a single filter drops more than this fraction, warn
SINGLE_FILTER_WARN_THRESHOLD = 0.30


def filter_snapshot(
    df: pd.DataFrame,
    spot: float,
    log: bool = False,
) -> pd.DataFrame:
    """
    Apply the full contract filter waterfall to one (ticker, timestamp) snapshot.

    Parameters
    ----------
    df    : raw option chain DataFrame.  Must contain columns:
            bid, ask, strike, volume, oi (or openinterest), type, tte
            where tte is in years (calendar / 365.25).
    spot  : underlying price at this snapshot timestamp.
    log   : if True, print filter step statistics.

    Returns
    -------
    Filtered DataFrame with added column 'mid'.
    """
    df = df.copy()
    n_start = len(df)

    if n_start == 0:
        return df

    # Normalise OI column name
    if "openinterest" in df.columns and "oi" not in df.columns:
        df = df.rename(columns={"openinterest": "oi"})
    if "oi" not in df.columns:
        df["oi"] = 0

    # Ensure numeric types
    for col in ["bid", "ask", "strike", "volume", "oi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    drop_counts = {}

    # ── Step 1: Drop bid == 0 or ask == 0 ─────────────────────────────────
    mask = (df["bid"] > 0) & (df["ask"] > 0)
    _apply(df, mask, "bid_or_ask_zero", drop_counts)
    df = df[mask].copy()

    # ── Step 2: Drop crossed markets ──────────────────────────────────────
    mask = df["ask"] >= df["bid"]
    _apply(df, mask, "crossed_market", drop_counts)
    df = df[mask].copy()

    # ── Step 3: Mid price & intrinsic value check ──────────────────────────
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    tte = df.get("tte", pd.Series(0.0, index=df.index))
    r   = df.get("r",   pd.Series(0.0, index=df.index))
    q   = df.get("q",   pd.Series(0.0, index=df.index))

    # Discounted intrinsic
    call_intrinsic = (spot * np.exp(-q * tte) - df["strike"] * np.exp(-r * tte)).clip(lower=0)
    put_intrinsic  = (df["strike"] * np.exp(-r * tte) - spot * np.exp(-q * tte)).clip(lower=0)

    is_call = df["type"] == "c"
    intrinsic = np.where(is_call, call_intrinsic, put_intrinsic)

    mask = df["mid"] >= intrinsic - 1e-4   # small tolerance for rounding
    _apply(df, mask, "below_intrinsic", drop_counts)
    df = df[mask].copy()

    # ── Step 4: Drop phantom quotes (volume == 0 AND OI == 0) ─────────────
    mask = ~((df["volume"] == 0) & (df["oi"] == 0))
    _apply(df, mask, "phantom_quote", drop_counts)
    df = df[mask].copy()

    # ── Step 5: Moneyness filter ───────────────────────────────────────────
    if spot > 0:
        moneyness = df["strike"] / spot
        mask = (moneyness >= MONEYNESS_MIN) & (moneyness <= MONEYNESS_MAX)
        _apply(df, mask, "out_of_moneyness_range", drop_counts)
        df = df[mask].copy()

    # ── Step 6: TTE filter ─────────────────────────────────────────────────
    if "tte" in df.columns:
        tte_days = df["tte"] * 365.25
        mask = (tte_days >= TTE_MIN_DAYS) & (tte_days <= TTE_MAX_DAYS)
        _apply(df, mask, "out_of_tte_range", drop_counts)
        df = df[mask].copy()

    n_end = len(df)

    if log:
        _log_stats(drop_counts, n_start, n_end)

    # ── Warn on large single-step drops ───────────────────────────────────
    for step, n_dropped in drop_counts.items():
        if n_start > 0 and n_dropped / n_start > SINGLE_FILTER_WARN_THRESHOLD:
            logger.warning(
                "Filter '%s' dropped %.0f%% of contracts (snapshot may be low quality)",
                step, n_dropped / n_start * 100,
            )

    return df


def add_filter_stats_column(df: pd.DataFrame, spot: float) -> dict:
    """
    Return a dict with per-step drop counts for a snapshot.
    Used by batch processing to aggregate across the day.
    """
    stats: dict[str, int] = {}
    n = len(df)
    if n == 0:
        return stats

    if "openinterest" in df.columns and "oi" not in df.columns:
        df = df.rename(columns={"openinterest": "oi"})

    for col in ["bid", "ask", "strike", "volume", "oi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    stats["total_raw"]          = n
    stats["bid_or_ask_zero"]    = int(((df["bid"] <= 0) | (df["ask"] <= 0)).sum())
    stats["crossed_market"]     = int((df["ask"] < df["bid"]).sum())

    mid = (df["bid"] + df["ask"]) / 2.0
    tte = df.get("tte", pd.Series(0.0, index=df.index))

    is_call = df["type"] == "c"
    call_intrinsic = (spot - df["strike"]).clip(lower=0)
    put_intrinsic  = (df["strike"] - spot).clip(lower=0)
    intrinsic = np.where(is_call, call_intrinsic, put_intrinsic)

    stats["below_intrinsic"]    = int((mid < intrinsic - 1e-4).sum())
    stats["phantom_quote"]      = int(((df.get("volume", 0) == 0) & (df.get("oi", 0) == 0)).sum())

    if spot > 0:
        mono = df["strike"] / spot
        stats["out_of_moneyness"] = int(((mono < MONEYNESS_MIN) | (mono > MONEYNESS_MAX)).sum())

    if "tte" in df.columns:
        tte_days = df["tte"] * 365.25
        stats["out_of_tte"] = int(((tte_days < TTE_MIN_DAYS) | (tte_days > TTE_MAX_DAYS)).sum())

    return stats


# ── Helpers ────────────────────────────────────────────────────────────────

def _apply(df: pd.DataFrame, mask: pd.Series, name: str, counts: dict) -> None:
    counts[name] = int((~mask).sum())


def _log_stats(drop_counts: dict, n_start: int, n_end: int) -> None:
    print(f"  Contract filter: {n_start} → {n_end} ({n_start - n_end} dropped)")
    for step, n_dropped in drop_counts.items():
        pct = n_dropped / n_start * 100 if n_start else 0
        print(f"    {step:<28s}: {n_dropped:4d}  ({pct:.1f}%)")
