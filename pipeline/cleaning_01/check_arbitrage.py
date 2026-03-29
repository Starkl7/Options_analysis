"""
Phase 1.5 — No-Arbitrage Surface Checks

Two checks per (ticker, timestamp):
  1. Vertical spread (call spread) check:
     dC/dK ≤ 0 at 20 evenly spaced K-grid points per expiry slice.
     A positive slope implies a negative call spread — arbitrage.

  2. Calendar spread check:
     Total variance V(T) = IV(T)² × T must be non-decreasing across expiries.

Flags timestamps that fail either check.
These flagged timestamps are excluded from Heston calibration.

Usage
-----
    from pipeline.cleaning_01.check_arbitrage import run_arbitrage_checks

    flagged = run_arbitrage_checks(surface_df)
    # flagged is a set of (ticker, timestamp) tuples
"""
import math
from typing import NamedTuple

import numpy as np
import pandas as pd

# Number of K-grid points for vertical spread check
N_GRID_POINTS = 20

# Tolerance: allow tiny positive slopes due to spline noise
SLOPE_TOLERANCE = 1e-4


class ArbitrageResult(NamedTuple):
    ticker:        str
    timestamp:     pd.Timestamp
    vertical_fail: bool    # True if any expiry slice fails vertical check
    calendar_fail: bool    # True if calendar spread check fails
    n_expiries:    int
    details:       dict    # per-expiry breakdown


def check_vertical_spread(
    spline,        # fitted UnivariateSpline for one expiry
    spot: float,
    r: float,
    q: float,
    tte: float,
    strike_min: float,
    strike_max: float,
) -> bool:
    """
    Check that no call spread has negative value.

    Numerically: compute d(C)/dK at N_GRID_POINTS between strike_min and strike_max.
    If any slope > SLOPE_TOLERANCE, the call spread is violated.

    Returns True if check PASSES (no arbitrage detected).
    """
    from pipeline.utils.bs_model import bs_price as _bs_price

    Ks = np.linspace(strike_min, strike_max, N_GRID_POINTS)
    try:
        ivs = np.array([float(spline(k)) for k in Ks])
    except Exception:
        return False

    # Compute call prices at each K
    call_prices = np.array([
        _bs_price(spot, k, tte, r, 0.0, max(iv, 0.001), "c")
        for k, iv in zip(Ks, ivs)
    ])

    # Central differences for dC/dK
    dK = Ks[1] - Ks[0]
    slopes = np.diff(call_prices) / dK

    # All slopes must be ≤ SLOPE_TOLERANCE (call price decreases with K)
    return bool(np.all(slopes <= SLOPE_TOLERANCE))


def check_calendar_spread(expiry_surfaces: list[dict]) -> bool:
    """
    expiry_surfaces: list of dicts with keys 'tte' and 'atm_iv',
                     sorted by tte ascending.

    Checks V(T1) ≤ V(T2) for all consecutive pairs, where V(T) = IV² × T.
    Returns True if check PASSES.
    """
    if len(expiry_surfaces) < 2:
        return True   # nothing to compare

    total_vars = [(s["tte"], s["atm_iv"]**2 * s["tte"]) for s in expiry_surfaces if s["atm_iv"]]
    total_vars.sort(key=lambda x: x[0])

    for i in range(1, len(total_vars)):
        tv_prev = total_vars[i - 1][1]
        tv_curr = total_vars[i][1]
        if tv_curr < tv_prev - 1e-6:   # allow tiny numerical noise
            return False
    return True


def run_arbitrage_checks(
    surface_df: pd.DataFrame,
    spline_store: dict | None = None,
) -> tuple[set, pd.DataFrame]:
    """
    Run vertical and calendar arbitrage checks over all (ticker, timestamp) groups.

    Parameters
    ----------
    surface_df  : output of build_iv_surfaces (rows per ticker × timestamp × expiry)
    spline_store: optional dict mapping (ticker, timestamp, expiry) → spline object
                  (needed for vertical check; if None, vertical check is skipped)

    Returns
    -------
    flagged     : set of (ticker, timestamp) tuples that fail at least one check
    results_df  : DataFrame of ArbitrageResult records for all timestamps
    """
    flagged  = set()
    records  = []

    group_keys = ["ticker", "timestamp"]
    for (ticker, ts), grp in surface_df.groupby(group_keys):
        grp = grp.sort_values("tte")
        n_expiries = len(grp)

        expiry_surfaces = grp[["tte", "atm_iv"]].dropna().to_dict("records")

        # ── Calendar check ────────────────────────────────────────────────
        cal_pass = check_calendar_spread(expiry_surfaces)

        # ── Vertical check ────────────────────────────────────────────────
        vert_pass = True
        vert_details = {}

        if spline_store is not None:
            expiry_col = "expiry_date" if "expiry_date" in grp.columns else "tte"
            for _, row in grp.iterrows():
                expiry = row[expiry_col]
                key    = (ticker, ts, expiry)
                spline = spline_store.get(key)
                if spline is None:
                    continue

                passes = check_vertical_spread(
                    spline       = spline,
                    spot         = float(row["spot"]),
                    r            = float(row["rf_rate"]),
                    q            = float(row["div_yield"]),
                    tte          = float(row["tte"]),
                    strike_min   = float(row.get("strike_min", row["spot"] * 0.85)),
                    strike_max   = float(row.get("strike_max", row["spot"] * 1.15)),
                )
                vert_details[str(expiry)] = passes
                if not passes:
                    vert_pass = False

        vertical_fail = not vert_pass
        calendar_fail = not cal_pass

        if vertical_fail or calendar_fail:
            flagged.add((ticker, ts))

        records.append(ArbitrageResult(
            ticker        = ticker,
            timestamp     = ts,
            vertical_fail = vertical_fail,
            calendar_fail = calendar_fail,
            n_expiries    = n_expiries,
            details       = vert_details,
        ))

    results_df = pd.DataFrame(records)

    if not results_df.empty:
        n_flagged = len(flagged)
        n_total   = len(results_df)
        print(f"Arbitrage check: {n_flagged}/{n_total} timestamps flagged "
              f"({n_flagged/n_total*100:.1f}%)")
        vert_fails = results_df["vertical_fail"].sum() if "vertical_fail" in results_df else 0
        cal_fails  = results_df["calendar_fail"].sum() if "calendar_fail" in results_df else 0
        print(f"  Vertical failures:  {vert_fails}")
        print(f"  Calendar failures:  {cal_fails}")

    return flagged, results_df


def filter_surface_by_arbitrage(
    surface_df: pd.DataFrame,
    flagged: set,
) -> pd.DataFrame:
    """
    Remove all surface rows corresponding to flagged (ticker, timestamp) pairs.
    """
    if not flagged:
        return surface_df

    def is_flagged(row) -> bool:
        return (row["ticker"], row["timestamp"]) in flagged

    mask = ~surface_df.apply(is_flagged, axis=1)
    n_before = len(surface_df)
    out = surface_df[mask].copy()
    print(f"Removed {n_before - len(out)} surface rows due to arbitrage violations.")
    return out


if __name__ == "__main__":
    from config import DATA_DIR

    print("=== Phase 1.5: No-Arbitrage Surface Checks ===")

    print("  Loading iv_surfaces.parquet …")
    surface_df = pd.read_parquet(DATA_DIR / "iv_surfaces.parquet")
    if surface_df.empty:
        print("  [ERROR] iv_surfaces.parquet is empty. Run build_surface first.")
        raise SystemExit(1)

    # Splines are not serialised — vertical check will be skipped (calendar only)
    flagged, results_df = run_arbitrage_checks(surface_df, spline_store=None)

    # Save flagged timestamps so run_calibration can load them
    if not results_df.empty:
        flag_df = pd.DataFrame(
            [(t, str(ts)) for t, ts in flagged],
            columns=["ticker", "timestamp"],
        )
        out_path = DATA_DIR / "flagged_timestamps.parquet"
        flag_df.to_parquet(out_path, index=False)
        print(f"  Flagged timestamps saved → {out_path}  ({len(flag_df)} rows)")
    else:
        print("  No surface data to check.")
