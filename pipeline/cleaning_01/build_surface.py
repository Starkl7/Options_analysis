"""
Phase 1.4 — IV Surface Construction

Per (ticker, timestamp, expiry):
  - Fit a smoothing spline across strikes
  - Compute ATM IV (at forward price F)
  - Compute 25-delta call / put IVs and derived skew & butterfly metrics
  - Store surface summary as a tidy DataFrame row

Output: data/iv_surfaces.parquet
  columns: ticker, timestamp, expiry, tte, atm_iv, skew_25d, butterfly_25d,
           iv_25d_call, iv_25d_put, n_contracts, spot, rf_rate, div_yield

Run:
    python -m pipeline.01_cleaning.build_surface
"""
import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq

from config import DATA_DIR, BACKTEST_START, BACKTEST_END
from pipeline.utils.bs_model import bs_price, forward_price

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Minimum contracts per expiry slice to attempt spline fitting
MIN_CONTRACTS_FOR_SPLINE = 12   # raised from 5; thin slices produce unreliable splines
ATM_IV_MIN = 0.03               # 3%  — reject implausibly low ATM IV (spline collapse)
ATM_IV_MAX = 1.50               # 150% — reject implausibly high ATM IV (spline explosion)


# ── Surface for one expiry slice ───────────────────────────────────────────

def fit_expiry_slice(
    df_slice: pd.DataFrame,   # contracts for one (ticker, ts, expiry)
    spot: float,
    r: float,
    q: float,
    tte: float,
) -> Optional[dict]:
    """
    Fit a smoothing spline to the IV smile for a single expiry.

    Returns a dict with surface metrics, or None if not enough data.
    """
    df = df_slice.copy().dropna(subset=["iv", "strike"])

    if len(df) < MIN_CONTRACTS_FOR_SPLINE:
        return None

    # Sort by strike
    df = df.sort_values("strike")
    strikes = df["strike"].values.astype(float)
    ivs     = df["iv"].values.astype(float)

    # Deduplicate strikes (take mean IV if duplicated — shouldn't happen often)
    strikes, unique_idx = np.unique(strikes, return_index=True)
    if len(strikes) < MIN_CONTRACTS_FOR_SPLINE:
        return None

    # Average IVs at duplicate strikes
    iv_means = np.array([ivs[df["strike"].values == k].mean() for k in strikes])

    # Fit smoothing spline (s is the smoothing factor — tune via residual)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spline = UnivariateSpline(strikes, iv_means, s=len(strikes) * 1e-4, k=3, ext=3)
    except Exception:
        return None

    # ATM: interpolate at forward price
    F = forward_price(spot, r, q, tte)

    # Reject if forward price falls outside the fitted strike range — any
    # value would be an extrapolation, not an interpolation.
    if F < strikes.min() or F > strikes.max():
        return None

    atm_iv = float(spline(F))

    # Sanity-check ATM IV: implausible values indicate a bad spline fit.
    if not (ATM_IV_MIN <= atm_iv <= ATM_IV_MAX):
        return None

    # 25-delta strike solving
    iv_25d_call = _solve_25delta_iv(spline, spot, r, q, tte, delta_target=0.25, option_type="c", strikes=strikes)
    iv_25d_put  = _solve_25delta_iv(spline, spot, r, q, tte, delta_target=-0.25, option_type="p", strikes=strikes)

    skew_25d      = (iv_25d_put  - iv_25d_call) if iv_25d_call and iv_25d_put else None
    butterfly_25d = ((iv_25d_put + iv_25d_call) / 2.0 - atm_iv) if iv_25d_call and iv_25d_put else None

    return {
        "atm_iv":       round(atm_iv, 6),
        "iv_25d_call":  round(iv_25d_call, 6)  if iv_25d_call  else None,
        "iv_25d_put":   round(iv_25d_put, 6)   if iv_25d_put   else None,
        "skew_25d":     round(skew_25d, 6)     if skew_25d     is not None else None,
        "butterfly_25d":round(butterfly_25d,6) if butterfly_25d is not None else None,
        "n_contracts":  len(df),
        "strike_min":   float(strikes.min()),
        "strike_max":   float(strikes.max()),
        "spline":       spline,   # kept in memory; not serialised to parquet
    }


def _solve_25delta_iv(
    spline: UnivariateSpline,
    spot: float,
    r: float,
    q: float,
    tte: float,
    delta_target: float,
    option_type: str,
    strikes: np.ndarray,
) -> Optional[float]:
    """
    Find the strike K such that BS delta == delta_target, then read IV
    from the spline at that K.
    """
    from scipy.stats import norm

    if tte <= 0:
        return None

    K_min = strikes.min() * 0.9
    K_max = strikes.max() * 1.1

    def delta_diff(K: float) -> float:
        iv = float(spline(K))
        if iv <= 0:
            return float("inf")
        if tte <= 0 or iv <= 0:
            return float("inf")
        sqrt_t = math.sqrt(tte)
        d1 = (math.log(spot / K) + (r - q + 0.5 * iv**2) * tte) / (iv * sqrt_t)
        if option_type == "c":
            d = math.exp(-q * tte) * norm.cdf(d1)
        else:
            d = math.exp(-q * tte) * (norm.cdf(d1) - 1.0)
        return d - delta_target

    try:
        f_lo = delta_diff(K_min)
        f_hi = delta_diff(K_max)
        if f_lo * f_hi > 0:
            return None
        K_25d = brentq(delta_diff, K_min, K_max, xtol=1e-4, maxiter=50)
        iv = float(spline(K_25d))
        return iv if 0 < iv < 5 else None
    except Exception:
        return None


# ── Batch surface builder ──────────────────────────────────────────────────

def build_iv_surfaces(
    iv_df: pd.DataFrame,
    output_path=DATA_DIR / "iv_surfaces.parquet",
) -> pd.DataFrame:
    """
    Build the IV surface summary for all (ticker, timestamp, expiry) combinations.

    Parameters
    ----------
    iv_df : output of compute_iv_batch — has columns:
            ticker, report_time, expiry_date, strike, tte, type, iv, spot, rf_rate, div_yield

    Returns
    -------
    DataFrame with one row per (ticker, timestamp, expiry)
    """
    if iv_df.empty:
        return pd.DataFrame()

    expiry_col = "expiry_date" if "expiry_date" in iv_df.columns else "tte"
    records = []

    group_keys = ["ticker", "report_time", expiry_col]
    for keys, grp in iv_df.groupby(group_keys):
        ticker, ts, expiry = keys

        # Get slice-level metadata from first row
        spot      = float(grp["spot"].iloc[0])
        r         = float(grp["rf_rate"].iloc[0])
        q         = float(grp["div_yield"].iloc[0])
        tte       = float(grp["tte"].mean())   # should be nearly constant per expiry

        surface = fit_expiry_slice(grp, spot, r, q, tte)
        if surface is None:
            continue

        record = {
            "ticker":       ticker,
            "timestamp":    ts,
            expiry_col:     expiry,
            "tte":          round(tte, 6),
            "spot":         spot,
            "rf_rate":      r,
            "div_yield":    q,
            **{k: v for k, v in surface.items() if k != "spline"},
        }
        records.append(record)

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records)
    out = out.sort_values(["ticker", "timestamp", "tte"])

    # ── Enforce monotone total variance across expiries ──────────────────
    # For each (ticker, timestamp), drop any expiry slice whose total_var
    # (= atm_iv² × tte) is lower than the previous kept slice.  A greedy
    # forward pass keeps the maximum number of slices while guaranteeing
    # the Heston-friendly no-calendar-arb property.
    out["total_var"] = out["atm_iv"] ** 2 * out["tte"]
    keep_mask = []
    for (_, _), grp in out.groupby(["ticker", "timestamp"], sort=False):
        tv_prev = -np.inf
        for tv in grp["total_var"].values:
            if tv >= tv_prev:
                keep_mask.append(True)
                tv_prev = tv
            else:
                keep_mask.append(False)

    out = out[keep_mask].drop(columns=["total_var"])

    # Serialise (exclude spline objects — stored in memory only)
    save_cols = [c for c in out.columns if c != "spline"]
    out[save_cols].to_parquet(output_path, index=False)
    print(f"IV surfaces saved: {len(out)} rows → {output_path}")

    return out


# ── Convenience: ATM IV time-series ───────────────────────────────────────

def get_atm_iv_series(
    surface_df: pd.DataFrame,
    ticker: str,
    min_tte_days: float = 14,
    max_tte_days: float = 30,
) -> pd.Series:
    """
    Return a timestamp-indexed Series of ATM IV for nearest qualifying expiry.
    """
    sub = surface_df[
        (surface_df["ticker"] == ticker) &
        (surface_df["tte"] * 365.25 >= min_tte_days) &
        (surface_df["tte"] * 365.25 <= max_tte_days)
    ].copy()

    if sub.empty:
        return pd.Series(dtype=float)

    # Pick the shortest qualifying expiry per timestamp
    sub = sub.sort_values(["timestamp", "tte"])
    sub = sub.groupby("timestamp").first().reset_index()

    return sub.set_index("timestamp")["atm_iv"]


if __name__ == "__main__":
    from config import DATA_DIR

    print("=== Phase 1.4: Building IV Surfaces ===")

    print("  Loading iv_data.parquet …")
    iv_df = pd.read_parquet(DATA_DIR / "iv_data.parquet")
    if iv_df.empty:
        print("  [ERROR] iv_data.parquet is empty. Run compute_iv first.")
        raise SystemExit(1)
    print(f"  Loaded {len(iv_df):,} rows  "
          f"({iv_df['ticker'].nunique()} tickers, "
          f"{iv_df['report_time'].nunique()} snapshots)")

    surface_df = build_iv_surfaces(iv_df, output_path=DATA_DIR / "iv_surfaces.parquet")

    if surface_df.empty:
        print("  [WARN] No surface rows produced.")
    else:
        print(f"\n  Summary:")
        print(f"    Tickers        : {surface_df['ticker'].nunique()}")
        print(f"    Timestamps     : {surface_df['timestamp'].nunique()}")
        print(f"    Expiry slices  : {len(surface_df)}")
        atm = surface_df['atm_iv'].dropna()
        print(f"    ATM IV range   : {atm.min():.3f} – {atm.max():.3f}")
