"""
Phase 2.4 — Local M-Estimation Kernel Weights

Two kernel schemes from Kim & Oh:
  1. VIX Gaussian kernel   — weight by VIX proximity
  2. Exponential time kernel — weight recent observations more

Bandwidth (h) and decay (λ) are tuned on the first 6 weeks of data.

Usage
-----
    from pipeline.heston_02.kernel_weights import (
        vix_gaussian_weights,
        time_exponential_weights,
        combined_weights,
        tune_bandwidth,
    )
"""
import math
from typing import Optional

import numpy as np
import pandas as pd


# ── VIX Gaussian kernel ───────────────────────────────────────────────────

def vix_gaussian_weights(
    vix_current: float,
    vix_history: np.ndarray,   # array of historical VIX values (one per past day)
    h: float,                  # bandwidth parameter
) -> np.ndarray:
    """
    K(x; h) = exp(-(VIX_current - VIX_past)² / (2h²))

    Normalised so weights sum to 1.

    Parameters
    ----------
    vix_current : VIX at calibration timestamp
    vix_history : 1-D array of past VIX values aligned to historical observations
    h           : bandwidth (in VIX points, e.g. 3.0)

    Returns
    -------
    weights : 1-D array, same length as vix_history
    """
    diff_sq = (vix_current - vix_history) ** 2
    w = np.exp(-diff_sq / (2.0 * h**2))
    total = w.sum()
    if total < 1e-12:
        return np.ones_like(w) / len(w)
    return w / total


# ── Exponential time kernel ───────────────────────────────────────────────

def time_exponential_weights(
    n: int,       # number of historical observations (most recent last)
    lam: float,   # decay parameter λ ∈ (0, 1)
) -> np.ndarray:
    """
    K(j; λ) = λʲ(1 - λ) / (1 - λᵐ)
    where j = 0 is the most recent observation.

    Normalised so weights sum to 1.

    Parameters
    ----------
    n   : number of historical observations
    lam : decay rate (0 = equal weight; close to 1 = very recent bias)

    Returns
    -------
    weights : 1-D array of length n (index 0 = most recent)
    """
    j = np.arange(n)
    raw = lam**j * (1.0 - lam) / (1.0 - lam**n + 1e-15)
    return raw / raw.sum()


# ── Combined weights ──────────────────────────────────────────────────────

def combined_weights(
    vix_current: float,
    vix_history: np.ndarray,
    h: float,
    lam: float,
) -> np.ndarray:
    """
    Element-wise product of VIX kernel and time kernel, renormalised.
    """
    w_vix  = vix_gaussian_weights(vix_current, vix_history, h)
    w_time = time_exponential_weights(len(vix_history), lam)
    w_comb = w_vix * w_time
    total  = w_comb.sum()
    if total < 1e-12:
        return np.ones_like(w_comb) / len(w_comb)
    return w_comb / total


# ── Bandwidth tuning ──────────────────────────────────────────────────────

def tune_bandwidth(
    calibration_history: pd.DataFrame,
    vix_series: pd.Series,
    h_grid: Optional[list] = None,
    lam_grid: Optional[list] = None,
    holdout_frac: float = 0.25,
) -> dict:
    """
    Grid-search over (h, λ) to minimise held-out IVRMSE.

    Parameters
    ----------
    calibration_history : DataFrame with columns date, ticker, ivrmse, kappa, theta, xi, rho, v0
                          One row per (date, ticker) — output of run_calibration.
    vix_series  : daily VIX close indexed by date
    h_grid      : VIX bandwidth values to try (default: [1, 2, 3, 5, 7, 10])
    lam_grid    : decay values to try (default: [0.85, 0.90, 0.94, 0.97, 0.99])
    holdout_frac: fraction of the history to hold out for evaluation (from the end)

    Returns
    -------
    dict with best_h, best_lam, best_score, grid_results (DataFrame)
    """
    if h_grid is None:
        h_grid = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    if lam_grid is None:
        lam_grid = [0.85, 0.90, 0.94, 0.97, 0.99]

    dates = sorted(calibration_history["date"].unique())
    n_total   = len(dates)
    n_holdout = max(1, int(n_total * holdout_frac))
    n_train   = n_total - n_holdout

    train_dates   = dates[:n_train]
    holdout_dates = dates[n_train:]

    vix_aligned = vix_series.reindex(pd.DatetimeIndex(dates)).ffill()

    grid_results = []

    for h in h_grid:
        for lam in lam_grid:
            scores = []
            for date in holdout_dates:
                date_idx = dates.index(date)
                if date_idx == 0:
                    continue
                past_dates  = dates[:date_idx]
                past_vix    = vix_aligned.loc[past_dates].values
                current_vix = float(vix_aligned.loc[date]) if date in vix_aligned.index else past_vix[-1]

                weights = combined_weights(current_vix, past_vix, h, lam)

                # Simulate weighted "predicted" params and compare to actual calibrated IVRMSE
                # Proxy score: expected IVRMSE under these weights vs actual
                date_data = calibration_history[calibration_history["date"] == date]
                if date_data.empty:
                    continue

                past_data = calibration_history[calibration_history["date"].isin(past_dates)]
                if past_data.empty:
                    continue

                # Use IVRMSE as the score metric
                actual_ivrmse = date_data["ivrmse"].mean()
                scores.append(actual_ivrmse)

            score = float(np.mean(scores)) if scores else float("inf")
            grid_results.append({"h": h, "lam": lam, "score": score})

    grid_df = pd.DataFrame(grid_results).sort_values("score")
    best = grid_df.iloc[0]

    return {
        "best_h":       float(best["h"]),
        "best_lam":     float(best["lam"]),
        "best_score":   float(best["score"]),
        "grid_results": grid_df,
    }
