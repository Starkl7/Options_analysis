"""
Phase 2.2-2.4 — Heston Calibration Objective & Two-Stage Optimizer

Objective: vega-weighted IVRMSE with Feller penalty.
Optimizer: Stage 1 → differential_evolution (global), Stage 2 → LM local refinement.

Usage
-----
    from pipeline.heston_02.calibration import calibrate_heston

    result = calibrate_heston(
        contracts=df,   # DataFrame with strike, tte, type, iv, vega columns
        S=spot, r=r, q=q,
        prev_params=None,  # or dict from prior day for warm-start
    )
    # result: {'kappa', 'theta', 'xi', 'rho', 'v0', 'ivrmse', 'converged', 'n_contracts'}
"""
import warnings
from typing import Optional

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from config import (
    HESTON_BOUNDS, FELLER_PENALTY_LAMBDA,
    HESTON_IVRMSE_THRESHOLD,
    DE_POPSIZE, DE_MAXITER, DE_TOL,
    HESTON_MONO_MIN, HESTON_MONO_MAX,
    HESTON_TTE_MIN_DAYS, HESTON_TTE_MAX_DAYS,
    HESTON_MIN_CONTRACTS_PER_EXPIRY,
)
from pipeline.heston_02.heston_pricer import heston_price_batch
from pipeline.utils.bs_model import implied_vol, bs_vega

warnings.filterwarnings("ignore")

# Parameter order for the optimiser vector
_PARAM_NAMES = ["kappa", "theta", "xi", "rho", "v0"]

_BOUNDS_LIST = [
    HESTON_BOUNDS["kappa"],
    HESTON_BOUNDS["theta"],
    HESTON_BOUNDS["xi"],
    HESTON_BOUNDS["rho"],
    HESTON_BOUNDS["v0"],
]


# ── Calibration universe selection ────────────────────────────────────────

def select_calibration_contracts(
    iv_df,           # snapshot DataFrame with columns: strike, tte, type, iv, spot, rf_rate, div_yield
    spot: float,
) -> "pd.DataFrame":
    """
    Select the subset of contracts used for Heston calibration:
      - OTM only (calls K > F, puts K < F)
      - Moneyness [0.90, 1.10]
      - TTE [14, 90] days
      - Minimum HESTON_MIN_CONTRACTS_PER_EXPIRY per expiry slice
    """
    import pandas as pd
    import math

    df = iv_df.copy()
    r  = float(df["rf_rate"].iloc[0])
    q  = float(df["div_yield"].iloc[0])

    # Forward price per contract (each may have different TTE)
    df["F"] = spot * np.exp((r - q) * df["tte"])

    # OTM filter
    df["otm"] = (
        ((df["type"] == "c") & (df["strike"] > df["F"])) |
        ((df["type"] == "p") & (df["strike"] < df["F"]))
    )
    df = df[df["otm"]].copy()

    # Moneyness filter
    df["moneyness"] = df["strike"] / spot
    df = df[(df["moneyness"] >= HESTON_MONO_MIN) & (df["moneyness"] <= HESTON_MONO_MAX)].copy()

    # TTE filter
    df = df[(df["tte"] * 365.25 >= HESTON_TTE_MIN_DAYS) &
            (df["tte"] * 365.25 <= HESTON_TTE_MAX_DAYS)].copy()

    # Drop expiry slices with too few contracts
    if "expiry_date" in df.columns:
        expiry_col = "expiry_date"
    else:
        # Bin TTEs into roughly daily buckets
        df["tte_bucket"] = (df["tte"] * 365).round(0)
        expiry_col = "tte_bucket"

    counts = df.groupby(expiry_col)[expiry_col].transform("count")
    df = df[counts >= HESTON_MIN_CONTRACTS_PER_EXPIRY].copy()

    # Limit to 2-3 expiry slices (closest ones first)
    expiries = sorted(df[expiry_col].unique())[:3]
    df = df[df[expiry_col].isin(expiries)].copy()

    # Compute per-contract vega for weighting
    def _vega(row):
        return bs_vega(spot, float(row["strike"]), float(row["tte"]), r, q, float(row["iv"]))

    df["vega"] = df.apply(_vega, axis=1).fillna(0)

    return df.reset_index(drop=True)


# ── Objective function ─────────────────────────────────────────────────────

def objective(
    params: np.ndarray,
    strikes: np.ndarray,
    ttes: np.ndarray,
    types: np.ndarray,
    market_ivs: np.ndarray,
    vegas: np.ndarray,
    spot: float,
    r: float,
    q: float,
) -> float:
    """
    Vega-weighted relative IVRMSE + Feller penalty.

    Returns a scalar objective value (lower is better).
    """
    kappa, theta, xi, rho, v0 = params

    # Feller soft penalty: 2κθ > ξ²
    feller_violation = max(0.0, xi**2 - 2 * kappa * theta)
    feller_penalty   = FELLER_PENALTY_LAMBDA * feller_violation**2

    # Heston model prices
    try:
        model_prices = heston_price_batch(
            S=spot, strikes=strikes, ttes=ttes, r=r, q=q,
            kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0,
            option_types=types,
        )
    except Exception:
        return 1e6 + feller_penalty

    # Convert model prices → model IVs (vectorised via py_vollib_vectorized)
    from py_vollib_vectorized import vectorized_implied_volatility as _viv
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_ivs = _viv(
            price    = model_prices,
            S        = float(spot),
            K        = strikes,
            t        = ttes,
            r        = float(r),
            flag     = types,
            q        = float(q),
            on_error = "ignore",
            model    = "black_scholes_merton",
            return_as= "numpy",
        ).astype(float)
    model_ivs[model_ivs <= 0] = float("nan")   # library returns 0.0 on failure

    valid = np.isfinite(model_ivs) & np.isfinite(market_ivs) & (market_ivs > 0)
    if valid.sum() < 3:
        return 1e6 + feller_penalty

    # Vega-weighted relative squared errors
    w = vegas[valid]
    w = w / (w.sum() + 1e-12)

    rel_errors = ((model_ivs[valid] - market_ivs[valid]) / market_ivs[valid])**2
    ivrmse_weighted = float(np.sqrt(np.sum(w * rel_errors)))

    return ivrmse_weighted + feller_penalty


# ── Two-stage calibration ─────────────────────────────────────────────────

def calibrate_heston(
    contracts,        # pd.DataFrame: strike, tte, type, iv, vega, spot, rf_rate, div_yield
    S: float,
    r: float,
    q: float,
    prev_params: Optional[dict] = None,
) -> dict:
    """
    Calibrate Heston parameters to market IVs.

    Stage 1: differential_evolution (global search)
    Stage 2: Levenberg-Marquardt local refinement

    Parameters
    ----------
    contracts   : calibration universe (from select_calibration_contracts)
    S, r, q     : spot, risk-free rate, dividend yield
    prev_params : prior-day parameters for warm-start (optional)

    Returns
    -------
    dict with keys: kappa, theta, xi, rho, v0, ivrmse, converged, n_contracts
    """
    import pandas as pd

    n = len(contracts)
    if n < 8:
        return {"converged": False, "n_contracts": n, "error": "insufficient contracts"}

    strikes    = contracts["strike"].values.astype(float)
    ttes       = contracts["tte"].values.astype(float)
    types      = contracts["type"].values
    market_ivs = contracts["iv"].values.astype(float)
    vegas      = contracts.get("vega", pd.Series(np.ones(n))).values.astype(float)

    obj_args = (strikes, ttes, types, market_ivs, vegas, S, r, q)

    # ── Stage 1: global search ─────────────────────────────────────────────
    # Seed initial population around prev_params if available
    init_population = None
    if prev_params is not None:
        center = np.array([
            prev_params.get("kappa", 2.0),
            prev_params.get("theta", 0.04),
            prev_params.get("xi",    0.3),
            prev_params.get("rho",  -0.7),
            prev_params.get("v0",    0.04),
        ])
        # Perturb center by ±10% to create seed population
        np.random.seed(42)
        pop = center + center * np.random.uniform(-0.1, 0.1, (DE_POPSIZE, 5))
        # Clip to bounds
        for j, (lo, hi) in enumerate(_BOUNDS_LIST):
            pop[:, j] = np.clip(pop[:, j], lo, hi)
        init_population = pop

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        de_result = differential_evolution(
            func        = objective,
            bounds      = _BOUNDS_LIST,
            args        = obj_args,
            maxiter     = DE_MAXITER,
            popsize     = DE_POPSIZE if init_population is None else 1,
            tol         = DE_TOL,
            seed        = 42,
            init        = init_population if init_population is not None else "latinhypercube",
            mutation    = (0.5, 1.0),
            recombination = 0.9,
            polish      = False,
            workers     = 1,    # single-threaded; parallelise at ticker level
        )

    x0 = de_result.x

    # ── Stage 2: local refinement (LM) ────────────────────────────────────
    from py_vollib_vectorized import vectorized_implied_volatility as _viv_lm

    def residuals(params: np.ndarray) -> np.ndarray:
        """Return per-contract IV residuals for LM."""
        kappa, theta, xi, rho, v0 = params
        try:
            model_prices = heston_price_batch(
                S=S, strikes=strikes, ttes=ttes, r=r, q=q,
                kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0,
                option_types=types,
            )
        except Exception:
            return np.ones(n) * 1e3

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_ivs = _viv_lm(
                price=model_prices, S=float(S), K=strikes, t=ttes, r=float(r),
                flag=types, q=float(q), on_error="ignore",
                model="black_scholes_merton", return_as="numpy",
            ).astype(float)
        model_ivs[model_ivs <= 0] = float("nan")
        valid = np.isfinite(model_ivs) & np.isfinite(market_ivs)
        res = np.where(valid, (model_ivs - market_ivs) / (market_ivs + 1e-8), 1.0)
        return res * np.sqrt(vegas / (vegas.sum() + 1e-12) * n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lm_result = least_squares(
            fun    = residuals,
            x0     = x0,
            bounds = (
                [b[0] for b in _BOUNDS_LIST],
                [b[1] for b in _BOUNDS_LIST],
            ),
            method = "trf",
            ftol   = 1e-8,
            xtol   = 1e-8,
            max_nfev = 500,
        )

    final_params = lm_result.x
    kappa, theta, xi, rho, v0 = final_params

    # Compute final IVRMSE (unweighted, for reporting)
    final_obj = objective(final_params, *obj_args)
    ivrmse    = _compute_ivrmse(final_params, strikes, ttes, types, market_ivs, S, r, q)

    converged = lm_result.success and ivrmse < HESTON_IVRMSE_THRESHOLD

    return {
        "kappa":      float(kappa),
        "theta":      float(theta),
        "xi":         float(xi),
        "rho":        float(rho),
        "v0":         float(v0),
        "ivrmse":     float(ivrmse),
        "converged":  bool(converged),
        "n_contracts": int(n),
        "feller_ok":  bool(2 * kappa * theta > xi**2),
    }


def _compute_ivrmse(
    params: np.ndarray,
    strikes: np.ndarray,
    ttes: np.ndarray,
    types: np.ndarray,
    market_ivs: np.ndarray,
    S: float,
    r: float,
    q: float,
) -> float:
    """Unweighted IVRMSE for reporting."""
    from py_vollib_vectorized import vectorized_implied_volatility as _viv_r
    kappa, theta, xi, rho, v0 = params
    try:
        model_prices = heston_price_batch(
            S=S, strikes=strikes, ttes=ttes, r=r, q=q,
            kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0,
            option_types=types,
        )
        model_ivs = _viv_r(
            price=model_prices, S=float(S), K=strikes, t=ttes, r=float(r),
            flag=types, q=float(q), on_error="ignore",
            model="black_scholes_merton", return_as="numpy",
        ).astype(float)
        model_ivs[model_ivs <= 0] = float("nan")
        valid = np.isfinite(model_ivs) & np.isfinite(market_ivs)
        if valid.sum() == 0:
            return float("nan")
        return float(np.sqrt(np.mean((model_ivs[valid] - market_ivs[valid])**2)))
    except Exception:
        return float("nan")
