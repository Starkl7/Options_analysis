"""
Black-Scholes pricing and IV solver.

Pricing and Greeks functions work on scalars.
implied_vol() uses Jäckel's 'Let's Be Rational' algorithm (py_vollib_vectorized)
instead of Brent's iterative root-find — O(1) closed-form rational approximation,
~10-100× faster and more numerically robust near expiry and deep OTM.
"""
import math
import numpy as np
from scipy.stats import norm
from py_vollib_vectorized import vectorized_implied_volatility as _viv_vectorized


# ── Core BS price ──────────────────────────────────────────────────────────

def _d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return _d1(S, K, T, r, q, sigma) - sigma * math.sqrt(T)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str,   # 'c' or 'p'
) -> float:
    """
    Black-Scholes option price (continuous dividend yield q).

    Parameters
    ----------
    S            : spot price
    K            : strike
    T            : time to expiry in years
    r            : continuous risk-free rate
    q            : continuous dividend yield
    sigma        : implied / assumed vol (annualised)
    option_type  : 'c' for call, 'p' for put

    Returns
    -------
    option price (float)
    """
    if T <= 0 or sigma <= 0:
        # Return intrinsic value at expiry / zero-vol limit
        if option_type == "c":
            return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
        return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)

    d1 = _d1(S, K, T, r, q, sigma)
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "c":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    # put
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Vega: ∂Price/∂sigma."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)


# ── IV solver (Jäckel's Let's Be Rational) ────────────────────────────────

def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    iv_min: float = 0.001,
    iv_max: float = 5.0,
    tol: float = 1e-6,    # kept for API compatibility; not used by Jäckel solver
    maxiter: int = 100,   # kept for API compatibility; not used by Jäckel solver
) -> float:
    """
    Invert the BSM formula for implied vol using Jäckel's 'Let's Be Rational'
    algorithm (py_vollib_vectorized back-end).

    This is a direct O(1) rational-function inversion — no iterative root-find.
    Replaces Brent's method; ~10-100× faster and more robust near expiry/OTM.

    Returns
    -------
    IV (float) or NaN if the solver cannot find a valid solution or the result
    falls outside [iv_min, iv_max].
    """
    if T <= 0 or market_price <= 0:
        return float("nan")

    iv = _viv_vectorized(
        price = np.array([float(market_price)]),
        S     = float(S),
        K     = np.array([float(K)]),
        t     = np.array([float(T)]),
        r     = float(r),
        flag  = np.array([str(option_type)]),
        q     = float(q),
        on_error = "ignore",        # returns 0.0 on failure instead of raising
        model    = "black_scholes_merton",
        return_as = "numpy",
    )[0]

    # Library returns 0.0 on any failure; map out-of-bounds results to NaN
    if iv <= 0.0 or iv < iv_min or iv > iv_max:
        return float("nan")
    return float(iv)


# ── Forward price ──────────────────────────────────────────────────────────

def forward_price(S: float, r: float, q: float, T: float) -> float:
    """F = S * e^{(r - q) * T}"""
    return S * math.exp((r - q) * T)
