"""
Analytical Black-Scholes Greeks.

All functions accept scalar inputs and return floats.
Vectorised usage: wrap with np.vectorize or apply to DataFrame columns.
"""
import math
from scipy.stats import norm

from pipeline.utils.bs_model import _d1, _d2, bs_price


# ── First-order Greeks ─────────────────────────────────────────────────────

def delta(
    S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str
) -> float:
    """
    BS delta: ∂Price/∂S.
      Call: e^{-qT} N(d1)
      Put : e^{-qT} (N(d1) - 1)
    """
    if T <= 0 or sigma <= 0:
        if option_type == "c":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    d1 = _d1(S, K, T, r, q, sigma)
    if option_type == "c":
        return math.exp(-q * T) * norm.cdf(d1)
    return math.exp(-q * T) * (norm.cdf(d1) - 1.0)


def gamma(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """
    BS gamma: ∂²Price/∂S² (same for calls and puts).
    Γ = e^{-qT} N'(d1) / (S σ √T)
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, q, sigma)
    return math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))


def vega(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """
    BS vega: ∂Price/∂σ.
    ν = S e^{-qT} N'(d1) √T
    Returns vega per 1.0 change in sigma (not per 1% point).
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)


def theta(
    S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str
) -> float:
    """
    BS theta: ∂Price/∂T (per calendar year; divide by 365 for per-day).
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    sqrt_t = math.sqrt(T)
    common = -(S * math.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * sqrt_t)

    if option_type == "c":
        return (
            common
            + q * S * math.exp(-q * T) * norm.cdf(d1)
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        )
    return (
        common
        - q * S * math.exp(-q * T) * norm.cdf(-d1)
        + r * K * math.exp(-r * T) * norm.cdf(-d2)
    )


def rho(
    S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str
) -> float:
    """BS rho: ∂Price/∂r."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d2 = _d2(S, K, T, r, q, sigma)
    if option_type == "c":
        return K * T * math.exp(-r * T) * norm.cdf(d2)
    return -K * T * math.exp(-r * T) * norm.cdf(-d2)


# ── Heston finite-difference Greeks ───────────────────────────────────────

def heston_delta_fd(
    pricer_fn,   # callable(S, K, T, r, q, params) -> float
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: dict,
    eps_pct: float = 0.01,
) -> float:
    """
    Central finite-difference Heston delta.
    Δ = (P(S+ε) - P(S-ε)) / (2ε)   where ε = S × eps_pct
    """
    eps = S * eps_pct
    p_up   = pricer_fn(S + eps, K, T, r, q, params)
    p_down = pricer_fn(S - eps, K, T, r, q, params)
    return (p_up - p_down) / (2.0 * eps)


def heston_gamma_fd(
    pricer_fn,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: dict,
    eps_pct: float = 0.01,
) -> float:
    """
    Central finite-difference Heston gamma.
    Γ = (P(S+ε) - 2P(S) + P(S-ε)) / ε²
    """
    eps = S * eps_pct
    p_up   = pricer_fn(S + eps, K, T, r, q, params)
    p_mid  = pricer_fn(S,       K, T, r, q, params)
    p_down = pricer_fn(S - eps, K, T, r, q, params)
    return (p_up - 2.0 * p_mid + p_down) / (eps**2)
