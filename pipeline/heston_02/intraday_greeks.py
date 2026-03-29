"""
Phase 3.3 — Intraday Greeks Computation

Per 15-min bar:
  - Update v0 using VIX proxy: v0_t = η0 + η1 × VIX_t²
  - Compute Heston delta and gamma (finite difference)
  - Compute BS delta and gamma (analytical) for comparison
  - Store per-contract Greeks

Output: data/intraday_greeks.parquet
  columns: timestamp, ticker, strike, expiry_date, tte,
           heston_delta, heston_gamma, bs_delta, bs_gamma, iv, spot

Run:
    python -m pipeline.02_heston.intraday_greeks
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

from config import (
    TICKERS, DATA_DIR,
    BACKTEST_START, BACKTEST_END,
    V0_ETA0, V0_ETA1,
)
from pipeline.heston_02.heston_pricer import heston_price
from pipeline.utils.greeks import delta as bs_delta_fn, gamma as bs_gamma_fn
from pipeline.utils.greeks import heston_delta_fd, heston_gamma_fd

warnings.filterwarnings("ignore")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FD_EPS_PCT = 0.01   # finite difference step: 1% of spot


# ── v0 proxy fitting ──────────────────────────────────────────────────────

def fit_v0_proxy(
    heston_params: pd.DataFrame,
    external_df: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    """
    Fit v0_t = η0 + η1 × VIX_t² per ticker via OLS.

    Parameters
    ----------
    heston_params : output of run_calibration (date, ticker, v0)
    external_df   : has date, vix_close

    Returns
    -------
    dict mapping ticker → (eta0, eta1)
    """
    ext = external_df[["date", "vix_close"]].copy()
    ext["date"] = pd.to_datetime(ext["date"]).dt.date
    ext["vix_sq"] = (ext["vix_close"] / 100.0)**2   # convert % to decimal, square

    heston_params = heston_params.copy()
    heston_params["date"] = pd.to_datetime(heston_params["date"]).dt.date

    merged = heston_params.merge(ext, on="date", how="inner")
    merged = merged[merged["converged"] == True].dropna(subset=["v0", "vix_sq"])

    coefs = {}
    for ticker, grp in merged.groupby("ticker"):
        if len(grp) < 10:
            coefs[ticker] = (V0_ETA0, V0_ETA1)
            continue
        slope, intercept, r_val, p_val, _ = linregress(grp["vix_sq"], grp["v0"])
        eta0 = max(intercept, 0.001)
        eta1 = max(slope, 0.0)
        coefs[ticker] = (eta0, eta1)
        print(f"  {ticker}: η0={eta0:.4f}  η1={eta1:.4f}  R²={r_val**2:.3f}")

    return coefs


def v0_intraday(vix: float, eta0: float, eta1: float) -> float:
    """Intraday v0 proxy: v0_t = η0 + η1 × (VIX_t / 100)²"""
    return max(eta0 + eta1 * (vix / 100.0)**2, 0.001)


# ── Per-bar Greeks ────────────────────────────────────────────────────────

def compute_greeks_for_bar(
    contracts: pd.DataFrame,
    spot: float,
    r: float,
    q: float,
    heston_params: dict,
) -> pd.DataFrame:
    """
    Compute Heston and BS Greeks for all contracts in a single 15-min bar.

    Parameters
    ----------
    contracts     : option rows with columns strike, tte, type, iv
    spot          : underlying price
    r, q          : rates
    heston_params : {kappa, theta, xi, rho, v0}

    Returns
    -------
    contracts DataFrame with added columns:
      heston_delta, heston_gamma, bs_delta, bs_gamma
    """
    df = contracts.copy()
    if df.empty:
        for col in ["heston_delta", "heston_gamma", "bs_delta", "bs_gamma"]:
            df[col] = np.nan
        return df

    kappa = heston_params["kappa"]
    theta = heston_params["theta"]
    xi    = heston_params["xi"]
    rho   = heston_params["rho"]
    v0    = heston_params["v0"]

    def _heston_pricer(S, K, T, r_, q_, params):
        return heston_price(S, K, T, r_, q_,
                            params["kappa"], params["theta"],
                            params["xi"], params["rho"], params["v0"],
                            option_type="c")   # delta sign corrected below

    h_deltas = []
    h_gammas = []
    b_deltas = []
    b_gammas = []

    for _, row in df.iterrows():
        K   = float(row["strike"])
        T   = float(row["tte"])
        ot  = str(row["type"])
        iv  = float(row["iv"]) if pd.notna(row.get("iv")) else 0.2

        # ── Heston Greeks (finite difference) ────────────────────────────
        def pricer_call(S, K=K, T=T, r_=r, q_=q, p=heston_params):
            return heston_price(S, K, T, r_, q_,
                                p["kappa"], p["theta"], p["xi"],
                                p["rho"], p["v0"], option_type=ot)

        try:
            hd = heston_delta_fd(pricer_call, spot, K, T, r, q, heston_params, FD_EPS_PCT)
            hg = heston_gamma_fd(pricer_call, spot, K, T, r, q, heston_params, FD_EPS_PCT)
        except Exception:
            hd, hg = np.nan, np.nan

        # ── BS Greeks (analytical) ────────────────────────────────────────
        bd = bs_delta_fn(spot, K, T, r, q, iv, ot)
        bg = bs_gamma_fn(spot, K, T, r, q, iv)

        h_deltas.append(hd)
        h_gammas.append(hg)
        b_deltas.append(bd)
        b_gammas.append(bg)

    df["heston_delta"] = h_deltas
    df["heston_gamma"] = h_gammas
    df["bs_delta"]     = b_deltas
    df["bs_gamma"]     = b_gammas

    return df


# ── Full intraday Greeks pipeline ─────────────────────────────────────────

def run_intraday_greeks(
    start: str = BACKTEST_START,
    end: str   = BACKTEST_END,
    tickers: list[str] = TICKERS,
    output_path: Path = DATA_DIR / "intraday_greeks.parquet",
) -> pd.DataFrame:
    """
    For each (ticker, timestamp) bar, compute Heston + BS Greeks
    using calibrated daily params + intraday v0 proxy.
    """
    print("=== Phase 3.3: Intraday Greeks ===")

    # Load dependencies
    iv_df       = pd.read_parquet(DATA_DIR / "iv_data.parquet")
    heston_df   = pd.read_parquet(DATA_DIR / "heston_params.parquet")
    external_df = pd.read_parquet(DATA_DIR / "external_data.parquet")
    stock_df    = pd.read_parquet(DATA_DIR / "stock_bars.parquet")

    iv_df["report_time"]       = pd.to_datetime(iv_df["report_time"])
    stock_df["timestamp"]      = pd.to_datetime(stock_df["timestamp"])
    external_df["date"]        = pd.to_datetime(external_df["date"]).dt.date
    heston_df["date"]          = pd.to_datetime(heston_df["date"]).dt.date

    # Fit v0 proxy coefficients
    print("  Fitting v0 proxy (VIX → v0) …")
    v0_coefs = fit_v0_proxy(heston_df, external_df)

    results = []

    for ticker in tickers:
        print(f"  Processing {ticker} …")

        iv_t       = iv_df[iv_df["ticker"] == ticker].copy()
        stock_t    = stock_df[stock_df["ticker"] == ticker].sort_values("timestamp")
        heston_t   = heston_df[heston_df["ticker"] == ticker].set_index("date")
        eta0, eta1 = v0_coefs.get(ticker, (V0_ETA0, V0_ETA1))

        for ts, grp in iv_t.groupby("report_time"):
            ts = pd.Timestamp(ts)
            date_key = ts.date()

            if str(date_key) < start or str(date_key) > end:
                continue

            # Look up calibrated params for this date
            if date_key not in heston_t.index:
                continue
            day_params = heston_t.loc[date_key]
            if not day_params.get("converged", False):
                continue

            heston_params = {
                "kappa": float(day_params["kappa"]),
                "theta": float(day_params["theta"]),
                "xi":    float(day_params["xi"]),
                "rho":   float(day_params["rho"]),
                "v0":    float(day_params["v0"]),
            }

            # Intraday v0 update using VIX
            ext_row = external_df[external_df["date"] == date_key]
            if not ext_row.empty:
                vix_t = float(ext_row["vix_close"].iloc[0])
                heston_params["v0"] = v0_intraday(vix_t, eta0, eta1)

            # Look up spot price
            idx = stock_t["timestamp"].searchsorted(ts, side="right") - 1
            if idx < 0:
                continue
            spot = float(stock_t.iloc[idx]["close"])

            # Look up r, q
            if not ext_row.empty:
                r = float(ext_row["rf_rate"].iloc[0])
                q_col = f"div_yield_{ticker}"
                q = float(ext_row[q_col].iloc[0]) if q_col in ext_row.columns else 0.0
            else:
                r, q = 0.04, 0.0

            # Compute Greeks
            bar_greeks = compute_greeks_for_bar(grp, spot, r, q, heston_params)
            bar_greeks["spot"]      = spot
            bar_greeks["timestamp"] = ts
            results.append(bar_greeks)

    if not results:
        print("  No Greeks computed.")
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)

    save_cols = [
        "timestamp", "ticker", "strike", "tte", "type",
        "expiry_date" if "expiry_date" in out.columns else "tte",
        "iv", "spot",
        "heston_delta", "heston_gamma", "bs_delta", "bs_gamma",
        "bid", "ask", "mid", "volume", "oi",
    ]
    save_cols = [c for c in save_cols if c in out.columns]
    save_cols = list(dict.fromkeys(save_cols))   # deduplicate

    out[save_cols].to_parquet(output_path, index=False)
    print(f"  Saved {len(out):,} Greek rows → {output_path}")

    return out


if __name__ == "__main__":
    run_intraday_greeks()
