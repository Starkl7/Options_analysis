"""
Phase 6.4-6.5 — Performance Metrics & Walk-Forward Validation

Functions:
  - compute_metrics(daily_pnl_df)  → per-strategy Sharpe, drawdown, win rate, etc.
  - portfolio_metrics(trade_log)   → aggregate + correlation matrix
  - permutation_test(daily_pnl_df) → shuffle test for statistical significance
  - spread_sensitivity(...)        → P&L degradation at 2×/3× spreads

Run:
    python -m pipeline.04_backtest.metrics
"""
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS_DIR

warnings.filterwarnings("ignore")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS_PER_YEAR = 252
N_PERMUTATIONS = 500   # for permutation test


# ── Core metrics ──────────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, annualise: bool = True) -> float:
    """Annualised Sharpe (assume daily returns; mean / std × √252)."""
    if returns.std() < 1e-10 or len(returns) < 5:
        return 0.0
    sr = returns.mean() / returns.std()
    return float(sr * np.sqrt(TRADING_DAYS_PER_YEAR) if annualise else sr)


def max_drawdown(cum_pnl: pd.Series) -> float:
    """Maximum peak-to-trough drawdown in same units as cum_pnl."""
    roll_max = cum_pnl.cummax()
    dd       = cum_pnl - roll_max
    return float(dd.min())


def win_rate(pnl_series: pd.Series) -> float:
    if len(pnl_series) == 0:
        return 0.0
    return float((pnl_series > 0).mean())


def avg_holding_period(trade_log: pd.DataFrame, strategy: str) -> float:
    """Average hours between entry and exit for a strategy."""
    sub = trade_log[trade_log["strategy"] == strategy].copy()
    sub["entry_time"] = pd.to_datetime(sub["entry_time"])
    sub["exit_time"]  = pd.to_datetime(sub["exit_time"])
    if sub.empty or sub["exit_time"].isna().all():
        return 0.0
    durations = (sub["exit_time"] - sub["entry_time"]).dt.total_seconds() / 3600
    return float(durations.mean())


def rolling_sharpe(daily_returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling window Sharpe (not annualised — used for intra-year monitoring)."""
    roll_mean = daily_returns.rolling(window, min_periods=window // 2).mean()
    roll_std  = daily_returns.rolling(window, min_periods=window // 2).std()
    return (roll_mean / (roll_std + 1e-10)) * np.sqrt(TRADING_DAYS_PER_YEAR)


# ── Strategy-level metrics ─────────────────────────────────────────────────

def compute_strategy_metrics(
    trade_log: pd.DataFrame,
    daily_pnl: pd.DataFrame,
    strategy_id: str,
    portfolio_value: float = 1_000_000,
) -> dict:
    """Compute all metrics for a single strategy."""
    trades = trade_log[trade_log["strategy"] == strategy_id].copy()

    if "date" not in daily_pnl.columns:
        daily_pnl = daily_pnl.copy()
        daily_pnl["date"] = pd.to_datetime(daily_pnl["date"]) if "date" in daily_pnl.columns else pd.Series()

    pnl_col = f"{strategy_id.lower()}_pnl"
    if pnl_col not in daily_pnl.columns:
        pnl_col = "total_pnl"

    daily_ret = daily_pnl[pnl_col] / portfolio_value if pnl_col in daily_pnl.columns else pd.Series()

    return {
        "strategy":            strategy_id,
        "n_trades":            len(trades),
        "sharpe":              sharpe_ratio(daily_ret),
        "max_drawdown":        max_drawdown((daily_ret).cumsum() * portfolio_value),
        "max_drawdown_pct":    max_drawdown(daily_ret.cumsum()),
        "win_rate":            win_rate(trades["pnl_net"]) if "pnl_net" in trades.columns else 0.0,
        "avg_pnl_per_trade":   float(trades["pnl_net"].mean()) if "pnl_net" in trades.columns and len(trades) else 0.0,
        "total_pnl":           float(trades["pnl_net"].sum()) if "pnl_net" in trades.columns else 0.0,
        "total_cost":          float(trades["cost"].sum()) if "cost" in trades.columns else 0.0,
        "avg_hold_hours":      avg_holding_period(trade_log, strategy_id),
        "trades_per_day":      len(trades) / max(1, daily_pnl["date"].nunique()) if "date" in daily_pnl.columns else 0.0,
    }


# ── Portfolio-level metrics ────────────────────────────────────────────────

def compute_portfolio_metrics(
    daily_pnl: pd.DataFrame,
    trade_log: pd.DataFrame,
    portfolio_value: float = 1_000_000,
    external_df: pd.DataFrame | None = None,
) -> dict:
    """
    Aggregate metrics across all strategies.
    Includes VIX-regime split and inter-strategy correlations.
    trade_log is filtered to the date range of daily_pnl before computing
    per-strategy trade counts and win rates.
    """
    daily_pnl = daily_pnl.copy()
    daily_pnl["date"] = pd.to_datetime(daily_pnl["date"])

    # Filter trade_log to match the date window of daily_pnl
    if not daily_pnl.empty and "entry_time" in trade_log.columns and not trade_log.empty:
        min_date = daily_pnl["date"].min()
        max_date = daily_pnl["date"].max()
        entry_dates = pd.to_datetime(trade_log["entry_time"]).dt.normalize()
        trade_log = trade_log[(entry_dates >= min_date) & (entry_dates <= max_date)].copy()

    strategy_cols = [c for c in ["s1_pnl", "s4_pnl"] if c in daily_pnl.columns]
    if not strategy_cols and "total_pnl" in daily_pnl.columns:
        strategy_cols = ["total_pnl"]

    total_daily = daily_pnl[strategy_cols].sum(axis=1)
    daily_ret   = total_daily / portfolio_value

    metrics = {
        "aggregate_sharpe":   sharpe_ratio(daily_ret),
        "aggregate_max_dd":   max_drawdown(daily_ret.cumsum()),
        "aggregate_total_pnl": float(total_daily.sum()),
        "n_trading_days":     len(daily_pnl),
    }

    # Inter-strategy correlation
    if len(strategy_cols) >= 2:
        corr = daily_pnl[strategy_cols].corr()
        metrics["strategy_correlation"] = corr.to_dict()

    # VIX regime split
    if external_df is not None and "vix_close" in external_df.columns:
        ext = external_df[["date", "vix_close"]].copy()
        ext["date"] = pd.to_datetime(ext["date"])
        merged = daily_pnl.merge(ext, on="date", how="left")

        high_vix = merged[merged["vix_close"] > 20]
        low_vix  = merged[merged["vix_close"] < 15]

        for label, sub in [("high_vix", high_vix), ("low_vix", low_vix)]:
            if sub.empty:
                continue
            sub_ret = sub[strategy_cols].sum(axis=1) / portfolio_value
            metrics[f"{label}_sharpe"] = sharpe_ratio(sub_ret)
            metrics[f"{label}_n_days"] = len(sub)

    # Per-strategy breakdown
    per_strategy = {}
    for s in ["S1", "S4"]:
        per_strategy[s] = compute_strategy_metrics(trade_log, daily_pnl, s, portfolio_value)
    metrics["per_strategy"] = per_strategy

    return metrics


# ── Walk-forward split ─────────────────────────────────────────────────────

def walk_forward_split(
    daily_pnl: pd.DataFrame,
    insample_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split daily_pnl into in-sample and OOS halves."""
    daily_pnl = daily_pnl.copy()
    daily_pnl["date"] = pd.to_datetime(daily_pnl["date"])
    cutoff = pd.Timestamp(insample_end)
    in_sample = daily_pnl[daily_pnl["date"] <= cutoff]
    oos       = daily_pnl[daily_pnl["date"] >  cutoff]
    return in_sample, oos


# ── Sanity checks ─────────────────────────────────────────────────────────

def permutation_test(
    daily_pnl: pd.DataFrame,
    portfolio_value: float = 1_000_000,
    n_perms: int = N_PERMUTATIONS,
    trade_log: pd.DataFrame | None = None,
) -> dict:
    """
    Signal-shuffling permutation test.

    Null hypothesis: signal direction has no predictive value.
    Method: for each permutation, randomly flip the sign of every trade's P&L
    (equivalent to randomly assigning long/short at the same entry times).
    The actual Sharpe should be significantly higher than the null distribution
    if the signals have real predictive power.

    Note: shuffling daily returns does NOT work (Sharpe is invariant under
    permutation of its inputs — mean and std are order-independent).
    """
    total = daily_pnl[[c for c in ["s1_pnl", "s4_pnl", "total_pnl"] if c in daily_pnl.columns]].sum(axis=1)
    actual_sharpe = sharpe_ratio(total / portfolio_value)

    # Need trade-level P&L to do signal shuffling
    if trade_log is None or trade_log.empty or "pnl_net" not in trade_log.columns:
        return {
            "actual_sharpe":    actual_sharpe,
            "perm_mean_sharpe": float("nan"),
            "perm_std_sharpe":  float("nan"),
            "p_value":          float("nan"),
            "significant":      False,
            "note":             "trade_log required for permutation test",
        }

    # Build per-trade daily P&L with random sign flips
    tl = trade_log.copy()
    tl["entry_date"] = pd.to_datetime(tl["entry_time"]).dt.normalize()
    dates = daily_pnl["date"].values
    n_trades = len(tl)

    rng = np.random.default_rng(seed=42)
    perm_sharpes = []
    for _ in range(n_perms):
        # Randomly flip each trade's P&L sign (null: direction is random)
        signs = rng.choice([-1, 1], size=n_trades)
        perm_pnl = tl["pnl_net"].values * signs

        # Aggregate to daily P&L
        tl_perm = tl.copy()
        tl_perm["pnl_perm"] = perm_pnl
        daily_agg = (
            tl_perm.groupby("entry_date")["pnl_perm"].sum()
            .reindex(pd.to_datetime(dates), fill_value=0.0)
        )
        perm_sharpes.append(sharpe_ratio(daily_agg / portfolio_value))

    perm_sharpes = np.array(perm_sharpes)
    p_value = float((perm_sharpes >= actual_sharpe).mean())

    return {
        "actual_sharpe":    actual_sharpe,
        "perm_mean_sharpe": float(perm_sharpes.mean()),
        "perm_std_sharpe":  float(perm_sharpes.std()),
        "p_value":          p_value,
        "significant":      p_value < 0.05,
    }


def spread_sensitivity(
    trade_log: pd.DataFrame,
    multipliers: list[float] | None = None,
) -> pd.DataFrame:
    """
    Recompute net P&L assuming 2× and 3× option bid-ask spreads.
    Assumes cost column represents bid-ask costs (option half-spread × quantity).
    """
    if multipliers is None:
        multipliers = [1.0, 2.0, 3.0]

    results = []
    for mult in multipliers:
        adj = trade_log.copy()
        if "cost" in adj.columns and "pnl_gross" in adj.columns:
            adj["pnl_net_adj"] = adj["pnl_gross"] - adj["cost"] * mult
        elif "pnl_net" in adj.columns:
            adj["pnl_net_adj"] = adj["pnl_net"]
        else:
            adj["pnl_net_adj"] = 0.0

        results.append({
            "spread_multiplier":  mult,
            "total_pnl":          float(adj["pnl_net_adj"].sum()),
            "win_rate":           win_rate(adj["pnl_net_adj"]),
            "n_profitable_trades": int((adj["pnl_net_adj"] > 0).sum()),
        })

    return pd.DataFrame(results)


# ── Main runner ───────────────────────────────────────────────────────────

def run_metrics(
    trade_log_path: Path = RESULTS_DIR / "trade_log_net.parquet",
    daily_pnl_path: Path = RESULTS_DIR / "daily_pnl_net.parquet",
    portfolio_value: float = 1_000_000,
) -> dict:
    from config import INSAMPLE_END

    trade_log = pd.read_parquet(trade_log_path)
    daily_pnl = pd.read_parquet(daily_pnl_path)

    try:
        external_df = pd.read_parquet(RESULTS_DIR.parent / "data" / "external_data.parquet")
    except Exception:
        external_df = None

    # In-sample vs OOS
    insample, oos = walk_forward_split(daily_pnl, INSAMPLE_END)

    print("=== Performance Metrics ===")
    print("\n  In-sample:")
    is_metrics = compute_portfolio_metrics(insample, trade_log, portfolio_value, external_df)
    _print_metrics(is_metrics)

    print("\n  Out-of-sample:")
    oos_metrics = compute_portfolio_metrics(oos, trade_log, portfolio_value, external_df)
    _print_metrics(oos_metrics)

    # Permutation test (signal-shuffling)
    print("\n  Permutation test (full period):")
    perm = permutation_test(daily_pnl, portfolio_value, trade_log=trade_log)
    print(f"    Actual Sharpe: {perm['actual_sharpe']:.3f}")
    if not math.isnan(perm.get("perm_mean_sharpe", float("nan"))):
        print(f"    Perm mean:     {perm['perm_mean_sharpe']:.3f} ± {perm['perm_std_sharpe']:.3f}")
        print(f"    p-value:       {perm['p_value']:.3f}  {'✓ Significant' if perm['significant'] else '✗ Not significant'}")
    else:
        print(f"    {perm.get('note', 'insufficient data')}")

    # Spread sensitivity
    print("\n  Spread sensitivity:")
    spread_df = spread_sensitivity(trade_log)
    print(spread_df.to_string(index=False))

    # Rolling Sharpe
    total_ret = daily_pnl[[c for c in ["s1_pnl", "s4_pnl", "total_pnl"] if c in daily_pnl.columns]].sum(axis=1) / portfolio_value
    roll_sh = rolling_sharpe(total_ret)

    out = {
        "insample":         is_metrics,
        "oos":              oos_metrics,
        "permutation_test": perm,
        "spread_sensitivity": spread_df.to_dict("records"),
        "rolling_sharpe":   roll_sh.to_dict(),
    }

    # Save rolling Sharpe
    roll_df = pd.DataFrame({"date": daily_pnl["date"], "rolling_sharpe": roll_sh.values})
    roll_df.to_parquet(RESULTS_DIR / "rolling_sharpe.parquet", index=False)

    return out


def _print_metrics(m: dict) -> None:
    print(f"    Aggregate Sharpe:  {m.get('aggregate_sharpe', 0):.3f}")
    print(f"    Max Drawdown:      {m.get('aggregate_max_dd', 0)*100:.2f}%")
    print(f"    Total P&L:         ${m.get('aggregate_total_pnl', 0):,.0f}")
    if "high_vix_sharpe" in m:
        print(f"    High-VIX Sharpe:   {m['high_vix_sharpe']:.3f}  ({m.get('high_vix_n_days',0)} days)")
    if "low_vix_sharpe" in m:
        print(f"    Low-VIX Sharpe:    {m['low_vix_sharpe']:.3f}  ({m.get('low_vix_n_days',0)} days)")
    per = m.get("per_strategy", {})
    for strat, sm in per.items():
        print(f"    {strat}:  Sharpe={sm.get('sharpe',0):.3f}  WR={sm.get('win_rate',0)*100:.1f}%  "
              f"Trades={sm.get('n_trades',0)}")


if __name__ == "__main__":
    run_metrics()
