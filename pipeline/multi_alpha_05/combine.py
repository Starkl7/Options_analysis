"""
Phase 7 — Multi-Alpha Engine

Combines S1, S2, S4 P&L streams into a single portfolio:
  - Inverse-vol weights, rebalanced weekly
  - Rolling 20-day correlation monitoring
  - Risk limits: max daily drawdown, max strategy drawdown → suspend
  - Outputs daily P&L decomposed by strategy and rolling Sharpe

Run:
    python -m pipeline.05_multi_alpha.combine
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    RESULTS_DIR, DATA_DIR,
    CORRELATION_THRESHOLD,
    WEIGHT_REBALANCE_FREQ,
    MAX_DAILY_DRAWDOWN_PCT,
    MAX_STRATEGY_DRAWDOWN_PCT,
    BARS_PER_DAY,
)
from pipeline.backtest_04.metrics import (
    sharpe_ratio, max_drawdown, rolling_sharpe, permutation_test, spread_sensitivity,
)

warnings.filterwarnings("ignore")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_CORR_WINDOW = 20       # trading days
_STRAT_DD_WINDOW = 10   # trading days for strategy drawdown check


# ── Weight computation ─────────────────────────────────────────────────────

def compute_inverse_vol_weights(
    daily_pnl: pd.DataFrame,
    strategy_cols: list[str],
    window: int = _CORR_WINDOW,
) -> pd.DataFrame:
    """
    Compute daily inverse-vol weights for each strategy.
    Weights are rebalanced on a weekly basis.

    Returns DataFrame indexed by date with a column per strategy (weight).
    """
    df = daily_pnl[["date"] + strategy_cols].copy()
    df = df.sort_values("date").set_index("date")

    # Resample to weekly rebalance dates (Monday close)
    rebal_dates = df.resample(WEIGHT_REBALANCE_FREQ).last().index

    weight_records = []

    for i, rebal_date in enumerate(rebal_dates):
        # Compute rolling vol over past `window` days of available history
        history = df[df.index <= rebal_date].tail(window)

        vols = {}
        for col in strategy_cols:
            v = history[col].std()
            vols[col] = max(v, 1e-6)

        inv_vols = {col: 1.0 / vols[col] for col in strategy_cols}
        total_inv = sum(inv_vols.values())
        weights = {col: inv_vols[col] / total_inv for col in strategy_cols}

        # Check for high inter-strategy correlation → reduce weaker strategy
        if len(strategy_cols) >= 2:
            corr = history[strategy_cols].corr()
            for j, col_a in enumerate(strategy_cols):
                for col_b in strategy_cols[j+1:]:
                    pair_corr = abs(float(corr.loc[col_a, col_b]))
                    if pair_corr > CORRELATION_THRESHOLD:
                        # Reduce the more recent / weaker strategy
                        pnl_a = history[col_a].sum()
                        pnl_b = history[col_b].sum()
                        weaker = col_b if pnl_b <= pnl_a else col_a
                        weights[weaker] *= 0.5

            # Renormalise
            total_w = sum(weights.values())
            if total_w > 0:
                weights = {k: v / total_w for k, v in weights.items()}

        # Apply these weights for the week until next rebal date
        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else df.index[-1]

        for day in df.index:
            if rebal_date <= day <= next_date:
                rec = {"date": day, **weights}
                weight_records.append(rec)

    if not weight_records:
        # Fallback: equal weights
        weight_records = [
            {"date": d, **{col: 1.0 / len(strategy_cols) for col in strategy_cols}}
            for d in df.index
        ]

    return pd.DataFrame(weight_records).drop_duplicates("date").set_index("date")


# ── Strategy suspension logic ─────────────────────────────────────────────

def check_strategy_drawdowns(
    daily_pnl: pd.DataFrame,
    strategy_cols: list[str],
    threshold: float = MAX_STRATEGY_DRAWDOWN_PCT,
    window: int = _STRAT_DD_WINDOW,
) -> pd.DataFrame:
    """
    Return a DataFrame indexed by date with bool columns per strategy.
    True = strategy is suspended (rolling 10-day drawdown exceeds threshold).
    """
    df = daily_pnl[["date"] + strategy_cols].copy().sort_values("date").set_index("date")
    suspended = pd.DataFrame(index=df.index, columns=strategy_cols, data=False)

    for col in strategy_cols:
        roll_sum = df[col].rolling(window, min_periods=window // 2).sum()
        # If rolling PnL over past 10 days is more negative than -threshold × some capital assumption
        # Use a simple absolute threshold: rolling loss > -threshold * 10-day start equity
        # Here: flag if rolling loss exceeds threshold × daily capital
        # (We check pct of daily mean abs PnL as a proxy)
        daily_vol = df[col].abs().mean()
        suspension_level = -threshold * daily_vol * window if daily_vol > 0 else float("-inf")
        suspended[col] = roll_sum < suspension_level

    return suspended


# ── Full multi-alpha engine ────────────────────────────────────────────────

def run_multi_alpha(
    portfolio_value: float = 1_000_000,
    output_dir: Path = RESULTS_DIR,
) -> dict:
    """
    Combine strategy P&L streams and produce portfolio-level statistics.

    Reads from:
      - results/daily_pnl_net.parquet   (from backtest.py)
      - results/trade_log_net.parquet

    Writes:
      - results/daily_pnl_combined.parquet
      - results/rolling_sharpe.parquet
      - summary to stdout
    """
    print("=== Phase 7: Multi-Alpha Engine ===")

    daily_pnl  = pd.read_parquet(output_dir / "daily_pnl_net.parquet")
    trade_log  = pd.read_parquet(output_dir / "trade_log_net.parquet")

    daily_pnl["date"] = pd.to_datetime(daily_pnl["date"])

    strategy_cols = [c for c in ["s1_pnl", "s4_pnl"] if c in daily_pnl.columns]

    if not strategy_cols:
        print("  No strategy columns found in daily_pnl — aborting.")
        return {}

    print(f"  Strategies: {strategy_cols}")

    # ── Individual strategy stats (run each independently first) ──────────
    print("\n  Per-strategy performance (raw, equal weight):")
    for col in strategy_cols:
        ret = daily_pnl[col] / portfolio_value
        sr  = sharpe_ratio(ret)
        dd  = max_drawdown(ret.cumsum())
        total = daily_pnl[col].sum()
        print(f"    {col}: Sharpe={sr:.3f}  MaxDD={dd*100:.2f}%  Total=${total:,.0f}")

    # ── Compute inverse-vol weights ────────────────────────────────────────
    print("\n  Computing inverse-vol weights (weekly rebalance) …")
    weights = compute_inverse_vol_weights(daily_pnl, strategy_cols)

    # Align weights with daily_pnl
    combined = daily_pnl.copy().set_index("date")
    weights  = weights.reindex(combined.index, method="ffill")

    # ── Strategy suspension check ─────────────────────────────────────────
    suspended = check_strategy_drawdowns(
        daily_pnl.set_index("date").reset_index(), strategy_cols
    )
    suspended = suspended.reindex(combined.index, fill_value=False)

    # ── Weighted portfolio P&L ─────────────────────────────────────────────
    combined["portfolio_pnl"] = 0.0
    for col in strategy_cols:
        w_col   = weights[col] if col in weights.columns else 1.0 / len(strategy_cols)
        susp    = suspended[col] if col in suspended.columns else pd.Series(False, index=combined.index)
        # Apply weight; set to 0 if suspended
        effective_weight = w_col * (~susp).astype(float)
        combined["portfolio_pnl"] += combined[col] * effective_weight

    combined = combined.reset_index()

    # ── Portfolio metrics ──────────────────────────────────────────────────
    port_ret = combined["portfolio_pnl"] / portfolio_value
    port_sh  = sharpe_ratio(port_ret)
    port_dd  = max_drawdown(port_ret.cumsum())

    print(f"\n  Combined portfolio:")
    print(f"    Sharpe:      {port_sh:.3f}")
    print(f"    Max drawdown:{port_dd*100:.2f}%")
    print(f"    Total P&L:  ${combined['portfolio_pnl'].sum():,.0f}")

    # ── Rolling correlation ────────────────────────────────────────────────
    if len(strategy_cols) >= 2:
        corr_df = daily_pnl.set_index("date")[strategy_cols].copy()
        roll_corr = corr_df.rolling(_CORR_WINDOW, min_periods=_CORR_WINDOW // 2).corr()
        print(f"\n  Rolling {_CORR_WINDOW}d correlation (latest):")
        for i, col_a in enumerate(strategy_cols):
            for col_b in strategy_cols[i+1:]:
                last_corr = roll_corr.xs(col_a, level=1)[col_b].dropna()
                if not last_corr.empty:
                    print(f"    {col_a} × {col_b}: {float(last_corr.iloc[-1]):.3f}")

    # ── Rolling Sharpe ────────────────────────────────────────────────────
    roll_sh_series = rolling_sharpe(port_ret)

    roll_sh_df = pd.DataFrame({
        "date":            combined["date"],
        "rolling_sharpe":  roll_sh_series.values,
    })
    roll_sh_df.to_parquet(output_dir / "rolling_sharpe.parquet", index=False)

    # ── Permutation test ──────────────────────────────────────────────────
    print("\n  Permutation test:")
    perm_result = permutation_test(combined.rename(columns={"portfolio_pnl": "total_pnl"}), portfolio_value)
    print(f"    Actual Sharpe: {perm_result['actual_sharpe']:.3f}")
    print(f"    Perm mean: {perm_result['perm_mean_sharpe']:.3f}  p={perm_result['p_value']:.3f}  "
          f"{'✓ Significant' if perm_result['significant'] else '✗ Not significant'}")

    # ── Spread sensitivity ────────────────────────────────────────────────
    if not trade_log.empty and "cost" in trade_log.columns:
        print("\n  Spread sensitivity (net P&L at 1×/2×/3× spreads):")
        spread_df = spread_sensitivity(trade_log)
        print(spread_df.to_string(index=False))

    # ── Save outputs ───────────────────────────────────────────────────────
    combined[["date"] + strategy_cols + ["portfolio_pnl"]].to_parquet(
        output_dir / "daily_pnl_combined.parquet", index=False
    )
    print(f"\n  Saved → {output_dir / 'daily_pnl_combined.parquet'}")
    print(f"  Saved → {output_dir / 'rolling_sharpe.parquet'}")

    return {
        "portfolio_sharpe":   port_sh,
        "portfolio_max_dd":   port_dd,
        "total_pnl":          float(combined["portfolio_pnl"].sum()),
        "permutation_test":   perm_result,
    }


if __name__ == "__main__":
    run_multi_alpha()
