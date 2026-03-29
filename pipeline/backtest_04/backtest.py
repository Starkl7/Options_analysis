"""
Phase 6.1-6.3 — Event-Driven Backtesting Engine

Iterates over 15-min timestamps in chronological order.
At each bar:
  1. Update IV surface state
  2. Update Heston v0 (intraday proxy)
  3. Update Greeks
  4. Refresh OI/GEX at 9:30
  5. Compute S2 GEX signal
  6. Compute S1 IV z-score
  7. Opening bars: compute S4 PCR signal
  8. Execute new entries (bid/ask pricing)
  9. Rebalance S1 delta hedges (every 2 bars)
  10. Check exit conditions
  11. Mark positions to mid price

Outputs:
  results/trade_log.parquet
  results/daily_pnl.parquet

Run:
    python -m pipeline.04_backtest.backtest
"""
import math
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    TICKERS, DATA_DIR, RESULTS_DIR,
    BACKTEST_START, BACKTEST_END,
    TARGET_DAILY_VOL_PCT, MAX_NOTIONAL_PER_STOCK,
    MAX_DAILY_DRAWDOWN_PCT,
    DELTA_REBALANCE_BARS,
    UNDERLYING_MARKET_IMPACT_BPS,
    S1_ZSCORE_EXIT, S1_MAX_HOLD_BARS,
    S1_MIN_TTE_DAYS, S1_MAX_TTE_DAYS,
    S1_ZSCORE_ENTRY,
    ATR_LOOKBACK_DAYS, BARS_PER_DAY,
    CONTRACT_MULTIPLIER,
)
from pipeline.backtest_04.positions import StraddlePosition, DirectionalPosition, OptionLeg

warnings.filterwarnings("ignore")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading helpers ──────────────────────────────────────────────────

def _load_data():
    """Load all pre-computed artefacts needed by the backtest."""
    iv_df       = pd.read_parquet(DATA_DIR / "iv_data.parquet")
    greeks_df   = pd.read_parquet(DATA_DIR / "intraday_greeks.parquet")
    stock_df    = pd.read_parquet(DATA_DIR / "stock_bars.parquet")
    external_df = pd.read_parquet(DATA_DIR / "external_data.parquet")
    s1_df       = pd.read_parquet(DATA_DIR / "signals_s1.parquet")
    s2_df       = pd.read_parquet(DATA_DIR / "signals_s2.parquet")
    s4_df       = pd.read_parquet(DATA_DIR / "signals_s4.parquet")

    for df in [iv_df, greeks_df, stock_df]:
        for col in ["report_time", "timestamp"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

    external_df["date"] = pd.to_datetime(external_df["date"]).dt.date
    s1_df["timestamp"]  = pd.to_datetime(s1_df["timestamp"])
    s2_df["timestamp"]  = pd.to_datetime(s2_df["timestamp"])
    s4_df["date"]       = pd.to_datetime(s4_df["date"]).dt.date

    return iv_df, greeks_df, stock_df, external_df, s1_df, s2_df, s4_df


# ── Position sizing ───────────────────────────────────────────────────────

def _position_size(
    realized_vol: float,
    spot: float,
    portfolio_value: float,
    pcr_z: float = 1.0,
    cap: float = 2.0,
) -> float:
    """
    Size in shares based on vol targeting.
    base_size = TARGET_DAILY_VOL_PCT × portfolio_value / realized_vol
    Scaled by |signal_z| / z_threshold, capped at cap×base_size.
    """
    if realized_vol <= 0 or spot <= 0:
        return 0.0
    base_shares = TARGET_DAILY_VOL_PCT * portfolio_value / (realized_vol * spot)
    scale = min(abs(pcr_z) / 1.5, cap) if pcr_z != 0 else 1.0
    return base_shares * scale


def _atr(stock_df: pd.DataFrame, ticker: str, as_of_date, n_days: int = ATR_LOOKBACK_DAYS) -> float:
    """20-day ATR at given date, from 15-min bars."""
    sub = stock_df[
        (stock_df["ticker"] == ticker) &
        (stock_df["timestamp"].dt.date < as_of_date)
    ].tail(n_days * BARS_PER_DAY)

    if len(sub) < BARS_PER_DAY:
        return 0.0

    highs  = sub["high"].values if "high" in sub.columns else sub["close"].values
    lows   = sub["low"].values  if "low"  in sub.columns else sub["close"].values
    closes = sub["close"].values
    closes_prev = np.roll(closes, 1)
    closes_prev[0] = closes[0]

    tr = np.maximum(highs - lows,
         np.maximum(np.abs(highs - closes_prev),
                    np.abs(lows - closes_prev)))
    return float(tr.mean())


# ── Main backtest loop ────────────────────────────────────────────────────

def run_backtest(
    start: str = BACKTEST_START,
    end: str   = BACKTEST_END,
    tickers: list[str] = TICKERS,
    portfolio_value: float = 1_000_000,
    gross_pnl_only: bool = False,   # if True, no transaction costs (gross run)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the event-driven backtest.

    Returns
    -------
    trade_log  : pd.DataFrame — one row per closed trade
    daily_pnl  : pd.DataFrame — daily P&L decomposed by strategy and ticker
    """
    print(f"=== Backtest {'(GROSS)' if gross_pnl_only else '(NET)'} ===")

    iv_df, greeks_df, stock_df, external_df, s1_df, s2_df, s4_df = _load_data()

    # Build sorted timestamp index — restrict to regular trading hours only.
    # iv_data has after-hours bars at 16:xx and 17:xx; exclude those so that
    # the last bar of every day is the 15:45:xx bar and the EOD force-close fires.
    # Note: raw timestamps have sub-second jitter (e.g. 15:45:02), so we filter
    # by hour rather than exact time to avoid accidentally dropping the 15:45 bars.
    ts_col = "report_time" if "report_time" in iv_df.columns else "timestamp"
    _ts_series = iv_df[ts_col]
    all_timestamps = sorted(
        iv_df[
            _ts_series.dt.date.between(
                pd.Timestamp(start).date(), pd.Timestamp(end).date()
            ) &
            (_ts_series.dt.hour >= 9) &    # exclude any pre-market bars
            (_ts_series.dt.hour <= 15)     # exclude 16:xx and 17:xx after-hours bars
        ][ts_col].unique()
    )
    print(f"  Processing {len(all_timestamps):,} timestamps across {len(tickers)} tickers")

    # State
    open_straddles:    dict[str, list[StraddlePosition]]   = defaultdict(list)
    open_directional:  dict[str, list[DirectionalPosition]] = defaultdict(list)
    trade_log  = []
    daily_pnl_records = []

    # Daily portfolio drawdown tracker
    daily_start_pnl = {}
    total_pnl = 0.0          # running cumulative portfolio P&L (updated at each EOD)

    # Running cumulative strategy P&L — used to compute INCREMENTAL daily P&L
    _cum_s1 = 0.0
    _cum_s4 = 0.0

    for ts in all_timestamps:
        ts = pd.Timestamp(ts)
        date_key = ts.date()

        is_open_bar   = (ts.hour == 9 and ts.minute == 30)
        is_second_bar = (ts.hour == 9 and ts.minute == 45)
        is_eod_bar    = (ts.hour == 15 and ts.minute == 45)
        bar_num_today = (ts.hour - 9) * 4 + ts.minute // 15 - 2  # 0-indexed from 9:30
        is_rebal_bar  = (bar_num_today % DELTA_REBALANCE_BARS == 0)

        # Daily drawdown initialisation
        if is_open_bar:
            daily_start_pnl[date_key] = total_pnl

        # Check max daily drawdown
        daily_dd = total_pnl - daily_start_pnl.get(date_key, total_pnl)
        if daily_dd < -MAX_DAILY_DRAWDOWN_PCT * portfolio_value:
            # Close all positions for today
            for ticker in tickers:
                _force_close_all(open_straddles[ticker], open_directional[ticker],
                                 ts, "max_daily_drawdown", trade_log,
                                 stock_df, iv_df, portfolio_value, gross_pnl_only)
            continue

        for ticker in tickers:
            # ── Lookup bar-level data ─────────────────────────────────────
            spot = _get_spot(stock_df, ticker, ts)
            if spot is None:
                continue

            ext_row = external_df[external_df["date"] == date_key]
            r = float(ext_row["rf_rate"].iloc[0]) if not ext_row.empty else 0.04
            q_col = f"div_yield_{ticker}"
            q = float(ext_row[q_col].iloc[0]) if not ext_row.empty and q_col in ext_row.columns else 0.0

            rv_row = stock_df[(stock_df["ticker"] == ticker) & (stock_df["timestamp"] == ts)]
            realized_vol = float(rv_row["realized_vol_20d"].iloc[0]) if not rv_row.empty else 0.15

            # ── S2 GEX regime ─────────────────────────────────────────────
            s2_bar = s2_df[(s2_df["ticker"] == ticker) & (s2_df["timestamp"] == ts)]
            regime = str(s2_bar["regime"].iloc[0]) if not s2_bar.empty else "neutral"

            # ── S1 signal ─────────────────────────────────────────────────
            s1_bar = s1_df[(s1_df["ticker"] == ticker) & (s1_df["timestamp"] == ts)]
            if not s1_bar.empty:
                s1_row  = s1_bar.iloc[0]
                s1_dir  = int(s1_row["direction"])
                s1_z    = float(s1_row["z_score"])
                s1_strike  = float(s1_row["atm_strike"])
                s1_expiry  = s1_row.get("expiry_date")
                s1_tte     = float(s1_row["tte"])

                if s1_dir != 0 and _can_enter_new(open_straddles[ticker], ticker):
                    pos = _enter_straddle(
                        ticker=ticker, ts=ts,
                        atm_strike=s1_strike, expiry=s1_expiry, tte=s1_tte,
                        direction=s1_dir, spot=spot,
                        iv_df=iv_df, greeks_df=greeks_df,
                        portfolio_value=portfolio_value, realized_vol=realized_vol,
                        gross=gross_pnl_only,
                    )
                    if pos:
                        open_straddles[ticker].append(pos)

            # ── S4 signal (opening window only) ──────────────────────────
            if is_open_bar or is_second_bar:
                s4_day = s4_df[(s4_df["ticker"] == ticker) & (s4_df["date"] == date_key)]
                if not s4_day.empty:
                    s4_row = s4_day.iloc[0]
                    s4_dir = int(s4_row["direction"])

                    if s4_dir != 0 and is_second_bar:   # enter at close of bar 2
                        atr = _atr(stock_df, ticker, date_key)
                        qty = _position_size(realized_vol, spot, portfolio_value,
                                             pcr_z=float(s4_row.get("pcr_z", 1.5)))
                        if qty > 0 and spot * qty <= MAX_NOTIONAL_PER_STOCK:
                            stop = spot - s4_dir * atr if atr > 0 else spot * (1 - 0.02)
                            dp = _enter_directional(
                                ticker=ticker, ts=ts,
                                direction=s4_dir, qty=qty, price=spot,
                                stop=stop, strategy="S4",
                                market_impact_bps=0 if gross_pnl_only else UNDERLYING_MARKET_IMPACT_BPS,
                            )
                            open_directional[ticker].append(dp)

            # ── Rebalance S1 delta hedges & bar counting ───────────────────
            for pos in list(open_straddles[ticker]):
                if pos.is_closed:
                    continue
                # bars_since_entry increments every bar (not just rebalance bars)
                # so that S1_MAX_HOLD_BARS = 4 means exactly 4 bars = 1 hour
                pos.bars_since_entry += 1
                if is_rebal_bar:
                    net_delta = _get_straddle_delta(pos, greeks_df, ts)
                    if net_delta is not None:
                        half_spread = 0 if gross_pnl_only else 0.0005  # 5bps for stock
                        pos.rebalance_delta_hedge(net_delta, spot, half_spread)
                    pos.bars_since_rebal += 1

            # ── MTM open S1 positions ─────────────────────────────────────
            for pos in list(open_straddles[ticker]):
                if pos.is_closed:
                    continue
                call_mid, put_mid = _get_straddle_mids(pos, iv_df, ts)
                if call_mid and put_mid:
                    pos.mark_to_market(call_mid, put_mid, spot)

            # ── MTM open directional positions ────────────────────────────
            for pos in list(open_directional[ticker]):
                if pos.is_closed:
                    continue
                pos.mark_to_market(spot)

            # ── Check exits ───────────────────────────────────────────────
            # S1 exits
            for pos in list(open_straddles[ticker]):
                if pos.is_closed:
                    continue
                s1_bar_for_exit = s1_df[(s1_df["ticker"] == ticker) & (s1_df["timestamp"] == ts)]
                z_now = float(s1_bar_for_exit["z_score"].iloc[0]) if not s1_bar_for_exit.empty else None

                exit_reason = None
                if is_eod_bar:
                    exit_reason = "eod"
                elif pos.bars_since_entry >= S1_MAX_HOLD_BARS:
                    exit_reason = "max_hold"
                elif z_now is not None and abs(z_now) < S1_ZSCORE_EXIT:
                    exit_reason = "z_reversion"

                if exit_reason:
                    call_mid, put_mid = _get_straddle_mids(pos, iv_df, ts)
                    pos.close(
                        call_mid   = call_mid  or 0,
                        put_mid    = put_mid   or 0,
                        stock_price = spot,
                        timestamp  = ts,
                        reason     = exit_reason,
                        bid_ask_half_spread_opt = 0 if gross_pnl_only else 0.002,
                        market_impact_bps = 0 if gross_pnl_only else UNDERLYING_MARKET_IMPACT_BPS,
                    )
                    _log_trade(trade_log, pos, "S1")

            # S4 exits
            for pos in list(open_directional[ticker]):
                if pos.is_closed:
                    continue
                exit_reason = None
                if is_eod_bar:
                    exit_reason = "eod"
                elif pos.check_stop(spot):
                    exit_reason = "stop_loss"
                elif regime == "mean_reversion" and pos.strategy_id == "S4":
                    # GEX flipped against directional bet
                    exit_reason = "gex_flip"

                if exit_reason:
                    pos.close(
                        price     = spot,
                        timestamp = ts,
                        reason    = exit_reason,
                        market_impact_bps = 0 if gross_pnl_only else UNDERLYING_MARKET_IMPACT_BPS,
                    )
                    _log_trade(trade_log, pos, "S4")

        # Compute bar-level portfolio P&L
        bar_pnl = sum(
            (p.total_pnl for positions in open_straddles.values() for p in positions if not p.is_closed),
            0.0,
        ) + sum(
            (p.total_pnl for positions in open_directional.values() for p in positions if not p.is_closed),
            0.0,
        )

        if is_eod_bar:
            # Cumulative P&L of ALL positions ever (closed positions retain their final total_pnl)
            _new_cum_s1 = sum(p.total_pnl for positions in open_straddles.values()
                              for p in positions)
            _new_cum_s4 = sum(p.total_pnl for positions in open_directional.values()
                              for p in positions if p.strategy_id == "S4")

            # INCREMENTAL daily P&L = today's cumulative − yesterday's cumulative
            day_s1 = _new_cum_s1 - _cum_s1
            day_s4 = _new_cum_s4 - _cum_s4
            _cum_s1 = _new_cum_s1
            _cum_s4 = _new_cum_s4

            daily_pnl_records.append({
                "date":      date_key,
                "s1_pnl":    day_s1,
                "s4_pnl":    day_s4,
                "total_pnl": day_s1 + day_s4,
            })

            # Update running portfolio P&L for the intraday kill-switch
            total_pnl = _new_cum_s1 + _new_cum_s4

    # ── Save outputs ───────────────────────────────────────────────────────
    trade_df = pd.DataFrame(trade_log)
    daily_df = pd.DataFrame(daily_pnl_records)

    suffix = "_gross" if gross_pnl_only else "_net"
    trade_df.to_parquet(RESULTS_DIR / f"trade_log{suffix}.parquet", index=False)
    daily_df.to_parquet(RESULTS_DIR / f"daily_pnl{suffix}.parquet", index=False)

    print(f"  Closed trades:  {len(trade_df)}")
    if not daily_df.empty and "total_pnl" in daily_df.columns:
        total = daily_df["total_pnl"].sum()
        print(f"  Total P&L:     ${total:,.0f}")

    return trade_df, daily_df


# ── Helper functions ──────────────────────────────────────────────────────

def _get_spot(stock_df: pd.DataFrame, ticker: str, ts: pd.Timestamp) -> Optional[float]:
    sub = stock_df[stock_df["ticker"] == ticker].sort_values("timestamp")
    idx = sub["timestamp"].searchsorted(ts, side="right") - 1
    if idx < 0:
        return None
    return float(sub.iloc[idx]["close"])


def _can_enter_new(positions: list, ticker: str, max_concurrent: int = 2) -> bool:
    open_count = sum(1 for p in positions if not p.is_closed)
    return open_count < max_concurrent


def _enter_straddle(
    ticker, ts, atm_strike, expiry, tte, direction, spot,
    iv_df, greeks_df, portfolio_value, realized_vol, gross,
) -> Optional[StraddlePosition]:
    """Create a StraddlePosition with proper entry pricing."""
    opt_bar = iv_df[
        (iv_df["ticker"] == ticker) &
        (iv_df["report_time"] == ts) &
        (iv_df["strike"] == atm_strike)
    ]
    if opt_bar.empty:
        return None

    call_row = opt_bar[opt_bar["type"] == "c"]
    put_row  = opt_bar[opt_bar["type"] == "p"]
    if call_row.empty or put_row.empty:
        return None

    # Entry price: buy at ask, sell at bid
    if direction == -1:   # selling straddle
        call_entry = float(call_row["bid"].iloc[0])
        put_entry  = float(put_row["bid"].iloc[0])
        qty = -1   # short
    else:
        call_entry = float(call_row["ask"].iloc[0])
        put_entry  = float(put_row["ask"].iloc[0])
        qty = +1   # long

    # Entry cost = half bid-ask spread per leg × 100 shares/contract
    cost = 0.0 if gross else (
        (abs(float(call_row["ask"].iloc[0]) - float(call_row["bid"].iloc[0])) / 2 +
         abs(float(put_row["ask"].iloc[0])  - float(put_row["bid"].iloc[0])) / 2)
        * CONTRACT_MULTIPLIER
    )

    call_iv = float(call_row["iv"].iloc[0]) if "iv" in call_row.columns else 0.2
    put_iv  = float(put_row["iv"].iloc[0])  if "iv" in put_row.columns  else 0.2

    call_leg = OptionLeg(ticker, atm_strike, expiry, tte, "c",
                         qty, call_entry, ts, call_iv)
    put_leg  = OptionLeg(ticker, atm_strike, expiry, tte, "p",
                         qty, put_entry,  ts, put_iv)

    call_leg.current_price = (float(call_row["mid"].iloc[0])
                              if "mid" in call_row.columns else call_entry)
    put_leg.current_price  = (float(put_row["mid"].iloc[0])
                              if "mid" in put_row.columns else put_entry)

    pos = StraddlePosition(
        ticker=ticker, entry_time=ts,
        atm_strike=atm_strike, expiry_date=expiry,
        tte_at_entry=tte, direction=direction,
        call_leg=call_leg, put_leg=put_leg, cost=cost,
    )

    # Initial delta hedge
    # hedge_qty is in SHARES: delta (per share) × CONTRACT_MULTIPLIER × n_contracts
    g_bar = greeks_df[
        (greeks_df["ticker"] == ticker) &
        (greeks_df["timestamp"] == ts) &
        (greeks_df["strike"] == atm_strike)
    ]
    if not g_bar.empty:
        c_delta = float(g_bar[g_bar["type"] == "c"]["heston_delta"].iloc[0]) if "heston_delta" in g_bar.columns else 0.5
        p_delta = float(g_bar[g_bar["type"] == "p"]["heston_delta"].iloc[0]) if "heston_delta" in g_bar.columns else -0.5
        net_delta = qty * (c_delta + p_delta) * CONTRACT_MULTIPLIER
        pos.hedge_qty = -net_delta
        pos.hedge_entry_price = spot
        impact_cost = abs(pos.hedge_qty) * spot * UNDERLYING_MARKET_IMPACT_BPS / 10_000
        pos.cost += 0 if gross else impact_cost

    return pos


def _enter_directional(ticker, ts, direction, qty, price, stop, strategy, market_impact_bps):
    cost = qty * price * market_impact_bps / 10_000   # one-way entry
    pos = DirectionalPosition(
        ticker=ticker, entry_time=ts,
        direction=direction, quantity=qty,
        entry_price=price, stop_price=stop,
        strategy_id=strategy, cost=cost,
    )
    return pos


def _get_straddle_delta(
    pos: StraddlePosition,
    greeks_df: pd.DataFrame,
    ts: pd.Timestamp,
) -> Optional[float]:
    g_bar = greeks_df[
        (greeks_df["ticker"] == pos.ticker) &
        (greeks_df["timestamp"] == ts) &
        (greeks_df["strike"] == pos.atm_strike)
    ]
    if g_bar.empty:
        return None
    delta_col = "heston_delta" if "heston_delta" in g_bar.columns else "bs_delta"
    c_delta = float(g_bar[g_bar["type"] == "c"][delta_col].iloc[0]) if not g_bar[g_bar["type"] == "c"].empty else 0.5
    p_delta = float(g_bar[g_bar["type"] == "p"][delta_col].iloc[0]) if not g_bar[g_bar["type"] == "p"].empty else -0.5
    # Return net delta in SHARES (per share delta × 100 shares/contract × n_contracts)
    return (c_delta + p_delta) * pos.call_leg.quantity * CONTRACT_MULTIPLIER


def _get_straddle_mids(
    pos: StraddlePosition,
    iv_df: pd.DataFrame,
    ts: pd.Timestamp,
) -> tuple[Optional[float], Optional[float]]:
    opt_bar = iv_df[
        (iv_df["ticker"] == pos.ticker) &
        (iv_df["report_time"] == ts) &
        (iv_df["strike"] == pos.atm_strike)
    ]
    call_mid = float(opt_bar[opt_bar["type"] == "c"]["mid"].iloc[0]) if not opt_bar[opt_bar["type"] == "c"].empty and "mid" in opt_bar.columns else None
    put_mid  = float(opt_bar[opt_bar["type"] == "p"]["mid"].iloc[0]) if not opt_bar[opt_bar["type"] == "p"].empty and "mid" in opt_bar.columns else None
    return call_mid, put_mid


def _log_trade(trade_log: list, pos, strategy_id: str) -> None:
    if isinstance(pos, StraddlePosition):
        trade_log.append({
            "strategy":    strategy_id,
            "ticker":      pos.ticker,
            "entry_time":  pos.entry_time,
            "exit_time":   pos.exit_time,
            "exit_reason": pos.exit_reason,
            "direction":   pos.direction,
            "atm_strike":  pos.atm_strike,
            "pnl_gross":   pos.option_pnl + pos.hedge_pnl,
            "pnl_net":     pos.total_pnl,
            "cost":        pos.cost,
        })
    else:
        trade_log.append({
            "strategy":    strategy_id,
            "ticker":      pos.ticker,
            "entry_time":  pos.entry_time,
            "exit_time":   pos.exit_time,
            "exit_reason": pos.exit_reason,
            "direction":   pos.direction,
            "entry_price": pos.entry_price,
            "pnl_gross":   pos.pnl,
            "pnl_net":     pos.total_pnl,
            "cost":        pos.cost,
        })


def _force_close_all(straddles, directionals, ts, reason, trade_log,
                      stock_df, iv_df, portfolio_value, gross):
    for pos in straddles:
        if not pos.is_closed:
            spot = _get_spot(stock_df, pos.ticker, ts) or pos.atm_strike
            call_mid, put_mid = _get_straddle_mids(pos, iv_df, ts)
            pos.close(call_mid or 0, put_mid or 0, spot, ts, reason,
                      0 if gross else 0.002, 0 if gross else UNDERLYING_MARKET_IMPACT_BPS)
            _log_trade(trade_log, pos, "S1")

    for pos in directionals:
        if not pos.is_closed:
            spot = _get_spot(stock_df, pos.ticker, ts) or pos.entry_price
            pos.close(spot, ts, reason, 0 if gross else UNDERLYING_MARKET_IMPACT_BPS)
            _log_trade(trade_log, pos, pos.strategy_id)


if __name__ == "__main__":
    # Run gross first, then net
    run_backtest(gross_pnl_only=True)
    run_backtest(gross_pnl_only=False)
