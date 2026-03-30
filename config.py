"""
Global configuration for the options backtesting pipeline.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

# External drive where cron-collected parquets live
SEAGATE_ROOT = Path("/Volumes/SEAGATE/crondata")

# ── Universe ───────────────────────────────────────────────────────────────
# Tickers confirmed present on Seagate drive (audited 2026-03-28)
# Original placeholder had SPY/QQQ which are NOT in the collected data.
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "V", "DIS",
]

# ── Market hours (Eastern) ─────────────────────────────────────────────────
MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR  = 16
MARKET_CLOSE_MINUTE = 0

# ── Data parameters ────────────────────────────────────────────────────────
# cron_stock.py was fixed to use index=True on this date, making stock
# parquets usable (DatetimeIndex preserved). Pre-cutover files have no
# timestamps and must be sourced from yfinance instead.
STOCK_PARQUET_CUTOVER = "2026-03-28"

# Backtest window — confirmed from Seagate drive audit (2026-03-28)
# Phase 1 (daily EOD only):   2026-01-16 → 2026-02-12  ← excluded; not 15-min
# Phase 2 (intraday 15-min):  2026-02-13 → 2026-03-24
# Phase 3 (intraday + stock): 2026-03-25 → 2026-03-27
BACKTEST_START = "2026-02-13"   # first date with intraday 15-min snapshots
BACKTEST_END   = "2026-03-27"   # last date on drive at time of audit

# Walk-forward split: first 10 trading days in-sample, remaining 19 OOS.
# Day 1 = 2026-02-13 (Fri); Feb-16 = Presidents' Day (closed);
# Day 10 = 2026-02-27 (Fri).  OOS window: 2026-03-02 → 2026-03-27 (19 days).
INSAMPLE_END = "2026-02-27"

# Snap frequency
BAR_MINUTES = 15
BARS_PER_DAY = 26   # 9:30–16:00 = 6.5 hrs × 4 bars/hr = 26

# ── IV computation ─────────────────────────────────────────────────────────
IV_MIN = 0.01
IV_MAX = 2.0
IV_SOLVER_BOUNDS = (0.001, 5.0)
IV_SOLVER_TOL = 1e-6
IV_SOLVER_MAXITER = 100

# Contract filters
MONEYNESS_MIN = 0.85   # K/S lower bound
MONEYNESS_MAX = 1.15   # K/S upper bound
TTE_MIN_DAYS  = 5
TTE_MAX_DAYS  = 90

CONTRACT_MULTIPLIER = 100

# ── Heston calibration ─────────────────────────────────────────────────────
HESTON_BOUNDS = {
    "kappa": (0.01, 20.0),
    "theta": (0.001, 1.0),
    "xi":    (0.01, 2.0),
    "rho":   (-0.99, -0.01),
    "v0":    (0.001, 1.0),
}
FELLER_PENALTY_LAMBDA = 100.0   # weight on Feller soft constraint

# Calibration quality threshold
HESTON_IVRMSE_THRESHOLD = 0.02  # 200 bps

# Global optimiser settings
DE_POPSIZE   = 15    # population multiplier per dimension (15×5=75 individuals)
DE_MAXITER   = 20    # sufficient given warm-starts dominate after day 1
DE_TOL       = 1e-4

# Intraday v0 update: v0_t = ETA0 + ETA1 * VIX_t^2
# (coefficients are fitted per-ticker in-sample)
V0_ETA0 = 0.02
V0_ETA1 = 0.5

# Moneyness range for calibration universe
HESTON_MONO_MIN = 0.90
HESTON_MONO_MAX = 1.10
HESTON_TTE_MIN_DAYS = 14
HESTON_TTE_MAX_DAYS = 90
HESTON_MIN_CONTRACTS_PER_EXPIRY = 8

# ── Signal parameters ──────────────────────────────────────────────────────
# Strategy 1 — Straddle IV Reversion
S1_ROLLING_WINDOW = 10           # bars (150 min)
S1_ZSCORE_ENTRY   = 1.5
S1_ZSCORE_EXIT    = 0.5
S1_MAX_HOLD_BARS  = 4
S1_MIN_TTE_DAYS   = 14
S1_MAX_TTE_DAYS   = 30
S1_SHORT_ONLY     = False        # if True, only take S1 short-straddle entries (direction=-1)
S1_USE_VIX_OPEN_FILTER = True   # if True, gate S1 direction using VIX open level
S1_VIX_OPEN_LONG_MAX   = 0.0   # allow long straddle only when vix_open < this
S1_VIX_OPEN_SHORT_MIN  = 16.0   # allow short straddle only when vix_open > this

# Strategy 2 — Net GEX
S2_GEX_WINDOW_DAYS    = 20       # trading days for rolling norm
S2_REGIME_THRESHOLD   = 1.0      # std deviations

# Strategy 4 — PCR Opening Signal
S4_OPENING_BARS   = 2            # bars 1–2 (9:30–10:00)
S4_MONEYNESS_BAND = 0.05         # ±5% of spot
S4_NEAREST_EXPIRIES = 2
S4_PCR_WINDOW_DAYS  = 20
S4_ZSCORE_THRESHOLD = 1.5

# ── Backtest / execution ───────────────────────────────────────────────────
# Transaction costs
OPTION_SLIPPAGE_HALF_SPREAD = True    # buy at ask, sell at bid
UNDERLYING_MARKET_IMPACT_BPS = 0.0    # one-way, in basis points

# Option exit transaction costs (round-trip, applied at close)
# Fixed commission: $1 per option contract (100 shares) per round trip
FIXED_COST_PER_CONTRACT = 1
# Fraction of bid-ask spread charged as explicit half-spread at entry/exit.
# Example: 0.50 = true half-spread, 0.25 = quarter-spread, 0.25 = quarter-spread.
ENTRY_HALF_SPREAD_PCT = 0.15
EXIT_HALF_SPREAD_PCT  = 0.15
# Slippage fraction: charged as slippage_pct × (ask−bid) per contract per round trip
# Three scenarios: 0.10 (benchmark), 0.25 (poor scenario), 0.50 (market order equivalent)
SLIPPAGE_SCENARIOS = [0.10, 0.20, 0.40, 0.70]
SLIPPAGE_PCT = 0.10   # default; override in run_backtest() for sensitivity runs

# Position sizing: fraction of portfolio to deploy per straddle trade.
# n_contracts = floor(CAPITAL_PER_TRADE_PCT × portfolio_value / straddle_mid_per_contract)
# minimum 1 contract. Applied equally to long and short straddles.
CAPITAL_PER_TRADE_PCT = 0.01   # 1 % of portfolio per trade

# Position sizing
TARGET_DAILY_VOL_PCT  = 0.01          # 1% daily vol per strategy signal
MAX_NOTIONAL_PER_STOCK = 1_000_000    # USD

# Risk limits
MAX_DAILY_DRAWDOWN_PCT  = 0.015       # close all if portfolio hits -1.5% intraday
MAX_STRATEGY_DRAWDOWN_PCT = 0.03      # suspend strategy if -3% over 10 days

# Delta hedge rebalance frequency (Strategy 1)
DELTA_REBALANCE_BARS = 2

# ATR stop-loss lookback (Strategy 4)
ATR_LOOKBACK_DAYS = 20

# ── Multi-alpha ────────────────────────────────────────────────────────────
CORRELATION_THRESHOLD = 0.6           # reduce weight if pairwise corr > this
WEIGHT_REBALANCE_FREQ = "W-MON"       # pandas offset alias

# ── External data tickers ─────────────────────────────────────────────────
VIX_TICKER = "^VIX"
TBILL_TICKER = "^IRX"      # 13-week T-bill annualised yield (%)
