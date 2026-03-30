"""
Phase 0.3 — External Data

Fetches:
  - VIX daily OHLCV  (^VIX)
  - 13-week T-bill daily yield  (^IRX)  → continuous risk-free rate
  - Dividend yields per ticker  (yfinance Ticker.info)

Output: data/external_data.parquet
    columns: date, vix_open, vix_close, rf_rate, {TICKER}_div_yield

Run:
    python -m pipeline.00_audit.fetch_external
"""
import math
import warnings
from datetime import timedelta

import pandas as pd
import yfinance as yf

from config import TICKERS, DATA_DIR, BACKTEST_START, BACKTEST_END, VIX_TICKER, TBILL_TICKER

warnings.filterwarnings("ignore")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Pull a few extra days before the window to allow rolling lookbacks
_BUFFER_DAYS = 30


def _fetch_daily_price(ticker: str, start: str, end: str) -> pd.Series:
    """Return daily closing prices for a yfinance ticker symbol."""
    hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True)
    if hist.empty:
        return pd.Series(dtype=float, name=ticker)
    return hist["Close"].rename(ticker)


def fetch_vix(start: str, end: str) -> pd.Series:
    """Daily VIX open/close (annualised vol units, e.g. 18.5 = 18.5%)."""
    print("  Fetching VIX …")
    hist = yf.Ticker(VIX_TICKER).history(start=start, end=end, interval="1d", auto_adjust=True)
    if hist.empty:
        return pd.DataFrame(columns=["vix_open", "vix_close"])
    return hist[["Open", "Close"]].rename(columns={"Open": "vix_open", "Close": "vix_close"})


def fetch_risk_free_rate(start: str, end: str) -> pd.Series:
    """
    Fetch 13-week T-bill annualised yield (%) and convert to
    continuous risk-free rate:  r = ln(1 + yield_pct / 100)
    """
    print("  Fetching risk-free rate (^IRX) …")
    yield_pct = _fetch_daily_price(TBILL_TICKER, start, end)
    rf = yield_pct.apply(lambda y: math.log(1 + y / 100) if pd.notna(y) and y > 0 else float("nan"))
    rf.name = "rf_rate"
    return rf


def fetch_dividend_yields(tickers: list[str] = TICKERS) -> dict:
    """
    Pull annualised dividend yield from yfinance Ticker.info per stock.
    Converts to continuous yield: q = ln(1 + div_yield)
    Returns dict {ticker: continuous_yield}.
    """
    print("  Fetching dividend yields …")
    yields = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            raw  = info.get("dividendYield", 0.0) or 0.0   # None guard
            q    = math.log(1 + raw) if raw > 0 else 0.0
            yields[ticker] = round(q, 6)
            print(f"    {ticker:6s}: div_yield={raw:.4f}  →  q={q:.6f}")
        except Exception as e:
            print(f"    [WARN] {ticker}: could not fetch div yield ({e})")
            yields[ticker] = 0.0
    return yields


def run_fetch_external(
    start: str = BACKTEST_START,
    end: str   = BACKTEST_END,
    tickers: list[str] = TICKERS,
    output_path=DATA_DIR / "external_data.parquet",
) -> pd.DataFrame:
    print("=== Phase 0.3: Fetching external data ===")

    # Extend window slightly for rolling lookback warmup
    ext_start = (pd.Timestamp(start) - timedelta(days=_BUFFER_DAYS)).strftime("%Y-%m-%d")

    vix = fetch_vix(ext_start, end)
    rf  = fetch_risk_free_rate(ext_start, end)
    div = fetch_dividend_yields(tickers)

    # Align VIX and RF on a common date index
    combined = pd.concat([vix, rf], axis=1).sort_index()
    combined.index = pd.DatetimeIndex(combined.index).tz_localize(None)
    combined.index.name = "date"

    # Forward-fill weekends / holidays (use last known rate)
    combined = combined.ffill()

    # Add dividend yields as constant columns (one per ticker)
    for ticker in tickers:
        combined[f"div_yield_{ticker}"] = div.get(ticker, 0.0)

    combined = combined.reset_index()
    combined.to_parquet(output_path, index=False)

    print(f"\n  Saved {len(combined)} rows → {output_path}")
    if "vix_open" in combined.columns:
        print(f"  VIX open range:  {combined['vix_open'].min():.2f} – {combined['vix_open'].max():.2f}")
    print(f"  VIX range: {combined['vix_close'].min():.2f} – {combined['vix_close'].max():.2f}")
    print(f"  RF range:  {combined['rf_rate'].min():.4f} – {combined['rf_rate'].max():.4f}")
    print(f"  Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")

    return combined


if __name__ == "__main__":
    run_fetch_external()
