"""
Top-level pipeline runner.

Executes all phases in dependency order.
Each phase reads from data/ and writes its artefacts back to data/ or results/.
Skip completed phases by passing --from-phase N.

Usage
-----
    python run_pipeline.py                  # full run
    python run_pipeline.py --from-phase 3   # resume from Phase 3 (Heston calibration)
    python run_pipeline.py --validate-only  # run Heston pricer validation only
    python run_pipeline.py --phases 0 1     # run only Phase 0 and Phase 1

Phases
------
  0  : Data audit + stock bars + external data
  1  : Contract filtering + IV computation + surface + arbitrage check
  2  : (implicitly part of 3) Heston pricer validation
  3  : Heston daily calibration + intraday Greeks
  4  : Signal generation (S1, S2, S4)
  5  : Backtest (gross + net)
  6  : Performance metrics
  7  : Multi-alpha combination
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, RESULTS_DIR, BACKTEST_START, BACKTEST_END, TICKERS, INSAMPLE_END, SLIPPAGE_SCENARIOS


def phase_0():
    from pipeline.audit_00.audit_data      import run_audit
    from pipeline.audit_00.build_stock_data import build_stock_bars
    from pipeline.audit_00.fetch_external   import run_fetch_external

    run_audit()
    build_stock_bars()
    run_fetch_external()


def phase_1():
    """Load raw option chains, filter, compute IV, build surfaces, check arbitrage."""
    import pandas as pd
    from pipeline.utils.data_loader         import load_all_options
    from pipeline.cleaning_01.compute_iv    import compute_iv_batch
    from pipeline.cleaning_01.build_surface import build_iv_surfaces
    from pipeline.cleaning_01.check_arbitrage import run_arbitrage_checks, filter_surface_by_arbitrage

    print("=== Phase 1: IV Surface Construction ===")

    print("  Loading raw option chains …")
    raw_options = load_all_options(BACKTEST_START, BACKTEST_END, TICKERS)

    if raw_options.empty:
        print("  [WARN] No raw options loaded — check Seagate drive connection.")
        return

    # Enrich with TTE if missing
    if "tte" not in raw_options.columns and "expiry_date" in raw_options.columns:
        raw_options["expiry_date"] = pd.to_datetime(raw_options["expiry_date"])
        raw_options["report_time"] = pd.to_datetime(raw_options["report_time"])
        raw_options["tte"] = (raw_options["expiry_date"] - raw_options["report_time"]).dt.days / 365.25

    stock_df   = pd.read_parquet(DATA_DIR / "stock_bars.parquet")
    external_df = pd.read_parquet(DATA_DIR / "external_data.parquet")

    print(f"  Computing IV for {len(raw_options):,} raw contracts …")
    iv_df = compute_iv_batch(raw_options, stock_df, external_df)

    if iv_df.empty:
        print("  [WARN] IV computation returned empty DataFrame.")
        return

    iv_df.to_parquet(DATA_DIR / "iv_data.parquet", index=False)
    print(f"  Saved {len(iv_df):,} rows with IV → data/iv_data.parquet")

    print("  Building IV surfaces …")
    surface_df = build_iv_surfaces(iv_df)

    print("  Checking no-arbitrage conditions …")
    flagged, _ = run_arbitrage_checks(surface_df)
    surface_clean = filter_surface_by_arbitrage(surface_df, flagged)

    surface_clean.to_parquet(DATA_DIR / "iv_surfaces_clean.parquet", index=False)
    print(f"  Saved clean IV surfaces → data/iv_surfaces_clean.parquet")


def phase_2_validate():
    """Validate the Heston pricer against known test cases."""
    from pipeline.heston_02.heston_pricer import validate_pricer
    print("=== Phase 2: Heston Pricer Validation ===")
    results = validate_pricer()
    all_pass = all(r["pass"] for r in results.values())
    if not all_pass:
        print("  [ERROR] Heston pricer validation failed — do not proceed to calibration.")
        sys.exit(1)
    print("  All test cases passed.")


def phase_3():
    from pipeline.heston_02.run_calibration  import run_calibration
    from pipeline.heston_02.intraday_greeks  import run_intraday_greeks
    run_calibration()
    run_intraday_greeks()


def phase_4():
    """Compute all three signals and save to data/."""
    import pandas as pd
    from pipeline.signals_03.signal_s1 import compute_s1_signals
    from pipeline.signals_03.signal_s2 import compute_s2_signals
    from pipeline.signals_03.signal_s4 import compute_s4_signals

    print("=== Phase 4: Signal Generation ===")

    iv_df       = pd.read_parquet(DATA_DIR / "iv_data.parquet")
    greeks_df   = pd.read_parquet(DATA_DIR / "intraday_greeks.parquet")
    stock_df    = pd.read_parquet(DATA_DIR / "stock_bars.parquet")
    external_df = pd.read_parquet(DATA_DIR / "external_data.parquet")

    print("  S1: Straddle IV Reversion …")
    s1 = compute_s1_signals(greeks_df, iv_df)
    s1.to_parquet(DATA_DIR / "signals_s1.parquet", index=False)
    print(f"  S1: {len(s1)} rows, {(s1['direction'] != 0).sum()} signals")

    print("  S2: Net GEX …")
    s2 = compute_s2_signals(iv_df, stock_df, external_df)
    s2.to_parquet(DATA_DIR / "signals_s2.parquet", index=False)
    regime_counts = s2["regime"].value_counts().to_dict() if "regime" in s2.columns else {}
    print(f"  S2: {len(s2)} rows  {regime_counts}")

    print("  S4: PCR Opening …")
    s4 = compute_s4_signals(iv_df, s2, stock_df)
    s4.to_parquet(DATA_DIR / "signals_s4.parquet", index=False)
    print(f"  S4: {len(s4)} rows, {(s4['direction'] != 0).sum()} signals")


def phase_5():
    from pipeline.backtest_04.backtest import run_backtest
    print("  Running gross backtest …")
    run_backtest(gross_pnl_only=True)
    print(f"  Running net backtests ({len(SLIPPAGE_SCENARIOS)} slippage scenarios) …")
    for slip in SLIPPAGE_SCENARIOS:
        print(f"    slippage = {int(slip*100)}% of spread …")
        run_backtest(gross_pnl_only=False, slippage_pct=slip)


def phase_6():
    from pipeline.backtest_04.metrics import run_metrics
    for slip in SLIPPAGE_SCENARIOS:
        slip_tag = f"slip{int(slip*100):02d}"
        print(f"\n=== Metrics: net {int(slip*100)}% slippage ===")
        run_metrics(
            trade_log_path=RESULTS_DIR / f"trade_log_net_{slip_tag}.parquet",
            daily_pnl_path=RESULTS_DIR / f"daily_pnl_net_{slip_tag}.parquet",
        )


def phase_7():
    from pipeline.multi_alpha_05.combine import run_multi_alpha
    run_multi_alpha()


# ── CLI ───────────────────────────────────────────────────────────────────

PHASE_MAP = {
    0: phase_0,
    1: phase_1,
    2: phase_2_validate,
    3: phase_3,
    4: phase_4,
    5: phase_5,
    6: phase_6,
    7: phase_7,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options backtesting pipeline runner")
    parser.add_argument("--from-phase", type=int, default=0, metavar="N",
                        help="Start from phase N (0-7)")
    parser.add_argument("--phases", nargs="+", type=int, metavar="N",
                        help="Run only these phases (e.g. --phases 0 1)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Run Heston pricer validation only (Phase 2)")
    args = parser.parse_args()

    if args.validate_only:
        phase_2_validate()
        sys.exit(0)

    if args.phases:
        phases_to_run = sorted(args.phases)
    else:
        phases_to_run = list(range(args.from_phase, max(PHASE_MAP.keys()) + 1))

    print(f"Running phases: {phases_to_run}")

    for p in phases_to_run:
        if p not in PHASE_MAP:
            print(f"  [WARN] Phase {p} not defined, skipping.")
            continue
        PHASE_MAP[p]()

    print("\n=== Pipeline complete ===")
