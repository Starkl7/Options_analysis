"""
Phase 5.1 — Position Objects

Two position classes:
  - StraddlePosition (Strategy 1): call leg + put leg + delta hedge
  - DirectionalPosition (Strategies 2/4): long/short underlying

Both track entry/exit metadata and accumulate realised P&L.

Note: option prices are quoted per share; each contract covers
CONTRACT_MULTIPLIER (100) shares.  All option P&L and cost
calculations apply this multiplier so results are in dollar terms.
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

import numpy as np

CONTRACT_MULTIPLIER: int = 100   # shares per option contract


@dataclass
class OptionLeg:
    """A single option contract leg."""
    ticker:      str
    strike:      float
    expiry_date: object   # datetime or string
    tte:         float    # at entry
    option_type: str      # 'c' or 'p'
    quantity:    float    # signed (+1 long, -1 short)
    entry_price: float    # mid price at entry (mid-to-mid accounting)
    entry_time:  datetime
    iv_at_entry: float

    # Running state
    current_price: float = 0.0
    current_iv:    float = 0.0
    current_delta: float = 0.0


@dataclass
class StraddlePosition:
    """
    Strategy 1: short (or long) ATM straddle with delta hedge.

    direction: -1 = sold straddle (short vol); +1 = bought straddle (long vol)
    hedge_qty:  signed shares held for delta hedge (positive = long stock)
    """
    ticker:        str
    entry_time:    datetime
    atm_strike:    float
    expiry_date:   object
    tte_at_entry:  float
    direction:     int           # -1 short, +1 long

    call_leg:      OptionLeg = field(default=None)
    put_leg:       OptionLeg = field(default=None)

    hedge_qty:     float = 0.0   # shares held in delta hedge
    hedge_entry_price: float = 0.0

    # Accumulated P&L components
    option_pnl:   float = 0.0
    hedge_pnl:    float = 0.0
    cost:         float = 0.0    # total transaction costs paid

    # Spread-sensitive portion of total cost (entry half-spread + exit half-spread + slippage).
    # Scales linearly with bid-ask spread width; used by spread_sensitivity analysis.
    # Fixed fees ($1/contract) are excluded — they do not scale with spread.
    spread_related_cost: float = 0.0

    # Detailed transaction-cost decomposition for auditability
    cost_entry_spread: float = 0.0
    cost_exit_spread: float = 0.0
    cost_slippage: float = 0.0
    cost_fixed_fees: float = 0.0
    cost_hedge_impact: float = 0.0

    # ── Entry snapshot (mid / bid / ask per leg) ──────────────────────────
    ep_call:     float = 0.0   # call mid at entry
    ep_call_bid: float = 0.0
    ep_call_ask: float = 0.0
    ep_put:      float = 0.0   # put mid at entry
    ep_put_bid:  float = 0.0
    ep_put_ask:  float = 0.0

    # ── Exit snapshot (populated in close()) ─────────────────────────────
    xp_call:     float = 0.0   # call mid at exit
    xp_call_bid: float = 0.0
    xp_call_ask: float = 0.0
    xp_put:      float = 0.0   # put mid at exit
    xp_put_bid:  float = 0.0
    xp_put_ask:  float = 0.0

    # ── IV at entry ───────────────────────────────────────────────────────
    iv_computed: float = 0.0   # Jäckel-solved IV (avg call+put, from `iv` column)
    iv_reported: float = 0.0   # exchange market_iv (avg call+put, from `market_iv` column)

    # ── Signal z-scores ───────────────────────────────────────────────────
    entry_z: float = 0.0       # S1 z-score that triggered the entry
    exit_z:  float = 0.0       # S1 z-score at the bar when the trade was closed

    exit_time:    Optional[datetime] = None
    exit_reason:  str = ""
    is_closed:    bool = False

    # Last rebalance bar count
    bars_since_entry: int = 0
    bars_since_rebal: int = 0

    @property
    def total_pnl(self) -> float:
        return self.option_pnl + self.hedge_pnl - self.cost

    def mark_to_market(
        self,
        call_mid: float,
        put_mid: float,
        stock_price: float,
    ) -> None:
        """Update MTM values without closing position."""
        if self.call_leg:
            old = self.call_leg.current_price
            self.call_leg.current_price = call_mid
            # sign: direction -1 means we sold → profit if price drops
            # ×100 because option prices are per share, contract covers 100 shares
            self.option_pnl += self.call_leg.quantity * (call_mid - old) * CONTRACT_MULTIPLIER

        if self.put_leg:
            old = self.put_leg.current_price
            self.put_leg.current_price = put_mid
            self.option_pnl += self.put_leg.quantity * (put_mid - old) * CONTRACT_MULTIPLIER

        # Hedge P&L: hedge_qty × Δstock
        if self.hedge_qty != 0:
            self.hedge_pnl += self.hedge_qty * (stock_price - self.hedge_entry_price)
            self.hedge_entry_price = stock_price   # reset cost basis

    def rebalance_delta_hedge(
        self,
        new_delta: float,
        stock_price: float,
        half_spread: float,
    ) -> None:
        """
        Rebalance delta hedge to target delta.
        Charges half_spread × |quantity change| as transaction cost.
        """
        # Target hedge: offset straddle net delta
        target_hedge = -new_delta
        qty_change   = target_hedge - self.hedge_qty

        if abs(qty_change) < 0.01:   # avoid tiny rebalances
            return

        self.hedge_qty = target_hedge
        self.hedge_entry_price = stock_price
        hedge_c = abs(qty_change) * stock_price * half_spread
        self.cost += hedge_c
        self.cost_hedge_impact += hedge_c
        self.bars_since_rebal = 0

    def close(
        self,
        call_bid: float,
        call_ask: float,
        put_bid: float,
        put_ask: float,
        stock_price: float,
        timestamp: datetime,
        reason: str,
        fixed_cost_per_contract: float = 1.0,
        slippage_pct: float = 0.25,
        exit_half_spread_pct: float = 0.25,
        market_impact_bps: float = 1.0,
    ) -> None:
        """
        Close position at mid price (mid-to-mid accounting).

        option_pnl reflects the pure mid-to-mid price move; all execution
        friction is captured explicitly in cost:
          entry half-spread  : charged at entry (in _enter_straddle)
          exit half-spread   : exit_half_spread_pct × (ask - bid) per leg, charged here
          slippage_pct       : fraction of (ask - bid) charged as additional
                               slippage cost (e.g. 0.25 = 25%)
          fixed_cost_per_contract : flat fee in dollars (e.g. $1.00)

        spread_related_cost accumulates all spread-proportional costs
        (entry hs + exit hs + slippage) for use by spread_sensitivity().
        Fixed fees are excluded as they don't scale with spread width.

        All option P&L is ×CONTRACT_MULTIPLIER (100 shares / contract).
        """
        for leg, bid, ask, mid_attr, bid_attr, ask_attr in [
            (self.call_leg, call_bid, call_ask, "xp_call", "xp_call_bid", "xp_call_ask"),
            (self.put_leg,  put_bid,  put_ask,  "xp_put",  "xp_put_bid",  "xp_put_ask"),
        ]:
            if leg is None:
                continue

            spread = max(ask - bid, 0.0)
            mid    = (bid + ask) / 2 if (bid + ask) > 0 else leg.current_price

            # Store exit snapshot
            setattr(self, mid_attr, mid)
            setattr(self, bid_attr, bid)
            setattr(self, ask_attr, ask)

            # P&L: move from last MTM mid to exit mid (pure mid-to-mid)
            self.option_pnl += leg.quantity * (mid - leg.current_price) * CONTRACT_MULTIPLIER

            # Exit half-spread: explicit friction cost (no longer embedded in option_pnl)
            half_sp_c = (exit_half_spread_pct * spread) * abs(leg.quantity) * CONTRACT_MULTIPLIER
            self.cost += half_sp_c
            self.cost_exit_spread += half_sp_c

            # Additional slippage on top of half-spread
            slip_c = slippage_pct * spread * abs(leg.quantity) * CONTRACT_MULTIPLIER
            self.cost += slip_c
            self.cost_slippage += slip_c

            # Fixed per-contract fee
            fixed_c = fixed_cost_per_contract * abs(leg.quantity)
            self.cost += fixed_c
            self.cost_fixed_fees += fixed_c

            # Spread-sensitive costs (entry hs tracked at entry; exit hs + slippage here)
            self.spread_related_cost += half_sp_c + slip_c

        # Unwind delta hedge
        if self.hedge_qty != 0:
            self.hedge_pnl += self.hedge_qty * (stock_price - self.hedge_entry_price)
            impact_cost = abs(self.hedge_qty) * stock_price * (market_impact_bps / 10_000)
            self.cost += impact_cost
            self.cost_hedge_impact += impact_cost

        self.exit_time   = timestamp
        self.exit_reason = reason
        self.is_closed   = True


@dataclass
class DirectionalPosition:
    """
    Strategies 2/4: long or short underlying.

    direction: +1 = long, -1 = short
    """
    ticker:        str
    entry_time:    datetime
    direction:     int           # +1 long, -1 short
    quantity:      float         # shares (always positive; sign in direction)
    entry_price:   float
    stop_price:    float         # stop-loss trigger

    # Strategy that generated the signal
    strategy_id:   str = ""

    # Running P&L
    pnl:           float = 0.0
    cost:          float = 0.0

    exit_time:     Optional[datetime] = None
    exit_reason:   str = ""
    is_closed:     bool = False

    @property
    def total_pnl(self) -> float:
        return self.pnl - self.cost

    @property
    def signed_qty(self) -> float:
        return self.quantity * self.direction

    def mark_to_market(self, price: float) -> None:
        """Compute unrealised P&L from entry. Does not change state."""
        self.pnl = self.signed_qty * (price - self.entry_price)

    def check_stop(self, price: float) -> bool:
        """Return True if stop-loss has been hit."""
        if self.direction == +1:
            return price <= self.stop_price
        return price >= self.stop_price

    def close(
        self,
        price: float,
        timestamp: datetime,
        reason: str,
        market_impact_bps: float,
    ) -> None:
        """Close position, applying market impact cost."""
        self.pnl = self.signed_qty * (price - self.entry_price)
        self.cost = abs(self.signed_qty) * price * (market_impact_bps / 10_000) * 2  # round-trip
        self.exit_time   = timestamp
        self.exit_reason = reason
        self.is_closed   = True
