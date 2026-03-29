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
    entry_price: float    # price transacted (bid or ask)
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
        target_hedge = -new_delta * (abs(self.call_leg.quantity) if self.call_leg else 1)
        qty_change   = target_hedge - self.hedge_qty

        if abs(qty_change) < 0.01:   # avoid tiny rebalances
            return

        self.hedge_qty = target_hedge
        self.hedge_entry_price = stock_price
        self.cost += abs(qty_change) * stock_price * half_spread
        self.bars_since_rebal = 0

    def close(
        self,
        call_mid: float,
        put_mid: float,
        stock_price: float,
        timestamp: datetime,
        reason: str,
        bid_ask_half_spread_opt: float,
        market_impact_bps: float,
    ) -> None:
        """
        Close position at current market prices.
        Buy back shorted options at ask, sell held options at bid.
        """
        # Options closing cost: transact at bid/ask
        # All option dollar P&L scaled by CONTRACT_MULTIPLIER (100 shares/contract)
        for leg, current_mid in [(self.call_leg, call_mid), (self.put_leg, put_mid)]:
            if leg is None:
                continue
            # If we sold (qty < 0), we buy back at ask = mid + half_spread
            # If we bought (qty > 0), we sell at bid = mid - half_spread
            slip = bid_ask_half_spread_opt if leg.quantity < 0 else -bid_ask_half_spread_opt
            exit_price = current_mid + slip
            self.option_pnl += leg.quantity * (exit_price - leg.current_price) * CONTRACT_MULTIPLIER
            self.cost += abs(leg.quantity) * bid_ask_half_spread_opt * CONTRACT_MULTIPLIER

        # Unwind delta hedge
        if self.hedge_qty != 0:
            self.hedge_pnl += self.hedge_qty * (stock_price - self.hedge_entry_price)
            impact_cost = abs(self.hedge_qty) * stock_price * (market_impact_bps / 10_000)
            self.cost += impact_cost

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
