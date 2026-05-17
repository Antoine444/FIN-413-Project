"""
swap_simulator.py
=================
Standalone Uniswap V3 swap simulator (Module 3, Task 3.1).

Implements the step-by-step swap algorithm from the Uniswap V3 whitepaper
(§6.2–6.3) on top of a reconstructed tick-level liquidity map and an initial
sqrtPriceX96. The simulator is pure Python/NumPy with no RPC dependence; it is
meant for analysis-grade execution-cost simulation, not gas-exact replay.

Public API
----------
simulate_swap(ticks_df, sqrt_price_x96_start, current_tick, direction,
              notional_usd, fee=0.0005) -> dict
    USD-notional API. Used by slippage_analysis.py to drive the simulation
    grid (Task 3.2).

simulate_swap_raw(ticks_df, sqrt_price_x96_start, current_tick, zero_for_one,
                  amount_in_gross_raw, fee=0.0005) -> dict
    Token-amount API. Used for validation against observed swaps where the
    actual raw token input |amount0| / |amount1| is known (Task 3.1 → Fig 3.1).

Conventions
-----------
For the USDC/WETH 0.05% pool studied:
    token0 = USDC (6 decimals)   |   token1 = WETH (18 decimals)

Sign / price conventions confirmed empirically from output/slot0_snapshots.parquet:
    sqrtPriceX96 ≈ sqrt(token1_raw / token0_raw) · 2^96, ticks ≈ +193k.
    Human price (USDC/WETH) = 10^12 / (sqrtPriceX96 / 2^96)^2.

Direction mapping:
    buy_weth  : USDC in  → WETH out   |  zeroForOne = True   |  sqrtP DECREASES
    sell_weth : WETH in  → USDC out   |  zeroForOne = False  |  sqrtP INCREASES

Whitepaper step formulas (within a tick interval where L is constant):
    Δy = L · (√P_new − √P_old)                                   (token1 delta)
    Δx = L · (1/√P_new − 1/√P_old)                               (token0 delta)
    √P_new given input Δx (zeroForOne):
        √P_new = √P_old · L / (L + Δx · √P_old)
    √P_new given input Δy (¬zeroForOne):
        √P_new = √P_old + Δy / L

The 0.05% fee is taken off the input before the curve math; this matches the
pool's behaviour, where the fee accrues to LP feeGrowth* trackers rather than
moving the price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pool constants
# ---------------------------------------------------------------------------
TICK_SPACING   = 10
FEE_TIER       = 0.0005          # 0.05 %
Q96            = 2 ** 96
DECIMALS_USDC  = 6
DECIMALS_WETH  = 18
DECIMAL_ADJ    = 10 ** (DECIMALS_WETH - DECIMALS_USDC)   # 1e12
MIN_TICK       = -887_272
MAX_TICK       =  887_272


# ---------------------------------------------------------------------------
# Price / tick helpers
# ---------------------------------------------------------------------------
def sqrt_price_at_tick(tick: int) -> float:
    """sqrt(1.0001^tick) in natural units (= sqrtPriceX96 / 2^96)."""
    return 1.0001 ** (tick / 2.0)


def mid_price_usdc_per_weth(sqrt_price_x96: int) -> float:
    """Convert raw sqrtPriceX96 to human-readable USDC/WETH.

    raw_price = (sqrtPriceX96 / Q96)^2 = token1_raw / token0_raw
              = (raw WETH per raw USDC at equal value)
    human USDC/WETH = 10^(WETH_dec − USDC_dec) / raw_price = 1e12 / raw_price
    """
    sqrt_p = sqrt_price_x96 / Q96
    return DECIMAL_ADJ / (sqrt_p * sqrt_p)


# ---------------------------------------------------------------------------
# Tick-array preparation
# ---------------------------------------------------------------------------
def _build_tick_array(ticks_df: pd.DataFrame):
    """Return (ticks, liquidity_net, active_liquidity) as ascending numpy arrays.

    Required columns in ticks_df: ``tick``, ``liquidity_net``, ``active_liquidity``.
    String-encoded int128 values are parsed via float64 — precision is far better
    than required for analysis-grade output (liquidity values are ~1e22 vs. the
    1e15 float64 mantissa).
    """
    df = ticks_df.sort_values("tick").reset_index(drop=True)
    ticks = df["tick"].to_numpy(dtype=np.int64)
    # liquidity_net / active_liquidity may be stored as object (str) — astype handles both
    l_net = df["liquidity_net"].astype("float64").to_numpy()
    l_act = df["active_liquidity"].astype("float64").to_numpy()
    return ticks, l_net, l_act


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------
def simulate_swap_raw(
    ticks_df: pd.DataFrame,
    sqrt_price_x96_start: int,
    current_tick: int,
    zero_for_one: bool,
    amount_in_gross_raw: float,
    fee: float = FEE_TIER,
    l_active_override: float | None = None,
) -> dict:
    """Step-by-step Uniswap V3 swap on a static liquidity map.

    Parameters
    ----------
    ticks_df              : initialised ticks for one snapshot (one row per tick)
    sqrt_price_x96_start  : pool sqrtPriceX96 before the swap
    current_tick          : pool's active tick before the swap
    zero_for_one          : True for USDC→WETH (token0 in), False for WETH→USDC
    amount_in_gross_raw   : raw token input INCLUDING fee (matches |amount0| or
                            |amount1| from a Swap event)
    fee                   : pool fee rate (default 0.0005)
    l_active_override     : optional starting active liquidity. When provided,
                            overrides the value derived from the tick array.
                            Used by validation to pass the exact ``liquidity``
                            field emitted by the Swap event, neutralising
                            snapshot-drift on the active-liquidity bucket. The
                            tick array is still used for liquidity_net at
                            crossings.

    Returns
    -------
    dict with keys:
        amount_in_used_gross : raw input actually consumed (incl. fee portion)
        amount_in_used_net   : raw input that moved the curve (net of fee)
        amount_out_raw       : raw output to the trader
        fee_amount_raw       : fee charged on the consumed input
        sqrt_price_x96_end   : pool sqrtPriceX96 after the swap (int)
        ticks_crossed        : number of initialised-tick boundaries crossed
        exhausted            : True iff pool ran out of liquidity before input
    """
    ticks_arr, l_net_arr, l_act_arr = _build_tick_array(ticks_df)
    n = len(ticks_arr)
    if n == 0:
        raise ValueError("ticks_df contains no initialised ticks")

    # Locate the starting interval: largest tick index i with ticks_arr[i] <= current_tick
    pos = int(np.searchsorted(ticks_arr, current_tick, side="right") - 1)
    if pos < 0:
        raise ValueError(
            f"current_tick {current_tick} is below all initialised ticks "
            f"(min initialised = {ticks_arr[0]})"
        )

    L      = float(l_active_override) if l_active_override is not None else float(l_act_arr[pos])
    sqrt_p = sqrt_price_x96_start / Q96

    amount_in_net_remaining = amount_in_gross_raw * (1.0 - fee)
    amount_in_used_net = 0.0
    amount_out_total   = 0.0
    ticks_crossed      = 0
    exhausted          = False

    # Index of the next initialised tick we'd cross
    next_idx = pos if zero_for_one else pos + 1

    while amount_in_net_remaining > 1e-18:
        # Out-of-range check
        if (zero_for_one and next_idx < 0) or (not zero_for_one and next_idx >= n):
            exhausted = True
            break

        tick_boundary    = int(ticks_arr[next_idx])
        sqrt_p_boundary  = sqrt_price_at_tick(tick_boundary)

        if zero_for_one:
            # Price moves DOWN. Token0 (USDC) input → Token1 (WETH) output.
            # Δx required to move sqrt_p → sqrt_p_boundary:
            #     Δx = L · (1/sqrt_p_boundary − 1/sqrt_p)     (positive, since boundary < current)
            if L > 0:
                amount_to_boundary = L * (1.0 / sqrt_p_boundary - 1.0 / sqrt_p)
            else:
                amount_to_boundary = 0.0
        else:
            # Price moves UP. Token1 (WETH) input → Token0 (USDC) output.
            # Δy required: Δy = L · (sqrt_p_boundary − sqrt_p)   (positive)
            if L > 0:
                amount_to_boundary = L * (sqrt_p_boundary - sqrt_p)
            else:
                amount_to_boundary = 0.0

        if L <= 0 or amount_to_boundary <= amount_in_net_remaining:
            # ---- Cross the tick boundary (or skip an empty interval) ----
            if L > 0:
                if zero_for_one:
                    amount_out_step = L * (sqrt_p - sqrt_p_boundary)
                else:
                    amount_out_step = L * (1.0 / sqrt_p - 1.0 / sqrt_p_boundary)
                amount_in_used_net      += amount_to_boundary
                amount_out_total        += amount_out_step
                amount_in_net_remaining -= amount_to_boundary
            sqrt_p = sqrt_p_boundary
            # Update active liquidity per crossing rule
            if zero_for_one:
                L = L - float(l_net_arr[next_idx])
                next_idx -= 1
            else:
                L = L + float(l_net_arr[next_idx])
                next_idx += 1
            ticks_crossed += 1
        else:
            # ---- Partial step inside current interval — trade completes here ----
            if zero_for_one:
                sqrt_p_new      = sqrt_p * L / (L + amount_in_net_remaining * sqrt_p)
                amount_out_step = L * (sqrt_p - sqrt_p_new)
            else:
                sqrt_p_new      = sqrt_p + amount_in_net_remaining / L
                amount_out_step = L * (1.0 / sqrt_p - 1.0 / sqrt_p_new)
            amount_in_used_net      += amount_in_net_remaining
            amount_out_total        += amount_out_step
            amount_in_net_remaining  = 0.0
            sqrt_p                   = sqrt_p_new

    # Derive gross / fee quantities from the net consumption
    amount_in_used_gross = amount_in_used_net / (1.0 - fee)
    fee_amount_raw       = amount_in_used_gross - amount_in_used_net
    sqrt_price_x96_end   = int(sqrt_p * Q96)

    return {
        "amount_in_used_gross": amount_in_used_gross,
        "amount_in_used_net":   amount_in_used_net,
        "amount_out_raw":       amount_out_total,
        "fee_amount_raw":       fee_amount_raw,
        "sqrt_price_x96_end":   sqrt_price_x96_end,
        "ticks_crossed":        ticks_crossed,
        "exhausted":            exhausted,
    }


def simulate_swap(
    ticks_df: pd.DataFrame,
    sqrt_price_x96_start: int,
    current_tick: int,
    direction: str,
    notional_usd: float,
    fee: float = FEE_TIER,
) -> dict:
    """USD-notional swap simulator (Task 3.2 grid driver).

    Returns
    -------
    dict with keys:
        avg_executed_price : USDC per WETH (trader perspective, includes fee)
        mid_price          : USDC/WETH from sqrt_price_x96_start
        price_impact_bps   : 1e4·(p_exec−p_mid)/p_mid  (sign-flipped for sells:
                             positive = costly to taker)
        slippage_bps       : price_impact_bps net of the fee (= impact − 1e4·fee)
        ticks_crossed      : # initialised-tick boundaries crossed
        amount_in_token    : raw input token amount (USDC for buy, WETH for sell)
        amount_out_token   : raw output token amount
        sqrt_price_x96_end : pool sqrtPriceX96 after the swap
        exhausted          : True iff pool ran out of liquidity
    """
    mid_price = mid_price_usdc_per_weth(sqrt_price_x96_start)

    if direction == "buy_weth":
        zero_for_one      = True
        # USDC ≈ $1; gross input in raw token0 units
        amount_in_gross_raw = notional_usd * (10 ** DECIMALS_USDC)
    elif direction == "sell_weth":
        zero_for_one      = False
        # Convert USD notional → WETH using mid price
        weth_amount         = notional_usd / mid_price
        amount_in_gross_raw = weth_amount * (10 ** DECIMALS_WETH)
    else:
        raise ValueError(f"unknown direction: {direction!r}")

    res = simulate_swap_raw(
        ticks_df, sqrt_price_x96_start, current_tick,
        zero_for_one, amount_in_gross_raw, fee,
    )

    # Average executed price: always in USDC/WETH human units, computed from the
    # trader's gross input (= what they paid out of pocket) and the output.
    if direction == "buy_weth":
        usdc_paid_human = res["amount_in_used_gross"] / (10 ** DECIMALS_USDC)
        weth_recv_human = res["amount_out_raw"]       / (10 ** DECIMALS_WETH)
        avg_executed_price = (usdc_paid_human / weth_recv_human) if weth_recv_human > 0 else float("nan")
        # Buy: avg_executed > mid (taker pays more USDC per WETH)
        price_impact_bps   = 1e4 * (avg_executed_price - mid_price) / mid_price
    else:  # sell_weth
        weth_paid_human = res["amount_in_used_gross"] / (10 ** DECIMALS_WETH)
        usdc_recv_human = res["amount_out_raw"]       / (10 ** DECIMALS_USDC)
        avg_executed_price = (usdc_recv_human / weth_paid_human) if weth_paid_human > 0 else float("nan")
        # Sell: avg_executed < mid. Sign-flip so positive = costly to taker.
        price_impact_bps   = 1e4 * (mid_price - avg_executed_price) / mid_price

    slippage_bps = price_impact_bps - 1e4 * fee   # 5 bps for fee=0.0005

    return {
        "avg_executed_price": avg_executed_price,
        "mid_price":          mid_price,
        "price_impact_bps":   price_impact_bps,
        "slippage_bps":       slippage_bps,
        "ticks_crossed":      res["ticks_crossed"],
        "amount_in_token":    res["amount_in_used_gross"],
        "amount_out_token":   res["amount_out_raw"],
        "sqrt_price_x96_end": res["sqrt_price_x96_end"],
        "exhausted":          res["exhausted"],
    }


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    DATA_DIR = "output"
    liq = pd.read_parquet(os.path.join(DATA_DIR, "liquidity_snapshots.parquet"))
    s0  = pd.read_parquet(os.path.join(DATA_DIR, "slot0_snapshots.parquet"))

    snap = s0.iloc[0]
    ticks = liq[liq["snapshot_block"] == snap["snapshot_block"]]
    print(f"Self-test: snapshot block {snap['snapshot_block']}, "
          f"{len(ticks)} initialised ticks, mid price ≈ {snap['price_usdc_per_weth']:.2f} USDC/WETH")

    for size in (1_000, 10_000, 100_000, 1_000_000):
        for direction in ("buy_weth", "sell_weth"):
            out = simulate_swap(ticks, int(snap["sqrtPriceX96"]),
                                int(snap["current_tick"]), direction, size)
            print(f"  ${size:>10,}  {direction:<10}  "
                  f"avg={out['avg_executed_price']:.4f}  "
                  f"impact={out['price_impact_bps']:+.3f} bps  "
                  f"slip={out['slippage_bps']:+.3f} bps  "
                  f"ticks={out['ticks_crossed']}")
