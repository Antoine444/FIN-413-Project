"""
lp_analytics.py
===============
Liquidity Provision Analytics (Module 4)

Produces:
  Fig 4.1  - Cumulative fee income time series for 5 synthetic LP positions
  Fig 4.2  - Impermanent loss time series for 5 synthetic LP positions
  Fig 4.3  - Net P&L (cumulative fees - IL) for 5 synthetic LP positions

Input files (Parquet, from Module 1):
  output/swap_events.parquet      all Swap events in the study window
  output/slot0_snapshots.parquet  daily price/tick snapshots

Outputs (Parquet, into output/):
  lp_positions.parquet    one row per position: range, L, x0, y0, ...
  lp_timeseries.parquet   one row per (position, snapshot): v_lp, v_hodl, IL, fees, net

The 5 synthetic positions are defined per the project spec (Module 4, Task 4.1):
  P1: +/-0.1%, P2: +/-0.5%, P3: +/-2%, P4: +/-10%, P5: full range.
All entered at the first daily snapshot block with $100,000 notional and exited
at the last. Fees are accrued literally per the PDF: share = L_p / swap.liquidity
(may exceed 1 for very narrow positions when L_p > realised pool depth).
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0.  CONFIGURATION
# ---------------------------------------------------------------------------

DATA_DIR   = "output"
OUTPUT_DIR = "figures"

SWAPS_PARQUET  = os.path.join(DATA_DIR, "swap_events.parquet")
SLOT0_PARQUET  = os.path.join(DATA_DIR, "slot0_snapshots.parquet")
POSITIONS_OUT  = os.path.join(DATA_DIR, "lp_positions.parquet")
TIMESERIES_OUT = os.path.join(DATA_DIR, "lp_timeseries.parquet")

# Uniswap V3 pool constants
POOL_FEE       = 0.0005           # 0.05 % fee tier
TICK_SPACING   = 10
Q96            = 2 ** 96
MIN_TICK       = -887_272
MAX_TICK       =  887_272

# Token decimals
DECIMALS_USDC  = 6
DECIMALS_WETH  = 18
DECIMAL_ADJ    = 10 ** (DECIMALS_WETH - DECIMALS_USDC)   # 1e12

# Module 4 spec
ENTRY_USD      = 100_000.0
RANGES_PCT = {
    "P1": 0.001,   # +/- 0.1 %   ultra-narrow, market-maker style
    "P2": 0.005,   # +/- 0.5 %   narrow, active LP
    "P3": 0.02,    # +/- 2 %     medium, typical retail LP
    "P4": 0.10,    # +/- 10 %    wide, passive LP
    "P5": None,    # full range, V2-equivalent
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1.  UNISWAP V3 MATH HELPERS
# Duplicated from liquidity_analysis.py:82-135 to keep this script
# self-contained per the per-module deliverable convention.
# ---------------------------------------------------------------------------

def tick_to_price(tick: int) -> float:
    """Convert V3 tick to USDC/WETH price (human-readable, decimal-adjusted).

    Native V3 price at tick = 1.0001^tick = WETH_raw / USDC_raw.
    Human USDC/WETH = USDC_raw/WETH_raw * 10^(WETH_dec - USDC_dec) = 10^12 / 1.0001^tick.

    (Note: liquidity_analysis.py:82 defines the same name with (1.0001^tick) * 1e12,
    which is mathematically the inverse and 16 orders of magnitude off. That function
    is defined-but-unused in Module 2. data_extraction.py:159 is the authoritative
    formula and is what writes the correct price_lower/price_upper to
    liquidity_snapshots.parquet.)
    """
    return DECIMAL_ADJ / (1.0001 ** tick)


def sqrt_price_at_tick(tick) -> float:
    """sqrt(1.0001^tick) in V3-native (non-decimal-adjusted) units.
    Equals sqrtPriceX96 / Q96."""
    return 1.0001 ** (tick / 2.0)


def token_amounts_from_liquidity(L, sqrt_pa, sqrt_pb, sqrt_pc):
    """Uniswap V3 virtual reserve formulas (whitepaper Sec. 6.2-6.3).

    All sqrt-price arguments must be in V3-native (non-decimal-adjusted)
    units, i.e. sqrt(1.0001^tick). Returns (amount_usdc, amount_weth) in
    human-readable units, so V_LP(p) = amount_weth * p + amount_usdc.

    Three cases:
      sqrt_pc <= sqrt_pa  (price above range, all USDC token0)
      sqrt_pc >= sqrt_pb  (price below range, all WETH token1)
      sqrt_pa <  sqrt_pc <  sqrt_pb  (in range, mixed)
    """
    if sqrt_pc <= sqrt_pa:
        raw_x = L * (1.0 / sqrt_pa - 1.0 / sqrt_pb)
        raw_y = 0.0
    elif sqrt_pc >= sqrt_pb:
        raw_x = 0.0
        raw_y = L * (sqrt_pb - sqrt_pa)
    else:
        raw_x = L * (1.0 / sqrt_pc - 1.0 / sqrt_pb)
        raw_y = L * (sqrt_pc - sqrt_pa)

    amount_usdc = raw_x / 10 ** DECIMALS_USDC
    amount_weth = raw_y / 10 ** DECIMALS_WETH
    return amount_usdc, amount_weth


# ---------------------------------------------------------------------------
# 2.  LOAD DATA
# ---------------------------------------------------------------------------

print("Loading parquet files ...")
slot0 = pd.read_parquet(SLOT0_PARQUET).sort_values("snapshot_block").reset_index(drop=True)
slot0["snapshot_timestamp"] = pd.to_datetime(slot0["snapshot_timestamp"], utc=True)

swaps = pd.read_parquet(SWAPS_PARQUET)
swaps["block_timestamp"] = pd.to_datetime(swaps["block_timestamp"], utc=True)

print(f"  {len(slot0)} daily snapshots "
      f"({slot0['snapshot_timestamp'].iloc[0].date()} -> "
      f"{slot0['snapshot_timestamp'].iloc[-1].date()})")
print(f"  {len(swaps):,} swap events")

entry_row     = slot0.iloc[0]
exit_row      = slot0.iloc[-1]
entry_block   = int(entry_row["snapshot_block"])
exit_block    = int(exit_row["snapshot_block"])
entry_tick    = int(entry_row["current_tick"])
entry_price   = float(entry_row["price_usdc_per_weth"])
exit_price    = float(exit_row["price_usdc_per_weth"])
# Exact V3-native sqrt-price at entry (sqrtPriceX96 / 2^96) - used for position
# sizing so V_LP(entry) = V_HODL(entry) = $100,000 to float precision.
entry_sqrt_pc = int(entry_row["sqrtPriceX96"]) / Q96
print(f"  Entry: block {entry_block:,}  tick {entry_tick:>7}  price ${entry_price:>9,.2f}")
print(f"  Exit:  block {exit_block:,}  tick {int(exit_row['current_tick']):>7}  price ${exit_price:>9,.2f}")
print(f"  Pool price moved {(exit_price / entry_price - 1) * 100:+.1f}% over the study window")


# ---------------------------------------------------------------------------
# 3.  TASK 4.1 - CONSTRUCT 5 LP POSITIONS
# ---------------------------------------------------------------------------

def usable_band_from_pct(entry_tick_int: int, k: float):
    """Convert a +/- k price band (USDC/WETH) around the entry tick to a
    pair of usable V3 ticks, rounded OUTWARD.

    Because USDC = token0, a HIGHER USDC/WETH price corresponds to a LOWER
    V3 tick. The price band [p*(1-k), p*(1+k)] therefore maps to a V3 tick
    band [entry - delta_low, entry + delta_high] where:
        delta_low  =  ln(1 + k) / ln(1.0001)   (ticks below entry: high price)
        delta_high = -ln(1 - k) / ln(1.0001)   (ticks above entry: low price)
    The two deltas differ slightly for non-trivial k (asymmetric in tick space).

    Round OUTWARD to multiples of TICK_SPACING so the realised tick band
    strictly contains the entry tick (V3 active condition is
    tick_lower <= currentTick < tick_upper).
    """
    delta_low  = math.log(1.0 + k) / math.log(1.0001)
    delta_high = -math.log(1.0 - k) / math.log(1.0001)
    raw_lo = entry_tick_int - delta_low
    raw_hi = entry_tick_int + delta_high
    tick_lower = math.floor(raw_lo / TICK_SPACING) * TICK_SPACING
    # floor + 1 (rather than ceil) guarantees tick_upper > raw_hi even when
    # raw_hi is exactly a multiple of TICK_SPACING.
    tick_upper = (math.floor(raw_hi / TICK_SPACING) + 1) * TICK_SPACING
    return tick_lower, tick_upper


def build_position(pos_id: str, k_pct, entry_tick_int, entry_price_float, entry_sqrt_pc):
    """Construct one synthetic LP position with $100K notional at entry.

    `entry_sqrt_pc` is the EXACT V3-native sqrt-price at the entry snapshot
    (= sqrtPriceX96 / 2^96 from slot0), not the tick-implied 1.0001^(tick/2).
    Using the exact value makes V_LP(entry) match V_HODL(entry) to float precision,
    so IL at the entry snapshot is exactly zero rather than ~$0.056.
    """
    if k_pct is None:
        # P5 - full range, aligned to TICK_SPACING
        tick_lower = math.ceil (MIN_TICK / TICK_SPACING) * TICK_SPACING   # -887270
        tick_upper = math.floor(MAX_TICK / TICK_SPACING) * TICK_SPACING   #  887270
    else:
        tick_lower, tick_upper = usable_band_from_pct(entry_tick_int, k_pct)

    sqrt_pa = sqrt_price_at_tick(tick_lower)
    sqrt_pb = sqrt_price_at_tick(tick_upper)
    sqrt_pc = entry_sqrt_pc                          # exact slot0 sqrt-price

    # Probe-and-scale: token_amounts is linear in L, so compute reserves at
    # L = 1, then scale to the $100K notional constraint.
    usdc_per_L, weth_per_L = token_amounts_from_liquidity(1.0, sqrt_pa, sqrt_pb, sqrt_pc)
    v_per_L = weth_per_L * entry_price_float + usdc_per_L
    L = ENTRY_USD / v_per_L
    x0_weth = weth_per_L * L
    y0_usdc = usdc_per_L * L
    entry_value = x0_weth * entry_price_float + y0_usdc

    # tick_lower (lower V3 tick) corresponds to HIGHER USDC/WETH price; flip
    # the labels so price_lower / price_upper read in USDC/WETH order.
    return {
        "position_id":               pos_id,
        "range_pct":                 k_pct,
        "tick_lower":                tick_lower,
        "tick_upper":                tick_upper,
        "price_lower_usdc_per_weth": tick_to_price(tick_upper),
        "price_upper_usdc_per_weth": tick_to_price(tick_lower),
        "L":                         L,
        "sqrt_pa":                   sqrt_pa,
        "sqrt_pb":                   sqrt_pb,
        "x0_weth":                   x0_weth,
        "y0_usdc":                   y0_usdc,
        "entry_value_usd":           entry_value,
        "entry_block":               entry_block,
        "entry_timestamp":           entry_row["snapshot_timestamp"],
        "exit_block":                exit_block,
        "exit_timestamp":            exit_row["snapshot_timestamp"],
    }


print("\nTask 4.1 - constructing 5 LP positions ...")
positions = [build_position(pid, k, entry_tick, entry_price, entry_sqrt_pc)
             for pid, k in RANGES_PCT.items()]

print(f"  {'id':<3} {'nominal':>7}  {'realised':>16}  "
      f"{'tick_lo':>8}  {'tick_hi':>8}  "
      f"{'p_low':>10}  {'p_high':>10}  {'L':>11}  "
      f"{'x0_WETH':>10}  {'y0_USDC':>10}  {'V_entry':>10}")
for p in positions:
    if p["range_pct"] is None:
        k_str       = "full"
        realised_str = "       full      "
    else:
        k_str = f"+/-{p['range_pct']*100:.1f}%"
        # Realised half-widths after outward TICK_SPACING rounding (asymmetric)
        rd_lo = (entry_price - p["price_lower_usdc_per_weth"]) / entry_price * 100
        rd_hi = (p["price_upper_usdc_per_weth"] - entry_price) / entry_price * 100
        realised_str = f"-{rd_lo:.3f}% / +{rd_hi:.3f}%"
    print(f"  {p['position_id']:<3} {k_str:>7}  {realised_str:>16}  "
          f"{p['tick_lower']:>8}  {p['tick_upper']:>8}  "
          f"{p['price_lower_usdc_per_weth']:>10,.2f}  "
          f"{p['price_upper_usdc_per_weth']:>10,.2f}  "
          f"{p['L']:>11.3e}  {p['x0_weth']:>10,.4f}  {p['y0_usdc']:>10,.2f}  "
          f"{p['entry_value_usd']:>10,.2f}")

# Write positions parquet. L can exceed int64 in extreme cases, so store as
# string per the project convention for uint128 values.
positions_df = pd.DataFrame([{
    "position_id":               p["position_id"],
    "range_pct":                 p["range_pct"],
    "tick_lower":                p["tick_lower"],
    "tick_upper":                p["tick_upper"],
    "price_lower_usdc_per_weth": p["price_lower_usdc_per_weth"],
    "price_upper_usdc_per_weth": p["price_upper_usdc_per_weth"],
    "L":                         str(int(p["L"])),
    "x0_weth":                   p["x0_weth"],
    "y0_usdc":                   p["y0_usdc"],
    "entry_value_usd":           p["entry_value_usd"],
    "entry_block":               p["entry_block"],
    "entry_timestamp":           p["entry_timestamp"],
    "exit_block":                p["exit_block"],
    "exit_timestamp":            p["exit_timestamp"],
} for p in positions])
positions_df.to_parquet(POSITIONS_OUT, index=False)
print(f"\n  Wrote {POSITIONS_OUT}  ({len(positions_df)} rows)")


# ---------------------------------------------------------------------------
# 4.  TASK 4.2 - FEE INCOME (VECTORISED)
# ---------------------------------------------------------------------------

print("\nTask 4.2 - pre-computing per-swap fee components ...")
swaps["liquidity_f"] = swaps["liquidity"].astype(float)

# Per-swap LP-side gross fee, BEFORE applying the share L_p / L_swap.
# Token0 (USDC) fees on swaps where USDC is deposited (amount0_decimal > 0);
# Token1 (WETH) fees on swaps where WETH is deposited (amount1_decimal > 0).
# The Swap event's amount0/amount1 INCLUDE the fee that stays in the pool, so
# multiplying by 0.0005 gives the LP-side fee directly.
swaps["usdc_fee_unit"] = np.where(swaps["amount0_decimal"] > 0,
                                  swaps["amount0_decimal"] * POOL_FEE, 0.0)
swaps["weth_fee_unit"] = np.where(swaps["amount1_decimal"] > 0,
                                  swaps["amount1_decimal"] * POOL_FEE, 0.0)
# Convert WETH fee to USD at the swap's own price (= price at time of collection)
swaps["usd_fee_unit"]  = (swaps["usdc_fee_unit"]
                          + swaps["weth_fee_unit"] * swaps["price_usdc_per_weth"])

total_pool_fees = float(swaps["usd_fee_unit"].sum())
print(f"  Total LP-side pool fees over study window: ${total_pool_fees:,.2f}")


def compute_fee_series(p, swaps_df, snap_blocks_arr):
    """Vectorised per-position fee accrual, aggregated by NEXT daily snapshot.

    Window: entry_block < block_number <= exit_block.
    Active: tick_lower <= swap.tick < tick_upper.
    Share:  L_p / swap.liquidity  (per PDF spec; may exceed 1 for narrow positions).

    Returns (cum_usd, cum_usdc, cum_weth, max_share, n_active_swaps).
      cum_usd  : cumulative fee in USD, with WETH fees converted at the
                 swap's own price (= ETH price prevailing at accrual time).
      cum_usdc : cumulative fee in USDC tokens (never converted).
      cum_weth : cumulative fee in WETH tokens (never converted).

    The three series let a reader re-aggregate under any valuation rule:
    e.g. value the WETH leg at the exit price instead of the swap-time price.
    """
    mask = ((swaps_df["block_number"] > p["entry_block"])
            & (swaps_df["block_number"] <= p["exit_block"])
            & (swaps_df["tick"] >= p["tick_lower"])
            & (swaps_df["tick"] <  p["tick_upper"]))
    cols = ["block_number", "usd_fee_unit", "usdc_fee_unit", "weth_fee_unit",
            "liquidity_f"]
    sub = swaps_df.loc[mask, cols].copy()

    zero_idx = pd.Index(snap_blocks_arr, name="snapshot_block")
    if len(sub) == 0:
        zeros = pd.Series(0.0, index=zero_idx)
        return zeros, zeros.copy(), zeros.copy(), float("nan"), 0

    sub["share"]      = p["L"] / sub["liquidity_f"]
    sub["usd_fee_p"]  = sub["usd_fee_unit"]  * sub["share"]
    sub["usdc_fee_p"] = sub["usdc_fee_unit"] * sub["share"]
    sub["weth_fee_p"] = sub["weth_fee_unit"] * sub["share"]
    max_share = float(sub["share"].max())

    # Assign each swap to its NEXT snapshot block via searchsorted. side="left"
    # returns the smallest idx with snap_blocks[idx] >= block_number, so a swap
    # within (snap_blocks[i-1], snap_blocks[i]] is bucketed to snap_blocks[i].
    idx = np.searchsorted(snap_blocks_arr, sub["block_number"].values, side="left")
    sub["snapshot_block"] = snap_blocks_arr[idx]

    grouped = sub.groupby("snapshot_block")[["usd_fee_p", "usdc_fee_p", "weth_fee_p"]].sum()
    grouped = grouped.reindex(snap_blocks_arr, fill_value=0.0)
    grouped.index.name = "snapshot_block"
    cum_usd  = grouped["usd_fee_p"].cumsum()
    cum_usdc = grouped["usdc_fee_p"].cumsum()
    cum_weth = grouped["weth_fee_p"].cumsum()
    return cum_usd, cum_usdc, cum_weth, max_share, int(len(sub))


print("\nComputing per-position fee accruals ...")
snap_blocks       = slot0["snapshot_block"].values
fee_series_usd    = {}
fee_series_usdc   = {}
fee_series_weth   = {}
max_shares        = {}
n_active          = {}
for p in positions:
    cum_usd, cum_usdc, cum_weth, max_share, n_swaps = compute_fee_series(
        p, swaps, snap_blocks)
    fee_series_usd [p["position_id"]] = cum_usd
    fee_series_usdc[p["position_id"]] = cum_usdc
    fee_series_weth[p["position_id"]] = cum_weth
    max_shares     [p["position_id"]] = max_share
    n_active       [p["position_id"]] = n_swaps
    print(f"  {p['position_id']}: {n_swaps:>7,} active swaps   "
          f"terminal fee ${cum_usd.iloc[-1]:>11,.2f}   "
          f"(USDC ${cum_usdc.iloc[-1]:>9,.2f}  +  WETH {cum_weth.iloc[-1]:>7.4f})   "
          f"max share {max_share:>7.3f}")


# ---------------------------------------------------------------------------
# 5.  TASK 4.3 - IMPERMANENT LOSS
# ---------------------------------------------------------------------------

print("\nTask 4.3 - computing V_LP, V_HODL, IL at all daily snapshots ...")
records = []
for snap in slot0.itertuples():
    p_t     = float(snap.price_usdc_per_weth)
    sqrt_pc = int(snap.sqrtPriceX96) / Q96      # V3-native sqrt-price
    for pos in positions:
        usdc_t, weth_t = token_amounts_from_liquidity(
            pos["L"], pos["sqrt_pa"], pos["sqrt_pb"], sqrt_pc)
        v_lp   = weth_t * p_t + usdc_t
        v_hodl = pos["x0_weth"] * p_t + pos["y0_usdc"]
        il     = v_hodl - v_lp
        records.append({
            "position_id":          pos["position_id"],
            "snapshot_block":       int(snap.snapshot_block),
            "snapshot_timestamp":   snap.snapshot_timestamp,
            "price_usdc_per_weth":  p_t,
            "v_lp_usd":             v_lp,
            "v_hodl_usd":           v_hodl,
            "impermanent_loss_usd": il,
        })

ts = pd.DataFrame(records)

# Attach cumulative fees via long-form merge (cleaner than row-wise apply).
# Three columns: USDC fees, WETH fees (each in native token, no conversion),
# and USD fees (WETH leg converted at swap-time price). The USD column is the
# one used for net P&L; the native columns let the reader re-aggregate under
# any other valuation rule.
fee_long = pd.concat(
    [pd.DataFrame({"position_id":         pid,
                   "snapshot_block":      fee_series_usd [pid].index.values,
                   "cumulative_fee_usd":  fee_series_usd [pid].values,
                   "cumulative_fee_usdc": fee_series_usdc[pid].values,
                   "cumulative_fee_weth": fee_series_weth[pid].values})
     for pid in fee_series_usd.keys()],
    ignore_index=True,
)
ts = ts.merge(fee_long, on=["position_id", "snapshot_block"], how="left")
ts["net_pnl_usd"] = ts["cumulative_fee_usd"] - ts["impermanent_loss_usd"]
ts = ts.sort_values(["position_id", "snapshot_block"]).reset_index(drop=True)

ts.to_parquet(TIMESERIES_OUT, index=False)
print(f"  Wrote {TIMESERIES_OUT}  ({len(ts)} rows = {len(positions)} x {len(slot0)})")

print("\nTerminal P&L summary:")
print(f"  {'id':<3}  {'fee_usd':>11}  {'fee_usdc':>10}  {'fee_weth':>10}  "
      f"{'IL':>11}  {'net':>11}  {'v_lp':>11}  {'v_hodl':>11}")
for pid in RANGES_PCT.keys():
    last = ts[ts["position_id"] == pid].iloc[-1]
    print(f"  {pid:<3}  "
          f"${last['cumulative_fee_usd']:>10,.2f}  "
          f"${last['cumulative_fee_usdc']:>9,.2f}  "
          f"{last['cumulative_fee_weth']:>10.4f}  "
          f"${last['impermanent_loss_usd']:>10,.2f}  "
          f"${last['net_pnl_usd']:>10,.2f}  "
          f"${last['v_lp_usd']:>10,.2f}  "
          f"${last['v_hodl_usd']:>10,.2f}")

# Re-aggregation cross-check: value WETH fees at exit price instead of swap-time
# price. The PDF phrase "ETH price prevailing at the time of collection" can be
# read either way; this shows the gap so the report can quote both.
print("\nFee re-aggregation under alternative WETH valuation rules:")
print(f"  {'id':<3}  {'@swap-time':>12}  {'@exit price':>12}  {'gap':>10}")
for pid in RANGES_PCT.keys():
    last = ts[ts["position_id"] == pid].iloc[-1]
    swap_time = last["cumulative_fee_usd"]
    at_exit   = last["cumulative_fee_usdc"] + last["cumulative_fee_weth"] * exit_price
    print(f"  {pid:<3}  ${swap_time:>11,.2f}  ${at_exit:>11,.2f}  "
          f"${at_exit - swap_time:>+9,.2f}")


# ---------------------------------------------------------------------------
# 6.  FIGURES
# ---------------------------------------------------------------------------

# Position labels: nominal half-width plus the realised half-widths after
# outward TICK_SPACING=10 rounding (asymmetric in % space). Without this the
# legend overstates the precision of P1, where the realised band can be ~50%
# wider than the requested +/-0.1%.
#
# Use Unicode +/-/- characters directly rather than matplotlib mathtext ($...$).
# The terminal-value suffix added by each plot function contains a literal `$`
# (e.g. "$14,493"), which unbalances the mathtext parser and would render the
# whole legend as raw source text. Sticking to Unicode sidesteps that entirely.
def _build_position_labels(pos_list, p_entry):
    labels = {}
    for p in pos_list:
        pid = p["position_id"]
        if p["range_pct"] is None:
            labels[pid] = "P5: full range"
            continue
        nom = f"{p['range_pct'] * 100:g}%"
        rd_lo = (p_entry - p["price_lower_usdc_per_weth"]) / p_entry * 100
        rd_hi = (p["price_upper_usdc_per_weth"] - p_entry) / p_entry * 100
        labels[pid] = f"{pid}: ±{nom} (realised −{rd_lo:.2f}% / +{rd_hi:.2f}%)"
    return labels

POSITION_LABELS = _build_position_labels(positions, entry_price)
COLORS = {"P1": "#d62728", "P2": "#ff7f0e", "P3": "#2ca02c",
          "P4": "#1f77b4", "P5": "#9467bd"}


def _format_date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def _fmt_terminal(value):
    """Format a USD terminal value with an escaped $ so matplotlib doesn't
    interpret the dollar sign as a mathtext delimiter. Returns "-$1,234" for
    negatives so the minus sign sits outside the escape."""
    if value < 0:
        return f"-\\${abs(value):,.0f}"
    return f"\\${value:,.0f}"


def plot_fee_income(ts_df, out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for pid in RANGES_PCT.keys():
        d = ts_df[ts_df["position_id"] == pid]
        term = d["cumulative_fee_usd"].iloc[-1]
        ax.plot(d["snapshot_timestamp"], d["cumulative_fee_usd"],
                color=COLORS[pid], lw=1.8,
                label=f"{POSITION_LABELS[pid]}  (terminal: {_fmt_terminal(term)})")
    ax.set_title("Cumulative Fee Income for \\$100,000 Synthetic LP Positions"
                 " (Uniswap V3 USDC/WETH 0.05%, swap-time WETH valuation)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative fee income (USD, log scale)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    _format_date_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  {out_path}")


def plot_il(ts_df, slot0_df, out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for pid in RANGES_PCT.keys():
        d = ts_df[ts_df["position_id"] == pid]
        term = d["impermanent_loss_usd"].iloc[-1]
        ax.plot(d["snapshot_timestamp"], d["impermanent_loss_usd"],
                color=COLORS[pid], lw=1.8,
                label=f"{POSITION_LABELS[pid]}  (terminal: {_fmt_terminal(term)})")
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.6)
    ax.set_title("Impermanent Loss (V_HODL − V_LP) for \\$100,000 Synthetic LP Positions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Impermanent loss (USD; positive = LP underperforms HODL)")
    ax.grid(True, ls=":", alpha=0.4)
    ax2 = ax.twinx()
    ax2.plot(slot0_df["snapshot_timestamp"], slot0_df["price_usdc_per_weth"],
             color="gray", lw=1.0, alpha=0.5)
    ax2.set_ylabel("ETH price (USDC/WETH)", color="gray")
    ax2.tick_params(axis="y", colors="gray")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    _format_date_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  {out_path}")


def plot_net(ts_df, out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for pid in RANGES_PCT.keys():
        d = ts_df[ts_df["position_id"] == pid]
        term = d["net_pnl_usd"].iloc[-1]
        ax.plot(d["snapshot_timestamp"], d["net_pnl_usd"],
                color=COLORS[pid], lw=1.8,
                label=f"{POSITION_LABELS[pid]}  (terminal: {_fmt_terminal(term)})")
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.6)
    ax.set_title("Net P&L (Cumulative Fees − Impermanent Loss) for \\$100,000 LP Positions"
                 "\n(swap-time WETH fee valuation; alternative exit-price valuation in script log)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net P&L (USD; positive = LP outperforms HODL)")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    _format_date_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  {out_path}")


print("\nWriting figures ...")
plot_fee_income(ts,        os.path.join(OUTPUT_DIR, "fig4_1_fee_income.png"))
plot_il        (ts, slot0, os.path.join(OUTPUT_DIR, "fig4_2_impermanent_loss.png"))
plot_net       (ts,        os.path.join(OUTPUT_DIR, "fig4_3_net_pnl.png"))


# ---------------------------------------------------------------------------
# 7.  VERIFICATION
# ---------------------------------------------------------------------------

print("\nVerification ...")

# 3.  Entry values within $1 of $100,000
err = (positions_df["entry_value_usd"] - ENTRY_USD).abs().max()
assert err < 1.0, f"Entry-value error too large: ${err:.6f}"
print(f"  [OK] Entry values within $1 of $100,000  (max err ${err:.4e})")

# 4.  Tick alignment + entry-tick containment
for p in positions:
    assert p["tick_lower"] % TICK_SPACING == 0, p["position_id"]
    assert p["tick_upper"] % TICK_SPACING == 0, p["position_id"]
    assert p["tick_lower"] <= entry_tick < p["tick_upper"], (
        f"entry tick {entry_tick} not in [{p['tick_lower']}, {p['tick_upper']})"
        f" for {p['position_id']}")
print(f"  [OK] All tick boundaries are multiples of {TICK_SPACING}"
      f" and contain entry tick {entry_tick}")

# 5.  Series shape
assert ts.shape[0] == len(positions) * len(slot0), (
    f"Wrong shape: {ts.shape[0]} vs {len(positions) * len(slot0)}")
print(f"  [OK] lp_timeseries has {ts.shape[0]} = {len(positions)} x {len(slot0)} rows")

# 6.  Cumulative fees non-decreasing
mono = ts.groupby("position_id")["cumulative_fee_usd"].apply(lambda s: s.is_monotonic_increasing)
assert mono.all(), f"Non-monotone fees: {mono[~mono].index.tolist()}"
print("  [OK] Cumulative fees are non-decreasing for all positions")

# 7.  P5 is always in range
p5 = next(p for p in positions if p["position_id"] == "P5")
slot0_ticks = slot0["current_tick"].astype(int)
assert ((slot0_ticks >= p5["tick_lower"]) & (slot0_ticks < p5["tick_upper"])).all(), (
    "P5 (full range) was not always in-range")
print("  [OK] P5 (full range) is in-range at every snapshot")

# 8.  IL ~ 0 at entry
entry_il = ts[ts["snapshot_block"] == entry_block]["impermanent_loss_usd"].abs().max()
assert entry_il < 1.0, f"IL at entry not ~0: ${entry_il}"
print(f"  [OK] IL at entry snapshot is ~ 0  (max ${entry_il:.4e})")

# 9.  Net identity
diff = (ts["net_pnl_usd"] - (ts["cumulative_fee_usd"] - ts["impermanent_loss_usd"])).abs().max()
assert diff < 1e-6, f"Net identity broken: {diff}"
print(f"  [OK] net_pnl_usd = cumulative_fee_usd - impermanent_loss_usd  (max diff {diff:.2e})")

# 11. Share-exceeds-1 spot check
print("\n  Per-position L_p / swap.liquidity (max share):")
for pid, sh in max_shares.items():
    flag = "  <-- exceeds 1 (synthetic LP concentrated above realised depth)" if sh > 1.0 else ""
    print(f"    {pid}: {sh:>7.3f}{flag}")

print("\nDone.")
