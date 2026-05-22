"""
liquidity_analysis.py
=====================
Liquidity Distribution Analysis

Produces:
  Fig 2.1  – Liquidity profile bar charts at three snapshots (start, high-vol, end)
  Fig 2.2  – Stacked area chart of TVL decomposition (in-range / above / below)
  Fig 2.3  – ILR(k) time series for k ∈ {0.1%, 0.5%, 1%, 2%, 5%}
  Fig 2.4  – L-HHI time series overlaid with ETH price (dual y-axis)

Input files (Parquet, from Module 1):
  liquidity_snapshots.parquet
  slot0_snapshots.parquet

All amounts are in USDC/WETH units as documented in the data dictionary.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0.  CONFIGURATION
# ---------------------------------------------------------------------------

DATA_DIR   = "output"          # directory containing the parquet files
OUTPUT_DIR = "figures"    # where PNGs are saved

LIQUIDITY_SNAP = os.path.join(DATA_DIR, "liquidity_snapshots.parquet")
SLOT0_SNAP     = os.path.join(DATA_DIR, "slot0_snapshots.parquet")

# Uniswap V3 pool constants
TICK_SPACING   = 10
FEE_TIER       = 0.0005   # 0.05 %
Q96            = 2 ** 96

# token decimals
DECIMALS_USDC  = 6
DECIMALS_WETH  = 18
DECIMAL_ADJ    = 10 ** (DECIMALS_WETH - DECIMALS_USDC)   # = 1e12

# ILR bandwidths
ILR_BANDS = [0.001, 0.005, 0.01, 0.02, 0.05]   # 0.1 %, 0.5 %, 1 %, 2 %, 5 %

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  LOAD DATA
# ---------------------------------------------------------------------------

print("Loading parquet files …")
liq_df   = pd.read_parquet(LIQUIDITY_SNAP)
slot0_df = pd.read_parquet(SLOT0_SNAP)

# Ensure datetime columns are parsed and sorted
liq_df["snapshot_timestamp"]   = pd.to_datetime(liq_df["snapshot_timestamp"], utc=True)
slot0_df["snapshot_timestamp"] = pd.to_datetime(slot0_df["snapshot_timestamp"], utc=True)

liq_df   = liq_df.sort_values(["snapshot_timestamp", "tick"]).reset_index(drop=True)
slot0_df = slot0_df.sort_values("snapshot_timestamp").reset_index(drop=True)

# Unique snapshot dates (as date objects for human-readable labelling)
snap_blocks = slot0_df["snapshot_block"].unique()
snap_dates  = slot0_df.set_index("snapshot_block")["snapshot_timestamp"]

print(f"  Loaded {len(snap_blocks)} daily snapshots "
      f"({slot0_df['snapshot_timestamp'].min().date()} → "
      f"{slot0_df['snapshot_timestamp'].max().date()})")

# ---------------------------------------------------------------------------
# 2.  HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def tick_to_price(tick: int) -> float:
    """Convert a Uniswap V3 tick index to USDC/WETH price (decimal-adjusted)."""
    return (1.0001 ** tick) * DECIMAL_ADJ


def sqrt_price_x96_to_price(sqrt_price_x96_str: str) -> float:
    """Convert sqrtPriceX96 string → USDC/WETH float."""
    sqp = int(sqrt_price_x96_str)
    return (sqp / Q96) ** 2 * DECIMAL_ADJ


def token_amounts_from_liquidity(L, sqrt_pa_raw, sqrt_pb_raw, sqrt_pc_raw):
    """
    Uniswap V3 virtual reserve formulas.

    All sqrt-price arguments must be in **raw** (non-decimal-adjusted) units,
    i.e.  sqrt_p_raw = sqrt(1.0001^tick)  with NO extra decimal factor.

    Formulas (from Uniswap V3 whitepaper, §6.2–6.3):

      Case 1 — current price below range  (pc ≤ pa):
        raw_x = L · (1/√pa − 1/√pb)   [all token0 = USDC]
        raw_y = 0

      Case 2 — current price above range  (pc ≥ pb):
        raw_x = 0
        raw_y = L · (√pb − √pa)        [all token1 = WETH]

      Case 3 — current price in range  (pa < pc < pb):
        raw_x = L · (1/√pc − 1/√pb)
        raw_y = L · (√pc − √pa)

    The raw amounts are then converted to human-readable token units:
        amount_usdc = raw_x / 1e6          (USDC has 6 decimals)
        amount_weth = raw_y / 1e18         (WETH has 18 decimals)

    USD value = amount_usdc + amount_weth * price_usdc_per_weth
    """
    if sqrt_pc_raw <= sqrt_pa_raw:
        # entirely above current price → all USDC
        raw_x = L * (1.0 / sqrt_pa_raw - 1.0 / sqrt_pb_raw)
        raw_y = 0.0
    elif sqrt_pc_raw >= sqrt_pb_raw:
        # entirely below current price → all WETH
        raw_x = 0.0
        raw_y = L * (sqrt_pb_raw - sqrt_pa_raw)
    else:
        # current price is inside the range
        raw_x = L * (1.0 / sqrt_pc_raw - 1.0 / sqrt_pb_raw)
        raw_y = L * (sqrt_pc_raw - sqrt_pa_raw)

    amount_usdc = raw_x / 10 ** DECIMALS_USDC
    amount_weth = raw_y / 10 ** DECIMALS_WETH
    return amount_usdc, amount_weth


def compute_tvl_for_snapshot(snap_liq: pd.DataFrame,
                              current_tick: int,
                              sqrt_pc_raw: float,
                              price_usdc: float) -> dict:
    """
    Compute TVL decomposition for a single snapshot.

    Parameters
    ----------
    snap_liq    : rows from liquidity_snapshots for this snapshot
    current_tick: pool's active tick at snapshot (from slot0)
    sqrt_pc_raw : sqrt(price) in raw (non-decimal-adjusted) units,
                  i.e. sqrt(1.0001^current_tick)
    price_usdc  : USDC/WETH price (decimal-adjusted, for USD conversion)

    Returns
    -------
    dict with keys: in_range_usd, above_usd, below_usd, total_usd
    """
    in_range_usd = 0.0
    above_usd    = 0.0
    below_usd    = 0.0

    # We iterate over each initialised tick range.
    # Each tick row represents the range [tick, tick + TICK_SPACING).
    # active_liquidity is the liquidity in that range.
    for _, row in snap_liq.iterrows():
        tick_lo = int(row["tick"])
        tick_hi = tick_lo + TICK_SPACING

        L       = float(row["active_liquidity"])
        if L <= 0:
            continue

        # Raw sqrt prices at range boundaries — NO decimal adjustment here.
        # The decimal correction is applied inside token_amounts_from_liquidity.
        sqrt_pa_raw = np.sqrt(1.0001 ** tick_lo)
        sqrt_pb_raw = np.sqrt(1.0001 ** tick_hi)

        amt_usdc, amt_weth = token_amounts_from_liquidity(
            L, sqrt_pa_raw, sqrt_pb_raw, sqrt_pc_raw
        )
        usd_val = amt_usdc + amt_weth * price_usdc

        if tick_lo <= current_tick < tick_hi:
            in_range_usd += usd_val
        elif tick_lo >= current_tick + TICK_SPACING:
            # entire range strictly above current price
            above_usd += usd_val
        else:
            # entire range strictly below current price
            below_usd += usd_val

    return {
        "in_range_usd": in_range_usd,
        "above_usd":    above_usd,
        "below_usd":    below_usd,
        "total_usd":    in_range_usd + above_usd + below_usd,
    }


# ---------------------------------------------------------------------------
# 3.  PRE-COMPUTE PER-SNAPSHOT METRICS
# ---------------------------------------------------------------------------

print("Computing per-snapshot metrics …")

records = []   # one dict per snapshot

slot0_indexed = slot0_df.set_index("snapshot_block")

for block in snap_blocks:
    s0   = slot0_indexed.loc[block]
    ts   = s0["snapshot_timestamp"]
    price_usdc  = float(s0["price_usdc_per_weth"])
    current_tick = int(s0["current_tick"])

    sqrt_pc_raw = np.sqrt(1.0001 ** current_tick)   # raw, no decimal adjustment

    snap_liq = liq_df[liq_df["snapshot_block"] == block].copy()

    # ---- TVL decomposition ----
    tvl = compute_tvl_for_snapshot(snap_liq, current_tick, sqrt_pc_raw, price_usdc)

    # ---- ILR(k) ----
    total_liq = snap_liq["active_liquidity"].astype(float).sum()
    ilr = {}
    for k in ILR_BANDS:
        lo = price_usdc * (1 - k)
        hi = price_usdc * (1 + k)
        # Strict containment: the tick interval [price_upper, price_lower] (low → high
        # in USDC/WETH; remember USDC=token0 so a higher V3 tick gives a LOWER
        # USDC/WETH value, hence price_upper < price_lower numerically) is entirely
        # inside [lo, hi]. This matches the report formula in §2.3.
        mask = (snap_liq["price_upper"] >= lo) & (snap_liq["price_lower"] <= hi)
        band_liq = snap_liq.loc[mask, "active_liquidity"].astype(float).sum()
        ilr[k] = band_liq / total_liq if total_liq > 0 else 0.0

    # ---- L-HHI ----
    # Computing HHI directly on per-tick active_liquidity rows is misleading:
    # a wide LP position spanning N ticks contributes to N rows, so wide
    # positions are over-counted relative to narrow ones, causing artefact spikes.
    #
    # Fix: aggregate into coarse price buckets (each bucket = 100 tick-spacings
    # = 1000 ticks wide) within a ±40 % price window, then compute HHI on the
    # bucketed shares. This treats each price region as one unit of concentration
    # regardless of how many tick rows it spans.
    lo_hhi = price_usdc * 0.60
    hi_hhi = price_usdc * 1.40
    # Same strict-containment convention as the ILR mask above
    # (price_upper < price_lower numerically because USDC=token0).
    hhi_mask = (snap_liq["price_upper"] >= lo_hhi) & (snap_liq["price_lower"] <= hi_hhi)
    hhi_sub = snap_liq.loc[hhi_mask].copy()

    BUCKET_TICKS = 100 * TICK_SPACING   # 1000 ticks per bucket
    if len(hhi_sub) > 0:
        hhi_sub["bucket"] = (hhi_sub["tick"] // BUCKET_TICKS) * BUCKET_TICKS
        bucket_liq = hhi_sub.groupby("bucket")["active_liquidity"].apply(
            lambda x: x.map(float).mean()
        )
        bucket_liq = bucket_liq[bucket_liq > 0]
        if bucket_liq.sum() > 0:
            shares = bucket_liq / bucket_liq.sum()
            l_hhi  = float((shares ** 2).sum())
        else:
            l_hhi = 0.0
    else:
        l_hhi = 0.0

    rec = {
        "snapshot_block":     block,
        "snapshot_timestamp": ts,
        "price_usdc_per_weth": price_usdc,
        "current_tick":       current_tick,
        "in_range_usd":       tvl["in_range_usd"],
        "above_usd":          tvl["above_usd"],
        "below_usd":          tvl["below_usd"],
        "total_usd":          tvl["total_usd"],
        "l_hhi":              l_hhi,
    }
    for k in ILR_BANDS:
        rec[f"ilr_{k}"] = ilr[k]

    records.append(rec)

metrics_df = pd.DataFrame(records).sort_values("snapshot_timestamp").reset_index(drop=True)
print(f"  Done. {len(metrics_df)} snapshots processed.")

# ---------------------------------------------------------------------------
# 4.  SELECT THREE SNAPSHOTS FOR FIG 2.1
# ---------------------------------------------------------------------------

# Start of window
start_snap = metrics_df.iloc[0]

# End of window
end_snap   = metrics_df.iloc[-1]

# High-volatility day: snapshot with the largest absolute daily price change
metrics_df["price_change"] = metrics_df["price_usdc_per_weth"].diff().abs()
hv_idx  = metrics_df["price_change"].idxmax()
hv_snap = metrics_df.loc[hv_idx]

selected = {
    "Start\n" + start_snap["snapshot_timestamp"].strftime("%Y-%m-%d"): int(start_snap["snapshot_block"]),
    "High Volatility\n" + hv_snap["snapshot_timestamp"].strftime("%Y-%m-%d"):  int(hv_snap["snapshot_block"]),
    "End\n" + end_snap["snapshot_timestamp"].strftime("%Y-%m-%d"):   int(end_snap["snapshot_block"]),
}

# ---------------------------------------------------------------------------
# 5.  FIGURE 2.1 — LIQUIDITY PROFILE AT THREE SNAPSHOTS
# ---------------------------------------------------------------------------

print("Generating Fig 2.1 …")

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
fig.suptitle(
    "Fig 2.1 — Liquidity Profile at Three Snapshots\n"
    "Uniswap V3 USDC/WETH 0.05 % Pool",
    fontsize=14, fontweight="bold", y=1.02
)

for ax, (label, block) in zip(axes, selected.items()):
    snap_liq = liq_df[liq_df["snapshot_block"] == block].copy()
    s0_row   = slot0_indexed.loc[block]
    cur_price = float(s0_row["price_usdc_per_weth"])

    # Focus on a ±20% window around current price to keep chart readable
    lo_price = cur_price * 0.80
    hi_price = cur_price * 1.20
    snap_liq = snap_liq[
        (snap_liq["price_lower"] >= lo_price) &
        (snap_liq["price_upper"] <= hi_price)
    ]

    prices = 0.5 * (snap_liq["price_lower"] + snap_liq["price_upper"])
    liqs   = snap_liq["active_liquidity"].astype(float)

    ax.bar(
        prices, liqs,
        width=(snap_liq["price_upper"] - snap_liq["price_lower"]),
        color="steelblue", alpha=0.75, linewidth=0.3, edgecolor="steelblue"
    )
    ax.axvline(cur_price, color="crimson", linewidth=1.6, linestyle="--", label=f"Price: ${cur_price:,.0f}")
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("Price (USDC / WETH)", fontsize=10)
    ax.set_ylabel("Active Liquidity (liquidity units)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e12:.1f}T" if x >= 1e12 else f"{x/1e9:.0f}B"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.35, linestyle=":")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
fig21_path = os.path.join(OUTPUT_DIR, "fig2_1_liquidity_profile.png")
plt.savefig(fig21_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {fig21_path}")

# ---------------------------------------------------------------------------
# 6.  FIGURE 2.1b — TILED TIME-SERIES OF LIQUIDITY PROFILES (all snapshots)
# ---------------------------------------------------------------------------

print("Generating Fig 2.1 tiled time-series …")

# Split the 182-day window into two chunks (Oct–Dec and Jan–Mar) and emit one
# landscape PNG per chunk. This keeps each tile large enough to read while
# preserving "all snapshots" coverage required by Task 2.1.
split_ts   = pd.Timestamp("2026-01-01", tz="UTC")
chunks     = [
    ("Oct–Dec 2025", metrics_df[metrics_df["snapshot_timestamp"] <  split_ts]),
    ("Jan–Mar 2026", metrics_df[metrics_df["snapshot_timestamp"] >= split_ts]),
]

for chunk_idx, (chunk_label, chunk_df) in enumerate(chunks, start=1):
    chunk_df = chunk_df.reset_index(drop=True)
    n_snaps  = len(chunk_df)
    # 12-wide grid keeps tiles square-ish (8.4" × 8.4" landscape-page area / 12 ≈ 0.7" tile)
    ncols    = 12
    nrows    = int(np.ceil(n_snaps / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 1.9 * nrows))
    axes = np.array(axes).flatten()

    fig.suptitle(
        f"Fig 2.1b ({chunk_idx}/2) — Liquidity Profile Evolution: {chunk_label}\n"
        "Uniswap V3 USDC/WETH 0.05 % Pool",
        fontsize=14, fontweight="bold"
    )

    for i, row in chunk_df.iterrows():
        ax    = axes[i]
        block = int(row["snapshot_block"])
        cur_price = row["price_usdc_per_weth"]

        snap_liq = liq_df[liq_df["snapshot_block"] == block].copy()
        lo_price  = cur_price * 0.85
        hi_price  = cur_price * 1.15
        snap_liq  = snap_liq[(snap_liq["price_lower"] >= lo_price) & (snap_liq["price_upper"] <= hi_price)]

        prices = 0.5 * (snap_liq["price_lower"] + snap_liq["price_upper"])
        liqs   = snap_liq["active_liquidity"].astype(float)

        ax.bar(prices, liqs,
               width=(snap_liq["price_upper"] - snap_liq["price_lower"]),
               color="steelblue", alpha=0.7, linewidth=0)
        ax.axvline(cur_price, color="crimson", linewidth=1.0, linestyle="--")
        ax.set_title(row["snapshot_timestamp"].strftime("%b %d"), fontsize=8)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax.tick_params(axis="x", labelsize=7, rotation=45)

    for j in range(n_snaps, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig21b_path = os.path.join(OUTPUT_DIR, f"fig2_1b_liquidity_profile_tiled_p{chunk_idx}.png")
    plt.savefig(fig21b_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fig21b_path}")

# ---------------------------------------------------------------------------
# 7.  FIGURE 2.2 — TVL DECOMPOSITION (stacked area chart)
# ---------------------------------------------------------------------------

print("Generating Fig 2.2 …")

dates    = metrics_df["snapshot_timestamp"].values
in_range = metrics_df["in_range_usd"].values / 1e6    # convert to M USD
above    = metrics_df["above_usd"].values    / 1e6
below    = metrics_df["below_usd"].values    / 1e6

fig, ax = plt.subplots(figsize=(14, 6))

ax.stackplot(
    dates,
    in_range, above, below,
    labels=["In-Range TVL (active, earning fees)",
            "Out-of-Range TVL — Above (100 % USDC)",
            "Out-of-Range TVL — Below (100 % WETH)"],
    colors=["#2ca02c", "#1f77b4", "#ff7f0e"],
    alpha=0.80
)

ax.set_title(
    "Fig 2.2 — TVL Decomposition: In-Range vs Out-of-Range\n"
    "Uniswap V3 USDC/WETH 0.05 % Pool  |  Oct 2025 – Mar 2026",
    fontsize=13, fontweight="bold"
)
ax.set_xlabel("Date (UTC)", fontsize=11)
ax.set_ylabel("TVL (USD millions)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
ax.tick_params(axis="x", rotation=45)
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3, linestyle=":")
plt.tight_layout()

fig22_path = os.path.join(OUTPUT_DIR, "fig2_2_tvl_decomposition.png")
plt.savefig(fig22_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {fig22_path}")

# ---------------------------------------------------------------------------
# 8.  FIGURE 2.3 — ILR(k) TIME SERIES
# ---------------------------------------------------------------------------

print("Generating Fig 2.3 …")

BAND_LABELS = {
    0.001: "±0.1 %",
    0.005: "±0.5 %",
    0.01:  "±1 %",
    0.02:  "±2 %",
    0.05:  "±5 %",
}
BAND_COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]

fig, ax = plt.subplots(figsize=(14, 6))

for k, color in zip(ILR_BANDS, BAND_COLORS):
    ax.plot(
        metrics_df["snapshot_timestamp"],
        metrics_df[f"ilr_{k}"] * 100,
        label=f"ILR {BAND_LABELS[k]}",
        color=color, linewidth=1.6
    )

ax.set_title(
    "Fig 2.3 — In-Range Liquidity Ratio ILR(k) by Bandwidth\n"
    "Uniswap V3 USDC/WETH 0.05 % Pool  |  Oct 2025 – Mar 2026",
    fontsize=13, fontweight="bold"
)
ax.set_xlabel("Date (UTC)", fontsize=11)
ax.set_ylabel("ILR (% of total active liquidity)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f} %"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle=":")
ax.set_ylim(bottom=0)
plt.tight_layout()

fig23_path = os.path.join(OUTPUT_DIR, "fig2_3_ilr_series.png")
plt.savefig(fig23_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {fig23_path}")

# ---------------------------------------------------------------------------
# 9.  FIGURE 2.4 — L-HHI OVERLAID WITH ETH PRICE (dual y-axis)
# ---------------------------------------------------------------------------

print("Generating Fig 2.4 …")

fig, ax1 = plt.subplots(figsize=(14, 6))

color_hhi   = "#d62728"
color_price = "#1f77b4"

ax1.plot(
    metrics_df["snapshot_timestamp"],
    metrics_df["l_hhi"],
    color=color_hhi, linewidth=1.8, label="L-HHI"
)
ax1.set_ylabel("Liquidity HHI (L-HHI)", fontsize=11, color=color_hhi)
ax1.tick_params(axis="y", labelcolor=color_hhi)
ax1.set_ylim(bottom=0)

ax2 = ax1.twinx()
ax2.plot(
    metrics_df["snapshot_timestamp"],
    metrics_df["price_usdc_per_weth"],
    color=color_price, linewidth=1.4, linestyle="--", alpha=0.85, label="ETH Price"
)
ax2.set_ylabel("ETH Price (USDC / WETH)", fontsize=11, color=color_price)
ax2.tick_params(axis="y", labelcolor=color_price)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

ax1.set_title(
    "Fig 2.4 — Liquidity HHI (L-HHI) vs. ETH Price\n"
    "Uniswap V3 USDC/WETH 0.05 % Pool  |  Oct 2025 – Mar 2026",
    fontsize=13, fontweight="bold"
)
ax1.set_xlabel("Date (UTC)", fontsize=11)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
ax1.tick_params(axis="x", rotation=45)
ax1.grid(alpha=0.3, linestyle=":")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")

plt.tight_layout()

fig24_path = os.path.join(OUTPUT_DIR, "fig2_4_hhi_vs_price.png")
plt.savefig(fig24_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {fig24_path}")

# ---------------------------------------------------------------------------
# 10.  SUMMARY TABLE
# ---------------------------------------------------------------------------

print("\n=== Summary Statistics ===")
print(metrics_df[["snapshot_timestamp", "price_usdc_per_weth",
                   "in_range_usd", "above_usd", "below_usd", "total_usd",
                   "l_hhi"]].describe().to_string())

print("\nAll figures saved to:", os.path.abspath(OUTPUT_DIR))
print("Done.")
