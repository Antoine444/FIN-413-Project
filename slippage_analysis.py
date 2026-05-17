"""
slippage_analysis.py
====================
Module 3 driver: runs the full simulation grid against the reconstructed pool
state, validates the simulator against observed swaps, and produces the three
required figures.

Outputs
-------
output/simulated_trades.parquet   – 7 sizes × N snapshots × 2 directions
figures/fig3_1_validation.png     – Simulator-vs-actual table (Task 3.1)
figures/fig3_2_price_impact.png   – Log-log impact curves, median + 10/90 bands
figures/fig3_3_effective_spread.png – Simulated vs. empirical effective spread

Dependencies: swap_simulator.py, pandas, numpy, matplotlib, tqdm
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

from swap_simulator import (
    simulate_swap, simulate_swap_raw, mid_price_usdc_per_weth,
    FEE_TIER, DECIMALS_USDC, DECIMALS_WETH, Q96,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR    = "output"
FIG_DIR     = "figures"
LIQ_PATH    = os.path.join(DATA_DIR, "liquidity_snapshots.parquet")
SLOT0_PATH  = os.path.join(DATA_DIR, "slot0_snapshots.parquet")
SWAPS_PATH  = os.path.join(DATA_DIR, "swap_events.parquet")
OUT_PARQUET = os.path.join(DATA_DIR, "simulated_trades.parquet")

# Task 3.2 grid
TRADE_SIZES_USD = [1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
DIRECTIONS      = ["buy_weth", "sell_weth"]

# Validation: stratified sample. The display table uses the smaller count;
# the tail-error statistics use the bigger count so the heavy right tail
# from snapshot-drift on the tick map is actually visible.
VALIDATION_DISPLAY_PER_BUCKET = 5     # rows shown in Fig 3.1
VALIDATION_STATS_PER_BUCKET   = 100   # rows used for median / p90 / p99 / max
RANDOM_SEED                   = 42

os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def bucket_of(notional: float) -> int:
    """Map a notional USD to its size-bucket index (0..6) using inclusive upper
    edges around each TRADE_SIZES_USD anchor."""
    edges = [0, 5_000, 30_000, 75_000, 175_000, 375_000, 750_000, np.inf]
    return int(np.searchsorted(edges, notional, side="right") - 1)


def fmt_money(x, _pos=None):
    if x >= 1_000_000:
        return f"${x/1e6:g}M"
    if x >= 1_000:
        return f"${x/1e3:g}K"
    return f"${x:g}"


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data …")
liq_df   = pd.read_parquet(LIQ_PATH)
slot0_df = pd.read_parquet(SLOT0_PATH)
swaps_df = pd.read_parquet(SWAPS_PATH)

slot0_df["snapshot_timestamp"] = pd.to_datetime(slot0_df["snapshot_timestamp"], utc=True)
slot0_df = slot0_df.sort_values("snapshot_block").reset_index(drop=True)

print(f"  {len(slot0_df):,} daily snapshots  ({slot0_df['snapshot_timestamp'].min().date()} "
      f"→ {slot0_df['snapshot_timestamp'].max().date()})")
print(f"  {len(liq_df):,} tick-rows  ({liq_df['snapshot_block'].nunique()} unique snapshots)")
print(f"  {len(swaps_df):,} swap events")

# Preserve on-chain log order. eth_getLogs returns logs in (block, txIndex,
# logIndex) ascending; web3.py and pandas preserve that through to parquet,
# so the original row order IS the log order. This lets us derive intra-block
# sequencing without re-extracting with explicit log_index/transaction_index
# columns. log_seq is the global chain position; intra_block_idx is the
# position within a block (0 = first swap of that block).
swaps_df["log_seq"]         = np.arange(len(swaps_df))
swaps_df["intra_block_idx"] = swaps_df.groupby("block_number").cumcount()

# Pre-group liquidity by snapshot_block for fast lookup
liq_by_block = {blk: g for blk, g in liq_df.groupby("snapshot_block", sort=False)}


# ---------------------------------------------------------------------------
# 2. Task 3.2 – Simulation grid
# ---------------------------------------------------------------------------
print(f"\nRunning {len(TRADE_SIZES_USD)} × {len(slot0_df)} × {len(DIRECTIONS)} = "
      f"{len(TRADE_SIZES_USD) * len(slot0_df) * len(DIRECTIONS):,} simulations …")

records = []
for snap_row in tqdm(slot0_df.itertuples(index=False), total=len(slot0_df)):
    snap_block = int(snap_row.snapshot_block)
    ticks      = liq_by_block.get(snap_block)
    if ticks is None or len(ticks) == 0:
        continue
    sqp_start  = int(snap_row.sqrtPriceX96)
    cur_tick   = int(snap_row.current_tick)

    for size in TRADE_SIZES_USD:
        for direction in DIRECTIONS:
            r = simulate_swap(ticks, sqp_start, cur_tick, direction, float(size))
            records.append({
                "snapshot_block":     snap_block,
                "snapshot_timestamp": snap_row.snapshot_timestamp,
                "direction":          direction,
                "notional_usd":       float(size),
                "avg_executed_price": r["avg_executed_price"],
                "mid_price":          r["mid_price"],
                "price_impact_bps":   r["price_impact_bps"],
                "slippage_bps":       r["slippage_bps"],
                "ticks_crossed":      r["ticks_crossed"],
            })

sim_df = pd.DataFrame.from_records(records)
sim_df.to_parquet(OUT_PARQUET, index=False)
print(f"  Wrote {OUT_PARQUET}  ({len(sim_df):,} rows)")


# ---------------------------------------------------------------------------
# 3. Task 3.1 – Validation table (Fig 3.1)
# ---------------------------------------------------------------------------
print("\nValidating simulator on a stratified sample of observed swaps …")

# Parse string-encoded uint160 / uint128 values as Python ints (object dtype).
# int64 cannot hold these, so we keep them as object-dtype Python ints.
swaps_df["sqrtPriceX96_int"] = swaps_df["sqrtPriceX96"].astype(str).map(int)
swaps_df["liquidity_int"]    = swaps_df["liquidity"].astype(str).map(int)

# --- Pre-state #1: pre-swap state for VALIDATION (Task 3.1) -----------------
# Not exact, but materially better than the previous block-level shift.
# The state immediately before THIS swap, in chain order, is the prior swap's
# emitted post-state. A global shift(1) along log order gives this, because
# the parquet row order IS chain order (eth_getLogs orders by block, txIndex,
# logIndex). For first-in-block swaps this resolves to the prior block's last
# swap; for later swaps in the same block it resolves to the prior swap in
# that block — fixing the bug that previously fed every swap in a block the
# same (prior-block) pre-state.
#
# Residual sources of error (visible in the right tail of the larger-sample
# validation, p99 ≈ 0.25 %, max ≈ 3 %):
#   • The tick map L(tick) used for crossings is the most recent daily
#     snapshot, not the per-block reconstruction. Mint/Burn events between
#     the snapshot and the swap shift liquidity_net at unrelated ticks; the
#     `liquidity_pre` override fixes the active-tick bucket but not the
#     surrounding ticks the simulator may cross.
#   • Mint/Burn events between two swaps in the same block also break the
#     "prior swap's emitted liquidity = pre-L of next swap" identity.
# Tail errors arise when stale daily tick maps matter, typically in swaps
# that cross ticks where Mint/Burn happened between snapshot and trade —
# rare per swap but enough to drive the right tail (p99 ≈ 0.25 %, max ≈ 3 %).
# The unconditional correlation with ticks_crossed is weak (~+0.06) because
# most multi-tick swaps still validate near-zero; only the unlucky overlap
# with intra-day Mint/Burn produces large errors.
swaps_df["sqrtPriceX96_pre_swap_int"] = swaps_df["sqrtPriceX96_int"].shift(1)
swaps_df["liquidity_pre"]             = swaps_df["liquidity_int"].shift(1)

# --- Pre-state #2: "block before" mid-price for EFFECTIVE SPREAD (Task 3.4) -
# PDF defines p_mid as "the pool price from slot0 at the block before the
# swap". I.e., the last post-state of the most recent earlier swap-block.
block_last_post  = swaps_df.groupby("block_number")["sqrtPriceX96_int"].last().sort_index()
prev_block_post  = block_last_post.shift(1)
swaps_df["sqrtPriceX96_pre_block_int"] = swaps_df["block_number"].map(prev_block_post)

# Drop the handful of swaps that lack any required pre-state (the very first
# swap in the dataset, and any swaps in the very first swap-block).
swaps_df = swaps_df.dropna(subset=[
    "sqrtPriceX96_pre_swap_int",
    "sqrtPriceX96_pre_block_int",
    "liquidity_pre",
]).copy()

# Determine the daily liquidity snapshot active at each swap (most recent snapshot_block ≤ block_number)
snap_blocks_sorted = np.sort(slot0_df["snapshot_block"].to_numpy())
def snap_block_for(bn: int) -> int:
    i = np.searchsorted(snap_blocks_sorted, bn, side="right") - 1
    return int(snap_blocks_sorted[max(i, 0)])

# Stratify by bucket and sample.
# Two samples: a small DISPLAY sample (5/bucket) for the printable Fig 3.1
# table, and a larger STATS sample (100/bucket) so the tail-error metrics
# are not hidden by the small display N.
rng = np.random.default_rng(RANDOM_SEED)
swaps_df["bucket"] = swaps_df["notional_usd"].map(bucket_of)
stats_rows, display_rows = [], []
for b in range(len(TRADE_SIZES_USD)):
    candidates = swaps_df[swaps_df["bucket"] == b]
    if len(candidates) == 0:
        continue
    take_stats   = min(VALIDATION_STATS_PER_BUCKET,   len(candidates))
    take_display = min(VALIDATION_DISPLAY_PER_BUCKET, take_stats)
    stats_sample = candidates.sample(n=take_stats, random_state=int(rng.integers(0, 1_000_000)))
    display_sample = stats_sample.head(take_display)   # subset of stats sample
    stats_rows.append(stats_sample)
    display_rows.append(display_sample)
stats_swaps   = pd.concat(stats_rows,   ignore_index=True)
display_swaps = pd.concat(display_rows, ignore_index=True)
display_keys  = set(display_swaps["log_seq"].tolist())
print(f"  Stats sample: {len(stats_swaps)} swaps across {stats_swaps['bucket'].nunique()} buckets")
print(f"  Display sample: {len(display_swaps)} swaps (subset of stats sample)")

validation_results = []
for row in stats_swaps.itertuples(index=False):
    snap_blk = snap_block_for(int(row.block_number))
    ticks    = liq_by_block.get(snap_blk)
    if ticks is None:
        continue
    sqp_pre  = int(row.sqrtPriceX96_pre_swap_int)
    L_pre    = float(row.liquidity_pre)
    # Derive the pre-state tick from the pre-state sqrtPriceX96
    # tick = floor(log_1.0001((sqp/Q96)^2)) = floor(2·log_1.0001(sqp/Q96))
    raw_p_pre = (sqp_pre / Q96) ** 2
    tick_pre  = int(np.floor(np.log(raw_p_pre) / np.log(1.0001)))

    if row.trade_direction == "buy_weth":
        # token0 in: input is |amount0_raw|
        amount_in_gross = abs(float(row.amount0_decimal)) * (10 ** DECIMALS_USDC)
        zero_for_one    = True
        actual_out      = abs(float(row.amount1_decimal)) * (10 ** DECIMALS_WETH)
    else:
        amount_in_gross = abs(float(row.amount1_decimal)) * (10 ** DECIMALS_WETH)
        zero_for_one    = False
        actual_out      = abs(float(row.amount0_decimal)) * (10 ** DECIMALS_USDC)

    # l_active_override carries the prior swap's emitted `liquidity` field,
    # which equals the active liquidity at the start of THIS swap unless a
    # Mint/Burn intervened between the two swaps (rare).
    sim = simulate_swap_raw(
        ticks, sqp_pre, tick_pre, zero_for_one, amount_in_gross,
        l_active_override=L_pre,
    )

    sim_out      = sim["amount_out_raw"]
    sim_sqp_end  = sim["sqrt_price_x96_end"]
    actual_sqp   = int(row.sqrtPriceX96_int)
    pct_err_out  = 100.0 * (sim_out - actual_out) / actual_out if actual_out > 0 else float("nan")
    pct_err_sqp  = 100.0 * (sim_sqp_end - actual_sqp) / actual_sqp if actual_sqp > 0 else float("nan")

    validation_results.append({
        "bucket":           int(row.bucket),
        "notional_usd":     float(row.notional_usd),
        "direction":        row.trade_direction,
        "block_number":     int(row.block_number),
        "intra_block_idx":  int(row.intra_block_idx),
        "log_seq":          int(row.log_seq),
        "actual_out":       actual_out,
        "sim_out":          sim_out,
        "pct_err_out":      pct_err_out,
        "actual_sqp_end":   actual_sqp,
        "sim_sqp_end":      sim_sqp_end,
        "pct_err_sqp":      pct_err_sqp,
        "ticks_crossed":    sim["ticks_crossed"],
    })

stats_df   = pd.DataFrame(validation_results)
abs_out    = stats_df["pct_err_out"].abs()
abs_sqp    = stats_df["pct_err_sqp"].abs()
print(f"  Validation tail stats over N={len(stats_df)} swaps:")
print(f"    |%err out|  median={abs_out.median():.6f}%  p90={np.percentile(abs_out, 90):.4f}%  "
      f"p99={np.percentile(abs_out, 99):.4f}%  max={abs_out.max():.4f}%")
print(f"    |%err √P|   median={abs_sqp.median():.6f}%  p90={np.percentile(abs_sqp, 90):.6f}%  "
      f"p99={np.percentile(abs_sqp, 99):.6f}%  max={abs_sqp.max():.6f}%")
ticks_corr = stats_df[["ticks_crossed"]].assign(abs_err_out=abs_out).corr().iloc[0, 1]
print(f"    ticks_crossed mean={stats_df['ticks_crossed'].mean():.2f}, "
      f"fraction crossing≥1: {(stats_df['ticks_crossed'] >= 1).mean() * 100:.1f}%, "
      f"corr(ticks_crossed, |%err_out|) = {ticks_corr:+.3f}")

# Build the display table from the small subset (same swaps, just fewer rows).
val_df = (stats_df[stats_df["log_seq"].isin(display_keys)]
          .sort_values(["bucket", "notional_usd"])
          .reset_index(drop=True))

# Render Fig 3.1 as a table
fig, ax = plt.subplots(figsize=(14, 0.4 * len(val_df) + 1.5), dpi=150)
ax.axis("off")
display_cols = [
    ("notional_usd",     "Notional (USD)",   lambda v: f"${v:,.0f}"),
    ("direction",        "Direction",         lambda v: v),
    ("intra_block_idx",  "Idx-in-blk",        lambda v: f"{int(v)}"),
    ("actual_out",       "Actual out (raw)",  lambda v: f"{v:.3e}"),
    ("sim_out",          "Sim out (raw)",     lambda v: f"{v:.3e}"),
    ("pct_err_out",      "% err (out)",       lambda v: f"{v:+.4f}%"),
    ("pct_err_sqp",      "% err (√P)",        lambda v: f"{v:+.6f}%"),
    ("ticks_crossed",    "Ticks",             lambda v: f"{v}"),
]
table_data = [[fmt(row[col]) for col, _, fmt in display_cols] for _, row in val_df.iterrows()]
col_labels = [label for _, label, _ in display_cols]
tab = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="right")
tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 1.2)
for j in range(len(col_labels)):
    tab[(0, j)].set_facecolor("#1f77b4"); tab[(0, j)].set_text_props(color="white", weight="bold")
plt.title(
    f"Fig 3.1 — Simulator validation: {len(val_df)} swaps shown (subset of N={len(stats_df)} stats sample)\n"
    f"|%err out|: median={abs_out.median():.4f}%, p90={np.percentile(abs_out,90):.4f}%, "
    f"p99={np.percentile(abs_out,99):.4f}%, max={abs_out.max():.4f}%   "
    f"(tail driven by daily-snapshot drift on the tick map for multi-tick swaps)",
    fontsize=10, pad=12,
)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_1_validation.png"), bbox_inches="tight")
plt.close()
print(f"  Wrote {FIG_DIR}/fig3_1_validation.png")


# ---------------------------------------------------------------------------
# 4. Task 3.3 – Price impact curves (Fig 3.2)
# ---------------------------------------------------------------------------
print("\nPlotting price impact curves …")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150, sharey=True)
for ax, direction, title in zip(axes, DIRECTIONS, ["Buy WETH (USDC → WETH)", "Sell WETH (WETH → USDC)"]):
    sub = sim_df[sim_df["direction"] == direction]
    # Per-snapshot thin curves
    for blk, g in sub.groupby("snapshot_block"):
        g = g.sort_values("notional_usd")
        ax.plot(g["notional_usd"], g["price_impact_bps"].abs(),
                color="#1f77b4", alpha=0.04, linewidth=0.6)
    # Median + 10/90 bands across snapshots, per trade size
    abs_impact = sub.assign(abs_imp=sub["price_impact_bps"].abs())
    stats = abs_impact.groupby("notional_usd")["abs_imp"].agg(
        p10=lambda s: np.percentile(s, 10),
        median="median",
        p90=lambda s: np.percentile(s, 90),
    ).sort_index()
    sizes = stats.index.to_numpy()
    ax.fill_between(sizes, stats["p10"], stats["p90"], color="#ff7f0e", alpha=0.25, label="10–90 % band")
    ax.plot(sizes, stats["median"], color="#d62728", linewidth=2.0, marker="o", label="Median")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Trade size (USD notional, log scale)")
    ax.set_ylabel("|Price impact| (bps, fee-inclusive)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_money))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")

plt.suptitle(
    "Fig 3.2 — Price impact curves across all daily snapshots\n"
    "(fee-INCLUSIVE total execution cost; the 5 bps floor is the pool fee. "
    "For pure impact scaling vs. CLOB √-law, see Fig 3.4's slippage_bps row.)",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_2_price_impact.png"), bbox_inches="tight")
plt.close()
print(f"  Wrote {FIG_DIR}/fig3_2_price_impact.png")


# ---------------------------------------------------------------------------
# 5. Task 3.4 – Effective spread from observed swaps (Fig 3.3)
# ---------------------------------------------------------------------------
print("\nComputing effective spread on observed swaps …")

# p_exec from amounts; p_mid from previous-block last sqrtPriceX96 (per PDF)
sqp_pre_float          = swaps_df["sqrtPriceX96_pre_block_int"].map(float)
swaps_df["p_mid"]      = (Q96 / sqp_pre_float) ** 2 * 1e12   # = 1e12 / raw_price
swaps_df["p_exec"]     = swaps_df["amount0_decimal"].abs() / swaps_df["amount1_decimal"].abs()
swaps_df["D"]          = np.where(swaps_df["trade_direction"] == "buy_weth", +1, -1)
swaps_df["ES"]         = 2.0 * swaps_df["D"] * (swaps_df["p_exec"] - swaps_df["p_mid"]) / swaps_df["p_mid"]
swaps_df["ES_bps"]     = swaps_df["ES"] * 1e4
# Half-spread: directly comparable to one-way simulated price impact.
swaps_df["half_es_bps"] = swaps_df["ES_bps"] / 2.0

# Median half-spread (ES/2) per bucket × direction — one-way cost, directly
# comparable to the simulator's one-way price impact.
emp = (swaps_df.groupby(["bucket", "trade_direction"])["half_es_bps"]
                .agg(["median", "count"]).reset_index())
emp["notional_anchor"] = emp["bucket"].map(lambda b: TRADE_SIZES_USD[b] if 0 <= b < len(TRADE_SIZES_USD) else np.nan)
emp = emp.dropna(subset=["notional_anchor"])

# Median simulated impact per (size, direction)
sim_med = (sim_df.groupby(["notional_usd", "direction"])["price_impact_bps"]
                  .apply(lambda s: np.median(s.abs())).reset_index())

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150, sharey=True)
for ax, direction, title in zip(axes, DIRECTIONS, ["Buy WETH", "Sell WETH"]):
    emp_d = emp[emp["trade_direction"] == direction].sort_values("notional_anchor")
    sim_d = sim_med[sim_med["direction"] == direction].sort_values("notional_usd")
    ax.plot(sim_d["notional_usd"], sim_d["price_impact_bps"],
            color="#d62728", marker="o", linewidth=2, label="Simulated median |impact| (one-way)")
    ax.plot(emp_d["notional_anchor"], emp_d["median"],
            color="#1f77b4", marker="s", linewidth=2, label="Empirical half-spread (ES/2, one-way)")
    for _, r in emp_d.iterrows():
        ax.annotate(f"n={int(r['count']):,}", (r["notional_anchor"], r["median"]),
                    xytext=(4, -10), textcoords="offset points", fontsize=7, color="#1f77b4")
    ax.set_xscale("log")
    ax.set_xlabel("Trade size bucket (USD notional)")
    ax.set_ylabel("One-way cost (bps)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_money))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")

plt.suptitle(
    "Fig 3.3 — Empirical half-spread (ES/2) vs. simulated one-way price impact, by trade size\n"
    r"$ES_i = 2 D_i (p_\mathrm{exec} - p_\mathrm{mid}) / p_\mathrm{mid}$; "
    "half-spread = ES$_i$/2 is the one-way cost comparable to the simulator's |impact|.",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_3_effective_spread.png"), bbox_inches="tight")
plt.close()
print(f"  Wrote {FIG_DIR}/fig3_3_effective_spread.png")


# ---------------------------------------------------------------------------
# 6. Regression tables — price-impact power law (D3.2)
# ---------------------------------------------------------------------------
# Two regressions on the same simulation grid:
#   • price_impact_bps  — fee-INCLUSIVE total execution cost (matches PDF
#     wording of "price impact"). β is mechanically flattened by the 5 bps
#     fee floor at small sizes, so the CLOB square-root comparison must use
#     the fee-net series instead.
#   • slippage_bps      — fee-NET pure price impact. This is the quantity
#     to compare to the CLOB square-root law (β = 0.5) or the V3 linear-
#     impact regime (β ≈ 1.0).
print("\nFitting log–log regressions …")

REG_OUT_PARQUET = os.path.join(DATA_DIR, "regression_results.parquet")


def loglog_fit(df: pd.DataFrame, ycol: str) -> dict:
    """Fit log10|y| = alpha + beta · log10(notional_usd) by OLS.
    Returns coefficients, R², residual standard error, and N."""
    x = np.log10(df["notional_usd"].to_numpy(dtype="float64"))
    y = np.log10(df[ycol].abs().to_numpy(dtype="float64"))
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(len(x))
    if n < 3:
        return {"alpha": np.nan, "beta": np.nan, "r_squared": np.nan, "stderr": np.nan, "n": n}
    beta, alpha = np.polyfit(x, y, 1)
    y_hat  = alpha + beta * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    stderr = float(np.sqrt(ss_res / max(n - 2, 1)))
    return {"alpha": float(alpha), "beta": float(beta), "r_squared": r2, "stderr": stderr, "n": n}


reg_rows = []
for metric_label, ycol in [("fee-incl (impact)", "price_impact_bps"),
                           ("fee-net (slippage)", "slippage_bps")]:
    for direction in DIRECTIONS:
        reg_rows.append({"metric": metric_label, "sample": direction,
                         **loglog_fit(sim_df[sim_df["direction"] == direction], ycol)})
    reg_rows.append({"metric": metric_label, "sample": "pooled",
                     **loglog_fit(sim_df, ycol)})

reg_df = pd.DataFrame(reg_rows)
reg_df.to_parquet(REG_OUT_PARQUET, index=False)
print(f"  Wrote {REG_OUT_PARQUET}")
for r in reg_rows:
    print(f"  {r['metric']:30s} {r['sample']:10s}  β={r['beta']:+.4f}  α={r['alpha']:+.4f}  "
          f"R²={r['r_squared']:.4f}  N={r['n']:,}")

# Render as a table figure (matches Fig 3.1's style).
fig, ax = plt.subplots(figsize=(13, 0.55 * len(reg_rows) + 2.5), dpi=150)
ax.axis("off")
reg_display_cols = [
    ("metric",     "Metric",         lambda v: v),
    ("sample",     "Sample",         lambda v: v),
    ("alpha",      "α (intercept)",  lambda v: f"{v:+.4f}"),
    ("beta",       "β (slope)",      lambda v: f"{v:+.4f}"),
    ("r_squared",  "R²",             lambda v: f"{v:.4f}"),
    ("stderr",     "Residual SE",    lambda v: f"{v:.4f}"),
    ("n",          "N",              lambda v: f"{int(v):,}"),
]
reg_table_data = [[fmt(row[col]) for col, _, fmt in reg_display_cols] for _, row in reg_df.iterrows()]
reg_col_labels = [label for _, label, _ in reg_display_cols]
tab = ax.table(cellText=reg_table_data, colLabels=reg_col_labels, loc="center", cellLoc="right")
tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 1.6)
for j in range(len(reg_col_labels)):
    tab[(0, j)].set_facecolor("#1f77b4"); tab[(0, j)].set_text_props(color="white", weight="bold")
# Visually separate the two metric blocks
n_per_block = len(reg_rows) // 2
for j in range(len(reg_col_labels)):
    tab[(n_per_block, j)].set_edgecolor("#1f77b4")
    tab[(n_per_block, j)].set_linewidth(1.5)
plt.title(
    r"Fig 3.4 — Log-log regression: $\log_{10}|y| = \alpha + \beta \, \log_{10}(\mathrm{notional}_\mathrm{USD})$"
    "\nfee-inclusive impact (top): β flattened by the 5 bps fee floor at small sizes — "
    "do NOT compare to CLOB √-law."
    "\nfee-net slippage (bottom): the impact-scaling quantity — compare β to 0.5 (CLOB √-law) "
    "and 1.0 (concentrated-liquidity V3 linear regime).",
    fontsize=10, pad=12,
)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_4_regression_table.png"), bbox_inches="tight")
plt.close()
print(f"  Wrote {FIG_DIR}/fig3_4_regression_table.png")

print("\nDone.")
