"""
hedge_backtest.py
=================
Module 5 — Dynamic hedging of impermanent loss (Tasks 5.1 plots, 5.3 backtest, 5.5 figure).

Inputs:
  output/lp_positions.parquet, lp_timeseries.parquet
  output/perp_prices.parquet, funding_rates.parquet   (from hyperliquid_fetch.py)
  output/slot0_snapshots.parquet

Outputs:
  figures/fig5_1a_lp_payoff.png       Task 5.1 — terminal payoff vs ETH price
  figures/fig5_1b_lp_delta_profile.png  Task 5.1 — |delta| vs price
  figures/fig5_1_hedging_results.png    Task 5.5 — residual IL & net P&L (Fig 5.1)
  output/hedge_results.parquet          hourly backtest for 15 strategy variants

Strategy: short |Δ_LP| ETH on Hyperliquid; rebalance every 1h / 4h / 24h.
Trading fee: 0.045% on notional at each rebalance (|Δq| * price).
Funding: funding_pnl = |Δ_LP| * oracle_price * funding_rate  (short receives if rate > 0).
"""

from __future__ import annotations

import math
import os
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config (aligned with lp_analytics.py)
# ---------------------------------------------------------------------------

DATA_DIR = "output"
FIG_DIR = "figures"

POSITIONS_PATH = os.path.join(DATA_DIR, "lp_positions.parquet")
TIMESERIES_PATH = os.path.join(DATA_DIR, "lp_timeseries.parquet")
PERP_PATH = os.path.join(DATA_DIR, "perp_prices.parquet")
FUNDING_PATH = os.path.join(DATA_DIR, "funding_rates.parquet")
HEDGE_OUT = os.path.join(DATA_DIR, "hedge_results.parquet")

FIG_PAYOFF = os.path.join(FIG_DIR, "fig5_1a_lp_payoff.png")
FIG_DELTA = os.path.join(FIG_DIR, "fig5_1b_lp_delta_profile.png")
FIG_RESULTS = os.path.join(FIG_DIR, "fig5_1_hedging_results.png")

DECIMAL_ADJ = 1e12
DECIMALS_USDC = 6
DECIMALS_WETH = 18
TRADING_FEE = 0.00045
REBALANCE_HOURS = (1, 4, 24)

STUDY_START = pd.Timestamp("2025-10-01", tz="UTC")
STUDY_END = pd.Timestamp("2026-04-01", tz="UTC")

COLORS = {"P1": "#d62728", "P2": "#ff7f0e", "P3": "#2ca02c",
          "P4": "#1f77b4", "P5": "#9467bd"}

os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Uniswap V3 math (same conventions as lp_analytics.py)
# ---------------------------------------------------------------------------

def sqrt_price_at_tick(tick: int) -> float:
    return 1.0001 ** (tick / 2.0)


def sqrt_price_from_human(p_usdc_per_weth: float) -> float:
    """V3-native sqrt(WETH_raw/USDC_raw) from human USDC/WETH price."""
    return math.sqrt(DECIMAL_ADJ / p_usdc_per_weth)


def token_amounts_from_liquidity(L, sqrt_pa, sqrt_pb, sqrt_pc):
    """Returns (usdc, weth) human-readable."""
    if sqrt_pc <= sqrt_pa:
        raw_x = L * (1.0 / sqrt_pa - 1.0 / sqrt_pb)
        raw_y = 0.0
    elif sqrt_pc >= sqrt_pb:
        raw_x = 0.0
        raw_y = L * (sqrt_pb - sqrt_pa)
    else:
        raw_x = L * (1.0 / sqrt_pc - 1.0 / sqrt_pb)
        raw_y = L * (sqrt_pc - sqrt_pa)
    return raw_x / 10 ** DECIMALS_USDC, raw_y / 10 ** DECIMALS_WETH


def v_lp_usd(L, sqrt_pa, sqrt_pb, p_usdc_per_weth) -> float:
    sqrt_pc = sqrt_price_from_human(p_usdc_per_weth)
    usdc, weth = token_amounts_from_liquidity(L, sqrt_pa, sqrt_pb, sqrt_pc)
    return weth * p_usdc_per_weth + usdc


def lp_delta_eth(L, sqrt_pa, sqrt_pb, p_usdc_per_weth) -> float:
    """Analytical d(V_LP)/d(p) in ETH units (WETH exposure), Task 5.1.2."""
    sqrt_pc = sqrt_price_from_human(p_usdc_per_weth)
    if sqrt_pc <= sqrt_pa:
        return 0.0
    if sqrt_pc >= sqrt_pb:
        weth = L * (sqrt_pb - sqrt_pa) / 10 ** DECIMALS_WETH
        return weth
    usdc, weth = token_amounts_from_liquidity(L, sqrt_pa, sqrt_pb, sqrt_pc)
    d_sqrt = -0.5 * sqrt_pc / p_usdc_per_weth
    d_weth = L / 10 ** DECIMALS_WETH * d_sqrt
    d_usdc = L / 10 ** DECIMALS_USDC * (0.5 / (p_usdc_per_weth * sqrt_pc))
    return weth + p_usdc_per_weth * d_weth + d_usdc


def lp_gamma_eth(L, sqrt_pa, sqrt_pb, p_usdc_per_weth, eps: float = 1e-4) -> float:
    """Numerical gamma for validation / report (Task 5.1.3)."""
    p0 = p_usdc_per_weth
    d1 = lp_delta_eth(L, sqrt_pa, sqrt_pb, p0 * (1 + eps))
    d0 = lp_delta_eth(L, sqrt_pa, sqrt_pb, p0 * (1 - eps))
    return (d1 - d0) / (2 * eps * p0)


# ---------------------------------------------------------------------------
# Task 5.1 — Payoff and delta profile figures
# ---------------------------------------------------------------------------

def plot_payoff_and_delta(positions: pd.DataFrame, entry_price: float) -> None:
    p_grid = np.linspace(0.5 * entry_price, 1.5 * entry_price, 300)

    fig, ax = plt.subplots(figsize=(11, 6))
    for _, row in positions.iterrows():
        pid = row["position_id"]
        L = float(row["L"])
        sa = sqrt_price_at_tick(int(row["tick_lower"]))
        sb = sqrt_price_at_tick(int(row["tick_upper"]))
        x0, y0 = row["x0_weth"], row["y0_usdc"]
        il_curve = []
        for p in p_grid:
            v_lp = v_lp_usd(L, sa, sb, p)
            v_hodl = x0 * p + y0
            il_curve.append(v_hodl - v_lp)
        ax.plot(p_grid, il_curve, color=COLORS[pid], lw=2, label=f"{pid}")
    ax.axvline(entry_price, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Terminal ETH price (USDC per WETH)")
    ax.set_ylabel("Impermanent loss at terminal price  IL = V_HODL − V_LP  (USD)")
    ax.set_title("LP Payoff Shape (no fees): IL vs terminal ETH price")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_PAYOFF, dpi=130)
    plt.close(fig)
    print(f"  {FIG_PAYOFF}")

    fig, ax = plt.subplots(figsize=(11, 6))
    for _, row in positions.iterrows():
        pid = row["position_id"]
        L = float(row["L"])
        sa = sqrt_price_at_tick(int(row["tick_lower"]))
        sb = sqrt_price_at_tick(int(row["tick_upper"]))
        deltas = [abs(lp_delta_eth(L, sa, sb, p)) for p in p_grid]
        ax.plot(p_grid, deltas, color=COLORS[pid], lw=2, label=f"{pid}")
    ax.axvline(entry_price, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("ETH price (USDC per WETH)")
    ax.set_ylabel("|Δ_LP|  (ETH to short-hedge)")
    ax.set_title("LP Delta Profile  |∂V_LP/∂p|  (hedge size)")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DELTA, dpi=130)
    plt.close(fig)
    print(f"  {FIG_DELTA}")


# ---------------------------------------------------------------------------
# Task 5.3 — Delta-hedging backtest
# ---------------------------------------------------------------------------

def run_backtest(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    funding: pd.DataFrame,
    daily_fees: pd.DataFrame,
) -> pd.DataFrame:
    """15 variants: 5 positions × rebalance {1h, 4h, 24h}."""
    prices = prices.sort_values("timestamp").reset_index(drop=True)
    funding = funding.set_index("timestamp").sort_index()["funding_rate"]
    records = []

    for _, pos in positions.iterrows():
        pid = pos["position_id"]
        L = float(pos["L"])
        sa = sqrt_price_at_tick(int(pos["tick_lower"]))
        sb = sqrt_price_at_tick(int(pos["tick_upper"]))
        x0, y0 = pos["x0_weth"], pos["y0_usdc"]
        fee_daily = daily_fees[daily_fees["position_id"] == pid].sort_values("timestamp")

        for reb_h in REBALANCE_HOURS:
            strategy_id = f"{pid}_{reb_h}h"
            q_short = 0.0
            cum_hedge = cum_funding = cum_trade_fee = 0.0
            p_prev = None

            for i, row in prices.iterrows():
                t = row["timestamp"]
                p = float(row["close"])
                fund_rate = float(funding.get(t, 0.0))

                delta = abs(lp_delta_eth(L, sa, sb, p))
                v_lp = v_lp_usd(L, sa, sb, p)
                v_hodl = x0 * p + y0
                gross_il = v_hodl - v_lp

                if p_prev is not None:
                    cum_hedge += -q_short * (p - p_prev)
                    cum_funding += q_short * p_prev * fund_rate

                if i == 0 or (i % reb_h == 0):
                    q_new = delta
                    if i > 0:
                        cum_trade_fee += abs(q_new - q_short) * p * TRADING_FEE
                    q_short = q_new

                net_hedge = cum_hedge + cum_funding - cum_trade_fee
                residual_il = gross_il - net_hedge

                fee_row = fee_daily[fee_daily["timestamp"] <= t]
                cum_fee = float(fee_row["cumulative_fee_usd"].iloc[-1]) if len(fee_row) else 0.0
                net_position_pnl = cum_fee - residual_il

                records.append({
                    "strategy_id": strategy_id,
                    "position_id": pid,
                    "rebalance_hours": reb_h,
                    "timestamp": t,
                    "eth_price": p,
                    "v_lp_usd": v_lp,
                    "v_hodl_usd": v_hodl,
                    "gross_il_usd": gross_il,
                    "lp_delta_eth": delta,
                    "hedge_size_eth": q_short,
                    "hedge_pnl_cum_usd": cum_hedge,
                    "funding_pnl_cum_usd": cum_funding,
                    "trading_fees_cum_usd": cum_trade_fee,
                    "net_hedge_pnl_usd": net_hedge,
                    "residual_il_usd": residual_il,
                    "cumulative_fee_usd": cum_fee,
                    "net_position_pnl_usd": net_position_pnl,
                })
                p_prev = p

    return pd.DataFrame(records)


def plot_hedging_results(results: pd.DataFrame) -> None:
    """Fig 5.1 — residual IL (daily rebalance) and terminal net P&L heatmap-style bars."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    ax0 = axes[0]
    for pid in ["P1", "P2", "P3", "P4", "P5"]:
        d = results[(results["position_id"] == pid) & (results["rebalance_hours"] == 24)]
        ax0.plot(d["timestamp"], d["residual_il_usd"], color=COLORS[pid], lw=1.5,
                 label=f"{pid} (24h rebalance)")
    ax0.set_ylabel("Residual IL (USD)")
    ax0.set_title("Residual IL after Delta Hedge  (gross IL − net hedge P&L)")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.xaxis.set_major_locator(mdates.MonthLocator())
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    ax1 = axes[1]
    terminal = (
        results.sort_values("timestamp")
        .groupby(["position_id", "rebalance_hours"], as_index=False)
        .last()
    )
    x_labels = [f"{r.position_id}\n{r.rebalance_hours}h" for r in terminal.itertuples()]
    colors_bar = [COLORS[r.position_id] for r in terminal.itertuples()]
    ax1.bar(range(len(terminal)), terminal["net_position_pnl_usd"], color=colors_bar, alpha=0.85)
    ax1.set_xticks(range(len(terminal)))
    ax1.set_xticklabels(x_labels, fontsize=7, rotation=0)
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_ylabel("Terminal net position P&L (USD)")
    ax1.set_title("Terminal Net P&L  (cumulative LP fees − residual IL)")
    ax1.grid(True, axis="y", ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(FIG_RESULTS, dpi=130)
    plt.close(fig)
    print(f"  {FIG_RESULTS}")


def main() -> None:
    for path in (POSITIONS_PATH, TIMESERIES_PATH, PERP_PATH, FUNDING_PATH):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {path}. Run lp_analytics.py and hyperliquid_fetch.py first."
            )

    positions = pd.read_parquet(POSITIONS_PATH)
    ts = pd.read_parquet(TIMESERIES_PATH)
    ts["snapshot_timestamp"] = pd.to_datetime(ts["snapshot_timestamp"], utc=True)
    entry_price = float(
        ts.sort_values("snapshot_timestamp").groupby("position_id").first()
        .loc["P1", "price_usdc_per_weth"]
    )

    print("Task 5.1 — LP payoff and delta figures ...")
    plot_payoff_and_delta(positions, entry_price)

    print("Task 5.3 — delta-hedging backtest (15 variants) ...")
    prices = pd.read_parquet(PERP_PATH)
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    funding = pd.read_parquet(FUNDING_PATH)
    funding["timestamp"] = pd.to_datetime(funding["timestamp"], utc=True)

    daily_fees = ts[["position_id", "snapshot_timestamp", "cumulative_fee_usd"]].rename(
        columns={"snapshot_timestamp": "timestamp"}
    )

    results = run_backtest(positions, prices, funding, daily_fees)
    results.to_parquet(HEDGE_OUT, index=False)
    print(f"  Wrote {HEDGE_OUT}  ({len(results):,} rows)")

    print("Task 5.5 — hedging results figure ...")
    plot_hedging_results(results)

    print("\nTerminal net position P&L (USD):")
    term = results.sort_values("timestamp").groupby("strategy_id").last()
    print(term[["position_id", "rebalance_hours", "net_position_pnl_usd", "residual_il_usd"]].to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
