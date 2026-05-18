"""
hyperliquid_fetch.py
====================
Module 5, Task 5.2 — Hyperliquid ETH perpetual data for the study window.

Fetches:
  output/perp_prices.parquet     hourly OHLCV (close = mark price)
  output/funding_rates.parquet   hourly funding rates
  figures/fig5_2_funding.png     funding environment chart

API: POST https://api.hyperliquid.xyz/info
  - candleSnapshot  (paginated, max 500 candles per request)
  - fundingHistory  (paginated; body uses top-level coin/startTime/endTime)

Note: Hyperliquid candle history can be sparse for very early dates in the window.
      Missing hours are filled from Uniswap slot0 daily snapshots (linear bridge),
      flagged in the `price_source` column. Funding history is available for the
      full window from the API.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "output"
FIG_DIR = "figures"
API_URL = "https://api.hyperliquid.xyz/info"
COIN = "ETH"
INTERVAL = "1h"
CHUNK_MS = 500 * 3600 * 1000  # 500 hourly candles per API page

STUDY_START = pd.Timestamp("2025-10-01", tz="UTC")
STUDY_END = pd.Timestamp("2026-04-01", tz="UTC")  # exclusive upper bound

PERP_OUT = os.path.join(DATA_DIR, "perp_prices.parquet")
FUNDING_OUT = os.path.join(DATA_DIR, "funding_rates.parquet")
SLOT0_PATH = os.path.join(DATA_DIR, "slot0_snapshots.parquet")
FIG_FUNDING = os.path.join(FIG_DIR, "fig5_2_funding.png")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def _post(payload: dict, retries: int = 4) -> object:
    for attempt in range(retries):
        r = requests.post(API_URL, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(1.5 * (attempt + 1))
    r.raise_for_status()
    return None


def fetch_candles(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Paginated candleSnapshot; returns raw API rows."""
    rows = []
    cursor = _ms(start)
    end_ms = _ms(end)
    while cursor < end_ms:
        chunk_end = min(cursor + CHUNK_MS, end_ms)
        data = _post({
            "type": "candleSnapshot",
            "req": {
                "coin": COIN,
                "interval": INTERVAL,
                "startTime": cursor,
                "endTime": chunk_end,
            },
        })
        if not data:
            cursor = chunk_end
            continue
        rows.extend(data)
        last_t = max(int(c["T"]) for c in data)
        cursor = last_t + 1
        if len(data) < 500:
            cursor = chunk_end
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    for col in ("o", "h", "l", "c", "v"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["price_source"] = "hyperliquid"
    return df[["timestamp", "open", "high", "low", "close", "volume", "price_source"]].drop_duplicates(
        subset=["timestamp"]
    )


def fetch_funding(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Paginated fundingHistory."""
    rows = []
    cursor = _ms(start)
    end_ms = _ms(end)
    while cursor < end_ms:
        chunk_end = min(cursor + 30 * 24 * 3600 * 1000, end_ms)
        data = _post({
            "type": "fundingHistory",
            "coin": COIN,
            "startTime": cursor,
            "endTime": chunk_end,
        })
        if not data:
            cursor = chunk_end
            continue
        rows.extend(data)
        last_t = max(int(r["time"]) for r in data)
        cursor = last_t + 1
        if len(data) < 500:
            cursor = chunk_end
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.floor("h")
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["premium"] = pd.to_numeric(df["premium"], errors="coerce")
    return df[["timestamp", "funding_rate", "premium", "coin"]].drop_duplicates(subset=["timestamp"])


def slot0_hourly_bridge() -> pd.DataFrame:
    """Interpolate daily Uniswap slot0 closes to hourly (for HL candle gaps)."""
    slot0 = pd.read_parquet(SLOT0_PATH)
    slot0["snapshot_timestamp"] = pd.to_datetime(slot0["snapshot_timestamp"], utc=True)
    daily = slot0.set_index("snapshot_timestamp")["price_usdc_per_weth"].sort_index()
    hourly_idx = pd.date_range(STUDY_START, STUDY_END, freq="1h", inclusive="left")
    hourly = daily.reindex(hourly_idx.union(daily.index)).sort_index().interpolate(method="time")
    hourly = hourly.reindex(hourly_idx)
    out = pd.DataFrame({
        "timestamp": hourly.index,
        "open": hourly.values,
        "high": hourly.values,
        "low": hourly.values,
        "close": hourly.values,
        "volume": np.nan,
        "price_source": "uniswap_slot0_bridge",
    })
    return out.reset_index(drop=True)


def build_hourly_prices(candles: pd.DataFrame) -> pd.DataFrame:
    """Full hourly grid: Hyperliquid where available, slot0 bridge elsewhere."""
    base = slot0_hourly_bridge().set_index("timestamp")
    if not candles.empty:
        hl = candles.set_index("timestamp")
        hl = hl[(hl.index >= STUDY_START) & (hl.index < STUDY_END)]
        base.update(hl)
        base.loc[hl.index.intersection(base.index), "price_source"] = "hyperliquid"
    merged = base.reset_index()
    merged = merged[(merged["timestamp"] >= STUDY_START) & (merged["timestamp"] < STUDY_END)]
    return merged.sort_values("timestamp").reset_index(drop=True)


def build_hourly_funding(funding: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Align funding to the hourly grid; attach oracle_price = candle close."""
    hourly_idx = pd.date_range(STUDY_START, STUDY_END, freq="1h", inclusive="left")
    f = funding.set_index("timestamp").sort_index()
    f = f.reindex(hourly_idx).ffill().bfill()
    out = f.reset_index().rename(columns={"index": "timestamp"})
    out = out.merge(
        prices[["timestamp", "close"]].rename(columns={"close": "oracle_price"}),
        on="timestamp",
        how="left",
    )
    return out


def plot_funding(funding: pd.DataFrame, prices: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax0 = axes[0]
    ax0.plot(prices["timestamp"], prices["close"], color="#1f77b4", lw=1.2)
    ax0.set_ylabel("ETH price (USD, Hyperliquid close\nor slot0 bridge)")
    ax0.set_title("Hyperliquid ETH Perpetual — Price and Funding (Study Window)")
    ax0.grid(True, ls=":", alpha=0.4)

    ax1 = axes[1]
    ax1.bar(
        funding["timestamp"],
        funding["funding_rate"] * 10_000,
        width=0.03,
        color=np.where(funding["funding_rate"] >= 0, "#2ca02c", "#d62728"),
        alpha=0.7,
        label="Hourly funding rate (bps)",
    )
    cum = (funding["funding_rate"]).cumsum()
    ax1b = ax1.twinx()
    ax1b.plot(funding["timestamp"], cum, color="#9467bd", lw=1.5, label="Cumulative funding (sum of hourly rates)")
    ax1.set_ylabel("Hourly funding rate (bps)")
    ax1b.set_ylabel("Cumulative sum of hourly rates")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.grid(True, ls=":", alpha=0.4)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper left", fontsize=8)
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  {out_path}")


def main() -> None:
    print("Fetching Hyperliquid ETH perp candles (1h) ...")
    candles = fetch_candles(STUDY_START, STUDY_END)
    print(f"  Raw API candles: {len(candles):,}")
    prices = build_hourly_prices(candles)
    n_hl = (prices["price_source"] == "hyperliquid").sum()
    n_br = (prices["price_source"] == "uniswap_slot0_bridge").sum()
    print(f"  Hourly grid: {len(prices):,} rows  (HL={n_hl:,}, slot0 bridge={n_br:,})")
    prices.to_parquet(PERP_OUT, index=False)
    print(f"  Wrote {PERP_OUT}")

    print("Fetching Hyperliquid funding history ...")
    funding_raw = fetch_funding(STUDY_START, STUDY_END)
    print(f"  Raw funding records: {len(funding_raw):,}")
    funding = build_hourly_funding(funding_raw, prices)
    funding.to_parquet(FUNDING_OUT, index=False)
    print(f"  Wrote {FUNDING_OUT}")

    print("Writing funding environment figure ...")
    plot_funding(funding, prices, FIG_FUNDING)
    print("Done.")


if __name__ == "__main__":
    main()
