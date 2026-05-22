"""
hyperliquid_fetch.py
====================
Module 5, Task 5.2 — Hyperliquid ETH perpetual data for the study window.

Fetches:
  output/perp_prices.parquet     hourly OHLCV (close = mark price)
  output/funding_rates.parquet   hourly funding rates
  figures/fig5_2_funding.png     funding environment chart

Source: Hyperliquid (single venue, per project brief)
  POST https://api.hyperliquid.xyz/info
    - candleSnapshot  (paginated, max 500 candles per request)
    - fundingHistory  (paginated; body uses top-level coin/startTime/endTime)

Data availability caveat (price candles only).
  Hyperliquid's public `candleSnapshot` endpoint retains 1h ETH candles for
  approximately seven months. Hours older than that are not returned by the
  API. For a six-month study window that starts more than ~6 months before
  the fetch date (e.g. study start 2025-10-01, fetch date 2026-05-22), the
  earliest ~3–4 weeks of the window are not retrievable at 1h resolution.

  Lower-resolution candles (4h, 1d) and the `fundingHistory` endpoint retain
  longer and cover the full window. Per Prof. Karyampas's directive (email,
  May 2026) we keep Hyperliquid as the sole data source and fill the
  uncovered hours by forward-filling Hyperliquid 4h candles onto the
  hourly grid: for each hour H without a native 1h candle, the row inherits
  OHLCV from the most recent 4h candle whose timestamp ≤ H. Each row's
  `price_source` column records `hyperliquid_1h` (native) or
  `hyperliquid_4h_upsampled` (4h-derived) so the limitation is transparent
  in the parquet, not only in the report.

  Implications of the upsample:
    - For upsampled hours, OHLCV all carry the 4h bar's aggregate values;
      4 consecutive hourly rows therefore share identical OHLCV. Only `close`
      is used downstream (hedge backtest mark price); volume/n_trades are
      kept for completeness but should not be summed across upsampled rows
      without de-duplicating.
    - During the upsampled window, 1h-rebalancing hedge variants produce
      hedge P&L only on 4h boundaries (since intermediate hourly prices are
      flat), which correctly reflects the available information.

  If neither 1h nor 4h Hyperliquid data covers a given hour,
  `build_hourly_prices` raises `RuntimeError` — there is no other source.

Oracle-price proxy.
  The PDF funding formula uses `oracle_price`. Hyperliquid's public API
  exposes mark candles and funding records, but no historical oracle series.
  We therefore store the hourly candle close as `oracle_price_proxy` in
  funding_rates.parquet. Per Hyperliquid's docs, the `premium` field is
  (impact_price - oracle)/oracle, where impact_price is order-book impact
  at a fixed notional, NOT the candle close. Mark and impact share order
  book drivers, so |premium| (mean ~4 bps, p99 ~7 bps, max ~36 bps over
  the study window) is a reasonable empirical sensitivity proxy for the
  mark-oracle gap, not a strict bound. Module 5 §5.2 of the report works
  through the induced funding-P&L error (~0.36% of cumulative funding worst
  case, a few dollars per position).
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

# Interval-aware page sizing: candleSnapshot returns at most 500 rows per call.
_INTERVAL_SECONDS = {"1h": 3600, "4h": 4 * 3600}

STUDY_START = pd.Timestamp("2025-10-01", tz="UTC")
STUDY_END = pd.Timestamp("2026-04-01", tz="UTC")  # exclusive upper bound

PERP_OUT = os.path.join(DATA_DIR, "perp_prices.parquet")
FUNDING_OUT = os.path.join(DATA_DIR, "funding_rates.parquet")
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


def fetch_candles(start: pd.Timestamp, end: pd.Timestamp, interval: str = "1h") -> pd.DataFrame:
    """Paginated candleSnapshot for a given interval; returns raw API rows.

    Supports `interval` in {"1h", "4h"}. The page size is fixed at 500 by the
    API, so the chunk width in milliseconds scales with the interval to keep
    one request per page.
    """
    if interval not in _INTERVAL_SECONDS:
        raise ValueError(f"unsupported interval {interval!r}")
    chunk_ms = 500 * _INTERVAL_SECONDS[interval] * 1000
    rows = []
    cursor = _ms(start)
    end_ms = _ms(end)
    while cursor < end_ms:
        chunk_end = min(cursor + chunk_ms, end_ms)
        data = _post({
            "type": "candleSnapshot",
            "req": {
                "coin": COIN,
                "interval": interval,
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
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "n": "n_trades"})
    df["n_trades"] = pd.to_numeric(df["n_trades"], errors="coerce")
    return df[["timestamp", "open", "high", "low", "close", "volume", "n_trades"]].drop_duplicates(
        subset=["timestamp"]
    ).sort_values("timestamp").reset_index(drop=True)


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


def build_hourly_prices(hl_1h: pd.DataFrame, hl_4h: pd.DataFrame) -> pd.DataFrame:
    """Full hourly grid backed by Hyperliquid 1h candles, with 4h upsample fallback.

    Layering order:
      1. Hyperliquid 4h candles, forward-filled onto the hourly grid
         (each 4h bar populates the four hours it covers).
      2. Hyperliquid 1h native candles, overwriting any 4h-derived rows.

    Hard-fails (RuntimeError) if any hour still lacks coverage — should not
    happen because 4h retention exceeds the six-month study window, so the
    raise is a guard against future API changes.
    """
    data_cols = ["open", "high", "low", "close", "volume", "n_trades"]
    cols = ["timestamp", *data_cols, "price_source"]
    hourly_idx = pd.date_range(STUDY_START, STUDY_END, freq="1h", inclusive="left")
    base = pd.DataFrame({c: np.nan for c in data_cols}, index=hourly_idx)
    base["price_source"] = None
    base.index.name = "timestamp"

    # Layer 4h candles via forward-fill onto the hourly grid.
    if not hl_4h.empty:
        h4 = hl_4h.set_index("timestamp").sort_index()
        h4 = h4[(h4.index >= STUDY_START) & (h4.index < STUDY_END)]
        if not h4.empty:
            h4_upsampled = h4[data_cols].reindex(hourly_idx, method="ffill")
            base.update(h4_upsampled)
            covered_by_4h = base.index[base["close"].notna()]
            base.loc[covered_by_4h, "price_source"] = "hyperliquid_4h_upsampled"

    # Layer native 1h candles on top (overwrites 4h-derived rows).
    if not hl_1h.empty:
        h1 = hl_1h.set_index("timestamp").sort_index()
        h1 = h1[(h1.index >= STUDY_START) & (h1.index < STUDY_END)]
        if not h1.empty:
            base.update(h1[data_cols])
            base.loc[h1.index.intersection(base.index), "price_source"] = "hyperliquid_1h"

    # Hard-fail on any uncovered hour.
    missing = base.index[base["close"].isna()]
    if len(missing) > 0:
        raise RuntimeError(
            f"Price coverage gap: {len(missing)} hour(s) have neither Hyperliquid 1h "
            f"nor 4h coverage. First 5: {missing[:5].tolist()}"
        )

    merged = base.reset_index().rename(columns={"index": "timestamp"})
    return merged.sort_values("timestamp").reset_index(drop=True)[cols]


def build_hourly_funding(funding: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Align funding to the hourly grid; attach oracle_price_proxy = candle close.

    Hard-fails if any hour lacks a funding record. The column is named
    `oracle_price_proxy` (not `oracle_price`) because the candle close is a
    proxy for Hyperliquid's true oracle price, which the public API does not
    expose historically. See module docstring.
    """
    hourly_idx = pd.date_range(STUDY_START, STUDY_END, freq="1h", inclusive="left")
    f = funding.set_index("timestamp").sort_index().reindex(hourly_idx)
    missing = f.index[f["funding_rate"].isna()]
    if len(missing) > 0:
        raise RuntimeError(
            f"Funding coverage gap: {len(missing)} hour(s) without a Hyperliquid "
            f"funding record. First 5: {missing[:5].tolist()}"
        )
    out = f.reset_index().rename(columns={"index": "timestamp"})
    out = out.merge(
        prices[["timestamp", "close"]].rename(columns={"close": "oracle_price_proxy"}),
        on="timestamp",
        how="left",
    )
    return out


def plot_funding(funding: pd.DataFrame, prices: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax0 = axes[0]
    ax0.plot(prices["timestamp"], prices["close"], color="#1f77b4", lw=1.2)
    ax0.set_ylabel("ETH price (USD, Hyperliquid mark;\n4h-upsampled where 1h unavailable)")
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
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="lower right", fontsize=8)
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  {out_path}")


def main() -> None:
    print("Fetching Hyperliquid ETH perp candles (1h) ...")
    hl_1h = fetch_candles(STUDY_START, STUDY_END, interval="1h")
    print(f"  Hyperliquid 1h candles returned: {len(hl_1h):,}")

    print("Fetching Hyperliquid ETH perp candles (4h) for upsample fallback ...")
    hl_4h = fetch_candles(STUDY_START, STUDY_END, interval="4h")
    print(f"  Hyperliquid 4h candles returned: {len(hl_4h):,}")

    prices = build_hourly_prices(hl_1h, hl_4h)
    counts = prices["price_source"].value_counts().to_dict()
    print(f"  Hourly grid: {len(prices):,} rows  by source: {counts}")
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
