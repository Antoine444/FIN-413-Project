# FIN-413-Project

## Setup

**Requirements: anaconda + Linux environment**
Those without a Linux environment may want to adapt the 'setup.sh' script. 

1. Clone the repository and navigate into it
2. Run `bash setup.sh`
3. Copy your RPC URL into a `.env` file: e.g. `RPC_URL=https://mainnet.infura.io/v3/YOUR_KEY`
4. Run `python data_extraction.py` (Module 1 — extracts on-chain data)
5. Run `python liquidity_analysis.py` (Module 2 — liquidity distribution figures)
6. Run `python slippage_analysis.py` (Module 3 — slippage and execution-cost figures)
7. Run `python lp_analytics.py` (Module 4 — LP fee income, IL, net P&L figures)

# Data Dictionary

All data files are stored in Parquet format. Timestamps are in UTC. Raw amounts are in the smallest token unit (wei for WETH, 1e-6 USDC for USDC). Decimal-adjusted amounts are human-readable.
Note that some values were stored in String format rather than their natural numerical types as a result of the pyarrow backend not being able to handle numerical types over 64 bits. 

---

## 1. `swap_events.parquet`

One row per Swap event emitted by the pool during the study window (1 October 2025 – 31 March 2026).

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `block_number` | int64 | block | Ethereum block number in which the swap was included |
| `block_timestamp` | datetime64[UTC] | UTC datetime | Timestamp of the block, fetched once per unique block |
| `transaction_hash` | string | — | Hex-encoded transaction hash (0x-prefixed) |
| `amount0_raw` | string | raw USDC (1e-6) | Signed USDC amount in raw token units; positive = USDC deposited into pool, negative = withdrawn |
| `amount0_decimal` | float64 | USDC | Human-readable signed USDC amount (amount0_raw / 1e6) |
| `amount1_raw` | string | raw WETH (1e-18) | Signed WETH amount in raw token units; positive = WETH deposited into pool, negative = withdrawn |
| `amount1_decimal` | float64 | WETH | Human-readable signed WETH amount (amount1_raw / 1e18) |
| `sqrtPriceX96` | string | raw | Square root of the pool price after the swap, encoded as sqrt(price) × 2^96; stored as string to avoid uint160 overflow |
| `price_usdc_per_weth` | float64 | USDC/WETH | Human-readable price derived from sqrtPriceX96, decimal-adjusted for token decimals |
| `liquidity` | string | liquidity units | Active liquidity in the pool at the time of the swap (uint128); stored as string to avoid overflow |
| `tick` | int32 | tick index | Active tick after the swap; corresponds to a price range of width 0.01% |
| `trade_direction` | string | — | Direction from the taker's perspective: `buy_weth` (USDC in, WETH out) or `sell_weth` (WETH in, USDC out) |
| `notional_usd` | float64 | USD | Notional trade size in USD, computed as the absolute value of amount0_decimal (USDC ≈ $1) |

**Ordering convention.** Rows are written in the order returned by `eth_getLogs`, which sorts by `(block_number, transactionIndex, logIndex)` ascending. Pandas preserves this through to Parquet, so the row order IS the on-chain log order. Downstream code in `slippage_analysis.py` derives an `intra_block_idx` (= within-block rank) from this row order; explicit `log_index` / `transaction_index` columns are not stored. If row order is ever shuffled (e.g. by a custom sort) the within-block sequence is lost, so future iterations may want to materialise these columns directly during extraction.

---

## 2. `mint_burn_events.parquet`

One row per Mint or Burn event emitted by the pool from deployment (block 12,376,729) through the end of the study window. The full history from deployment is required to reconstruct the liquidity map at any point in the study window.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `block_number` | int64 | block | Ethereum block number in which the event was included |
| `block_timestamp` | datetime64[UTC] | UTC datetime | Timestamp of the block, fetched once per unique block |
| `transaction_hash` | string | — | Hex-encoded transaction hash (0x-prefixed) |
| `event_type` | string | — | Either `mint` (liquidity added) or `burn` (liquidity removed) |
| `owner` | string | — | Ethereum address of the LP wallet that owns the position |
| `tick_lower` | int32 | tick index | Lower tick boundary of the LP position |
| `tick_upper` | int32 | tick index | Upper tick boundary of the LP position |
| `liquidity_raw` | string | liquidity units | Amount of liquidity added or removed (uint128); stored as string to avoid overflow |
| `amount0_raw` | string | raw USDC (1e-6) | USDC amount corresponding to the liquidity change, in raw token units |
| `amount0_decimal` | float64 | USDC | Human-readable USDC amount (amount0_raw / 1e6) |
| `amount1_raw` | string | raw WETH (1e-18) | WETH amount corresponding to the liquidity change, in raw token units |
| `amount1_decimal` | float64 | WETH | Human-readable WETH amount (amount1_raw / 1e18) |

---

## 3. `liquidity_snapshots.parquet`

One row per initialised tick per daily snapshot. Snapshots are taken at the block closest to 00:00 UTC each day of the study window. The liquidity map is reconstructed by replaying all Mint and Burn events up to each snapshot block.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `snapshot_block` | int64 | block | Block number of the daily snapshot (closest block to 00:00 UTC) |
| `snapshot_timestamp` | datetime64[UTC] | UTC datetime | Timestamp of the snapshot block |
| `tick` | int32 | tick index | Tick index of an initialised tick boundary (only ticks with liquidityGross > 0 are included) |
| `liquidity_net` | string | liquidity units | Net liquidity change at this tick: positive at a position's lower boundary, negative at its upper boundary. Used to compute active liquidity by walking from MIN_TICK upward (int64); stored as string to avoid overflow |
| `liquidity_gross` | string | liquidity units | Total absolute liquidity referencing this tick across all positions. A tick is initialised (active) when this value is greater than zero (int64); stored as string to avoid overflow |
| `active_liquidity` | string | liquidity units | Accumulated active liquidity in the range starting at this tick, computed by summing liquidityNet from MIN_TICK up to and including this tick (int64); stored as string to avoid overflow |
| `price_lower` | float64 | USDC/WETH | Human-readable price at the lower edge of this tick, computed as 1.0001^tick adjusted for token decimals |
| `price_upper` | float64 | USDC/WETH | Human-readable price at the upper edge of this tick, computed as 1.0001^(tick + 10) adjusted for token decimals |

---

## 4. `slot0_snapshots.parquet`

One row per daily snapshot. The pool's price and tick state obtained via a direct `eth_call` to the `slot0()` function at each snapshot block. Requires an archive node.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `snapshot_block` | int64 | block | Block number at which slot0() was called (closest block to 00:00 UTC each day) |
| `snapshot_timestamp` | datetime64[UTC] | UTC datetime | Timestamp of the snapshot block |
| `sqrtPriceX96` | string | raw | Square root of the pool price encoded as sqrt(price) × 2^96; stored as string to avoid uint160 overflow |
| `price_usdc_per_weth` | float64 | USDC/WETH | Human-readable price derived from sqrtPriceX96, decimal-adjusted for USDC (6 decimals) and WETH (18 decimals) |
| `current_tick` | int32 | tick index | The active tick at the snapshot block, i.e. the tick whose range contains the current price |
| `observation_index` | int32 | — | Index of the most recent TWAP observation in the pool's oracle array |
| `unlocked` | bool | — | Reentrancy lock flag; True means the pool is not mid-execution. Should always be True at snapshot blocks |

---

## 5. `collect_events.parquet`

One row per Collect event from pool deployment through the end of the study window. Collected during the same extraction pass as mint/burn events. Required for Module 4 (Liquidity Provision Analytics).

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `block_number` | int64 | block | Ethereum block number in which the event was included |
| `transaction_hash` | string | — | Hex-encoded transaction hash (0x-prefixed) |
| `owner` | string | — | Ethereum address of the LP wallet that owns the position |
| `recipient` | string | — | Ethereum address that received the collected fees |
| `tick_lower` | int32 | tick index | Lower tick boundary of the position whose fees were collected |
| `tick_upper` | int32 | tick index | Upper tick boundary of the position whose fees were collected |
| `amount0_raw` | string | raw USDC (1e-6) | USDC fees collected in raw token units |
| `amount1_raw` | string | raw WETH (1e-18) | WETH fees collected in raw token units |
| `amount0_decimal` | float64 | USDC | Human-readable USDC fees collected (amount0_raw / 1e6) |
| `amount1_decimal` | float64 | WETH | Human-readable WETH fees collected (amount1_raw / 1e18) |

---

## 6. `lp_positions.parquet`

One row per synthetic LP position built by Module 4 (`lp_analytics.py`). Five positions are constructed at the first daily snapshot block (entry) with a $100,000 USD notional each, then held to the last daily snapshot block (exit). Range widths are: ±0.1% (P1), ±0.5% (P2), ±2% (P3), ±10% (P4), full range (P5).

Tick boundaries are rounded OUTWARD to multiples of `TICK_SPACING = 10` so the realised tick band contains the entry tick. Because USDC = token0, a higher USDC/WETH price corresponds to a lower V3 tick, so `tick_lower` is associated with `price_upper` and vice versa.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `position_id` | string | — | Position label: `P1`, `P2`, `P3`, `P4`, `P5` |
| `range_pct` | float64 | fraction | Half-width of the requested price band (e.g. 0.001 for ±0.1%); `NaN` for P5 (full range) |
| `tick_lower` | int64 | tick index | Lower V3 tick boundary (corresponds to upper USDC/WETH price) |
| `tick_upper` | int64 | tick index | Upper V3 tick boundary (corresponds to lower USDC/WETH price) |
| `price_lower_usdc_per_weth` | float64 | USDC/WETH | Lower edge of the realised USDC/WETH price band, `1e12 / 1.0001^tick_upper` |
| `price_upper_usdc_per_weth` | float64 | USDC/WETH | Upper edge of the realised USDC/WETH price band, `1e12 / 1.0001^tick_lower` |
| `L` | string | liquidity units | V3 liquidity computed so that `V_LP(entry) = $100,000` (uint128; stored as string to avoid overflow) |
| `x0_weth` | float64 | WETH | Initial WETH deposit at entry, computed from `L` via the V3 virtual reserve formulas |
| `y0_usdc` | float64 | USDC | Initial USDC deposit at entry, computed from `L` via the V3 virtual reserve formulas |
| `entry_value_usd` | float64 | USD | `x0_weth * p_entry + y0_usdc`; should equal $100,000 to within float tolerance |
| `entry_block` | int64 | block | Block of the first daily snapshot (entry block) |
| `entry_timestamp` | datetime64[UTC] | UTC datetime | Timestamp of `entry_block` |
| `exit_block` | int64 | block | Block of the last daily snapshot (exit block) |
| `exit_timestamp` | datetime64[UTC] | UTC datetime | Timestamp of `exit_block` |

---

## 7. `lp_timeseries.parquet`

One row per `(position, snapshot)` pair — five positions × 182 daily snapshots = 910 rows. Provides V_LP, V_HODL, IL, cumulative fee income (in USD, native USDC, and native WETH), and net P&L for each synthetic LP position at every daily snapshot in the study window.

Fee income is computed from `swap_events.parquet` (not `collect_events.parquet`, which would represent real LP withdrawals). For each swap with `tick_lower <= swap.tick < tick_upper` and `entry_block < block_number <= exit_block`, the position's LP-side fee is `swap_amount * 0.0005 * L_position / swap.liquidity`, in the deposit-side token (USDC fee on USDC-in swaps; WETH fee on WETH-in swaps). The per-swap fees are bucketed by their *next* daily snapshot block and cumulated. We persist three cumulative columns: `cumulative_fee_usdc` and `cumulative_fee_weth` in their native tokens (no conversion ever applied), and `cumulative_fee_usd` with WETH fees converted at each swap's own price. The native columns let the reader re-aggregate under any valuation rule (e.g. value WETH at exit price instead of swap-time price). The PDF Task 4.2 spec phrase "ETH price prevailing at the time of collection" is read as "time of accrual" (= swap time) here; the alternative read (value at actual `collect()` time, i.e. exit for a passive LP) gives ~$2–3K smaller terminal USD fees per narrow position because ETH fell over the study window — see the `Fee re-aggregation` block in the script output.

Impermanent loss uses the Uniswap V3 virtual reserve formulas (with the three cases for current price below / above / inside the position range) to evaluate `V_LP(t)` at every daily snapshot, then `IL(t) = V_HODL(t) - V_LP(t)`. A positive IL means the LP underperforms the HODL benchmark. The position-sizing step uses the exact slot0 `sqrtPriceX96 / 2^96` at the entry block (not the tick-implied `1.0001^(tick/2)` approximation), so `V_LP(entry) = V_HODL(entry) = $100,000` to float precision.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `position_id` | string | — | Position label (`P1`…`P5`); matches `lp_positions.parquet` |
| `snapshot_block` | int64 | block | Block of the daily snapshot (from `slot0_snapshots.parquet`) |
| `snapshot_timestamp` | datetime64[UTC] | UTC datetime | Timestamp of the snapshot block |
| `price_usdc_per_weth` | float64 | USDC/WETH | Pool mid price at the snapshot block, copied from `slot0_snapshots` |
| `v_lp_usd` | float64 | USD | LP principal value at the snapshot: `amount_weth(t) * p_t + amount_usdc(t)` |
| `v_hodl_usd` | float64 | USD | HODL benchmark value at the snapshot: `x0_weth * p_t + y0_usdc` |
| `impermanent_loss_usd` | float64 | USD | `v_hodl_usd − v_lp_usd`; positive = LP underperforms HODL |
| `cumulative_fee_usd` | float64 | USD | Cumulative LP fee income through this snapshot, with WETH fees converted to USD at *each swap's own price* (= ETH price prevailing at accrual time). This is the column used to compute `net_pnl_usd`. |
| `cumulative_fee_usdc` | float64 | USDC | Cumulative LP fee income through this snapshot, USDC leg only (in native USDC tokens; no conversion applied) |
| `cumulative_fee_weth` | float64 | WETH | Cumulative LP fee income through this snapshot, WETH leg only (in native WETH tokens; no conversion applied) |
| `net_pnl_usd` | float64 | USD | `cumulative_fee_usd − impermanent_loss_usd`; matches the figure 4.3 quantity |
