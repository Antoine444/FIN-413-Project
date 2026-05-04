# FIN-413-Project

## Setup

**Requirements: anaconda + Linux environment**
Those without a Linux environment may want to adapt the 'setup.sh' script. 

1. Clone the repository and navigate into it
2. Run `bash setup.sh`
3. Copy your RPC URL into a `.env` file: e.g. `RPC_URL=https://mainnet.infura.io/v3/YOUR_KEY`
4. Run `python data_extraction.py`

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
| `liquidity_net` | int64 | liquidity units | Net liquidity change at this tick: positive at a position's lower boundary, negative at its upper boundary. Used to compute active liquidity by walking from MIN_TICK upward |
| `liquidity_gross` | int64 | liquidity units | Total absolute liquidity referencing this tick across all positions. A tick is initialised (active) when this value is greater than zero |
| `active_liquidity` | int64 | liquidity units | Accumulated active liquidity in the range starting at this tick, computed by summing liquidityNet from MIN_TICK up to and including this tick |
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
