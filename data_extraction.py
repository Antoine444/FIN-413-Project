"""
data_extraction.py
==================
Extracts all on-chain data required for the Uniswap V3 USDC/WETH (0.05%) pool analysis.

Produces four Parquet files:
    - swap_events.parquet
    - mint_burn_events.parquet
    - liquidity_snapshots.parquet
    - slot0_snapshots.parquet

Usage:
    conda activate uniswap_project
    python data_extraction.py

Requirements:
    - A .env file in the same directory containing: RPC_URL=https://...
    - All dependencies installed (see setup.sh)
"""

import os
import json
import time
import math
from datetime import datetime, timezone
from collections import defaultdict

import requests
import pickle
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from web3 import Web3
from dotenv import load_dotenv
from tqdm import tqdm

# =============================================================================
# CONSTANTS — Edit only this section if parameters change
# =============================================================================

# Pool identity
POOL_ADDRESS        = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"
POOL_DEPLOY_BLOCK   = 12_376_729          # Block at which the pool was deployed

# Study window — blocks corresponding to:
#   Start: 1 October 2025 00:00 UTC
#   End:   31 March 2026  23:59 UTC
# These were found using the find_blocks.py script.
STUDY_START_BLOCK   = 23_479_244          # ≈ 1 Oct 2025
STUDY_END_BLOCK     = 24_773_861          # ≈ 31 Mar 2026

# Token decimals
USDC_DECIMALS       = 6
WETH_DECIMALS       = 18

# Uniswap V3 pool parameters
TICK_SPACING        = 10
FEE_TIER            = 0.0005              # 0.05%

# Tick bounds (Uniswap V3 hard limits)
MIN_TICK            = -887_272
MAX_TICK            =  887_272

# RPC batching — number of blocks per eth_getLogs request.
# Free-tier providers typically reject ranges wider than 2000 blocks.
# Infura accepts up to 10_000
BLOCK_CHUNK_SIZE    = 1_500

# Retry settings for RPC calls
MAX_RETRIES         = 7
RETRY_BACKOFF       = 2.0                 # seconds; doubles on each failure

# Output file paths
OUT_SWAP            = "output/swap_events.parquet"
OUT_MINT_BURN       = "output/mint_burn_events.parquet"
OUT_LIQ_SNAP        = "output/liquidity_snapshots.parquet"
OUT_SLOT0_SNAP      = "output/slot0_snapshots.parquet"

# =============================================================================
# ABI — Minimal ABI covering only the events and functions we need.
# Full ABI available at:
# https://etherscan.io/address/0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640#code
# =============================================================================

# Load pool ABI from file (source: Etherscan contract page)
with open("pool_ABI.json", "r") as f:
    POOL_ABI = json.load(f)
    
# =============================================================================
# INITIALISATION
# =============================================================================

# Load RPC URL from .env file
load_dotenv()
RPC_URL = os.getenv("RPC_URL")
if not RPC_URL:
    raise EnvironmentError("RPC_URL not found in .env file. Please add it.")

# Connect to the Ethereum node via Web3
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise ConnectionError(f"Cannot connect to Ethereum node at {RPC_URL}")
print(f"Connected to Ethereum node. Latest block: {w3.eth.block_number:,}")

# Instantiate the pool contract object (used for eth_call)
pool = w3.eth.contract(
    address=Web3.to_checksum_address(POOL_ADDRESS),
    abi=POOL_ABI
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sqrtPriceX96_to_price(sqrt_price_x96: int) -> float:
    """
    Convert Uniswap V3's sqrtPriceX96 to a human-readable USDC per WETH price.

    sqrtPriceX96 is stored as sqrt(price) * 2^96 where price = token1/token0
    in raw (un-adjusted) units. For this pool:
        token0 = USDC (6 decimals)
        token1 = WETH (18 decimals)

    Formula:
        price_raw       = (sqrtPriceX96 / 2^96) ^ 2
        price_adjusted  = price_raw * 10^(USDC_DECIMALS - WETH_DECIMALS)
                        = price_raw * 10^(6 - 18)
                        = price_raw / 10^12

    This gives WETH/USDC. We invert to get USDC/WETH.
    """
    price_raw = (sqrt_price_x96 / (2 ** 96)) ** 2
    # Adjust for token decimals: USDC has 6, WETH has 18
    price_token1_per_token0 = price_raw * (10 ** (USDC_DECIMALS - WETH_DECIMALS))
    # price_token1_per_token0 is WETH per USDC; invert for USDC per WETH
    if price_token1_per_token0 == 0:
        return float("nan")
    return 1.0 / price_token1_per_token0
    
def get_event_topic(event_name: str) -> str:
    """
    Compute the keccak256 topic hash for a given event name directly
    from the ABI, without relying on Web3.py internal methods.
    
    This is the standard way to identify events in raw logs:
    topic0 = keccak256("EventName(type1,type2,...)")
    """
    # Find the event definition in the ABI
    event_abi = next(e for e in POOL_ABI if e.get("name") == event_name and e.get("type") == "event")
    
    # Build the canonical signature string e.g. "Swap(address,address,int256,...)"
    input_types = ",".join(i["type"] for i in event_abi["inputs"])
    signature   = f"{event_name}({input_types})"
    
    # Ensure 0x prefix — Alchemy requires it
    return "0x" + w3.keccak(text=signature).hex()


def tick_to_price(tick: int) -> float:
    """
    Convert a Uniswap V3 tick index to a human-readable USDC per WETH price.

    Price at tick i (in raw token units) = 1.0001^i
    After decimal adjustment: multiply by 10^(USDC_DECIMALS - WETH_DECIMALS) = 10^-12
    Then invert (same logic as sqrtPriceX96_to_price).
    """
    price_raw = 1.0001 ** tick
    price_token1_per_token0 = price_raw * (10 ** (USDC_DECIMALS - WETH_DECIMALS))
    if price_token1_per_token0 == 0:
        return float("nan")
    return 1.0 / price_token1_per_token0


def retry_call(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) with exponential backoff on failure.
    Retries up to MAX_RETRIES times before raising the last exception.
    """
    delay = RETRY_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            if "429" in str(e) or "Too Many Requests" in str(e):
                # Start at 60s and double each time — Alchemy's window is long
                wait = 60.0 * (2 ** attempt)
                print(f"  [rate limited] Waiting {wait:.0f}s before retry {attempt+1}/{MAX_RETRIES}...")
            else:
                wait = delay
                print(f"  [retry {attempt+1}/{MAX_RETRIES}] Error: {e}. Waiting {wait}s...")
            time.sleep(wait)
            delay *= 2 # Exponential backoff


def fetch_block_timestamps(block_numbers: list) -> dict:
    """
    Fetch timestamps for all unique blocks using JSON-RPC batching.
    Results are cached to disk so repeated runs don't re-fetch.
    Delete cache_timestamps.pkl to force a fresh fetch.
    
    Auto-saves progress every 500 blocks to prevent data loss.
    """	
    BATCH_SIZE  = 20  # Each call costs 20 CUs
    SLEEP_PER_BATCH = 1.0  # 1.0 s — respects 500 CU/s rate limit
    CACHE_FILE  = "cache/cache_timestamps.pkl"
    SAVE_INTERVAL = 500  # Save to disk every 500 successfully fetched blocks

    # Load existing cache if available
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            timestamps = pickle.load(f)
        print(f"  Loaded timestamp cache ({len(timestamps):,} blocks already cached)")
    else:
        timestamps = {}

    unique_blocks = list(set(block_numbers) - set(timestamps.keys()))
    if not unique_blocks:
        print(f"  All {len(set(block_numbers)):,} timestamps served from cache")
        return timestamps

    print(f"  Fetching {len(unique_blocks):,} new block timestamps "
          f"({len(timestamps):,} already cached)...")

    # Track progress for auto-saving
    blocks_fetched_since_last_save = 0
    initial_cache_size = len(timestamps)

    def save_cache():
        """Helper function to save cache to disk"""
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(timestamps, f)

    def fetch_batch(blocks_to_fetch: list, retry: int = 3) -> list:
        nonlocal blocks_fetched_since_last_save
        
        payload = [
            {"jsonrpc": "2.0", "method": "eth_getBlockByNumber",
             "params": [hex(bn), False], "id": j}
            for j, bn in enumerate(blocks_to_fetch)
        ]
	
        for attempt in range(retry):
            response = requests.post(RPC_URL, json=payload, timeout=30)
            results  = response.json()
	
            # Batch-level rejection
            if isinstance(results, dict):
                error_code = results.get("error", {}).get("code", None)
                if error_code == -32005:
                    wait = 2 ** attempt * 2   # 2s, 4s, 8s
                    print(f"  Batch-level rate limit, waiting {wait}s (attempt {attempt+1}/{retry})…")
                    time.sleep(wait)
                    continue
                print(f"  Batch-level error: {results.get('error', results)}")
                return blocks_to_fetch
	
            # Check if *any* entry is a rate limit error (no id field)
            rate_limited = any(
                isinstance(r, dict) and "id" not in r and r.get("code") == -32005
                for r in results
            )
            if rate_limited:
                wait = 2 ** attempt * 2
                print(f"  Entry-level rate limit, waiting {wait}s (attempt {attempt+1}/{retry})…")
                time.sleep(wait)
                continue
	
            # Process valid results
            failed = []
            for result in results:
                if not isinstance(result, dict) or "id" not in result:
                    print(f"  Warning: unexpected entry: {result}")
                    continue
                if "result" in result and result["result"]:
                    bn = int(result["result"]["number"], 16)
                    ts = int(result["result"]["timestamp"], 16)
                    timestamps[bn] = datetime.fromtimestamp(ts, tz=timezone.utc)
                    
                    # Increment counter and save if threshold reached
                    blocks_fetched_since_last_save += 1
                    if blocks_fetched_since_last_save >= SAVE_INTERVAL:
                        save_cache()
                        print(f"    Auto-saved after {blocks_fetched_since_last_save} new blocks "
                              f"(total: {len(timestamps):,})")
                        blocks_fetched_since_last_save = 0
                else:
                    failed.append(blocks_to_fetch[result["id"]])
	
            time.sleep(SLEEP_PER_BATCH)
            return failed

        # All retries exhausted
        print(f"  Warning: batch failed after {retry} attempts, marking all as failed.")
        return blocks_to_fetch
    
    # ---- Pass 1: full batches ------------------------------------------------
    all_failed = []
    for i in tqdm(range(0, len(unique_blocks), BATCH_SIZE), desc="  Block timestamps"):
        batch  = unique_blocks[i : i + BATCH_SIZE]
        failed = fetch_batch(batch)
        if failed:
            all_failed.extend(failed)

    # Save after pass 1 to ensure progress is captured
    save_cache()
    print(f"  Saved after pass 1 ({len(timestamps):,} blocks total)")
    blocks_fetched_since_last_save = 0  # Reset counter

    # ---- Pass 2: retry failed blocks in smaller batches ---------------------
    if all_failed:
        RETRY_BATCH_SIZE = 50
        print(f"  Retrying {len(all_failed):,} null results in batches of {RETRY_BATCH_SIZE}...")
        still_failed = []
        for i in tqdm(range(0, len(all_failed), RETRY_BATCH_SIZE), desc="  Retry batches"):
            batch  = all_failed[i : i + RETRY_BATCH_SIZE]
            failed = fetch_batch(batch)
            if failed:
                still_failed.extend(failed)
        
        # Save after retry pass
        if still_failed:
            print(f"  Warning: {len(still_failed):,} blocks could not be fetched after retry.")
        save_cache()
        print(f"  Saved after retry pass ({len(timestamps):,} blocks total)")
    
    # ---- Final save and cleanup ----------------------------------------------
    save_cache()
    
    # Calculate stats
    newly_fetched = len(timestamps) - initial_cache_size
    print(f"  Timestamp cache updated → {CACHE_FILE}")
    print(f"    Total blocks: {len(timestamps):,}")
    print(f"    Newly fetched: {newly_fetched:,}")
    print(f"    Cache misses: {len(unique_blocks) - newly_fetched:,}")
    
    return timestamps    
    
def fetch_logs_range(event, event_topic: str, from_block: int, to_block: int) -> list:
    """
    Fetch logs for a single block range, recursively splitting in half
    if the provider returns a 'too many results' error.
    """
    try:
        raw_logs = w3.eth.get_logs({
            "address":   Web3.to_checksum_address(POOL_ADDRESS),
            "topics":    [event_topic],
            "fromBlock": hex(from_block),
            "toBlock":   hex(to_block),
        })
        return [event.process_log(log) for log in raw_logs]

    except Exception as e:
        # Detect too-many-results error and split the range in half
        if "-32005" in str(e) or "10000 results" in str(e):
            mid = (from_block + to_block) // 2
            if mid == from_block:
                raise ValueError(f"Single block {from_block} has >10000 logs — cannot split further")
            print(f"  [split] {from_block}→{to_block} too large, splitting at {mid}")
            left  = fetch_logs_range(event, event_topic, from_block, mid)
            right = fetch_logs_range(event, event_topic, mid + 1, to_block)
            return left + right
        raise


def get_logs_chunked(event, from_block: int, to_block: int) -> list:
    """
    Fetch all logs for a given event, chunked by BLOCK_CHUNK_SIZE.
    Automatically splits any chunk that exceeds the provider's result limit.
    """
    all_logs = []
    event_topic = get_event_topic(event.event_name)

    chunks = range(from_block, to_block + 1, BLOCK_CHUNK_SIZE)
    for chunk_start in tqdm(chunks, desc=f"  Fetching {event.event_name} logs"):
        chunk_end = min(chunk_start + BLOCK_CHUNK_SIZE - 1, to_block)
        logs = fetch_logs_range(event, event_topic, chunk_start, chunk_end)
        all_logs.extend(logs)
        time.sleep(0.01)

    return all_logs

def get_logs_cached(event, from_block: int, to_block: int) -> list:
    """
    Wrapper around get_logs_chunked that caches results to disk.
    If the cache file exists, load from it instead of making RPC calls.
    Delete the .pkl file to force a fresh fetch.
    """
    cache_file = f"cache/cache_{event.event_name}_{from_block}_{to_block}.pkl"

    if os.path.exists(cache_file):
        print(f"  Loading {event.event_name} logs from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            logs = pickle.load(f)
        print(f"  Loaded {len(logs):,} {event.event_name} events from cache")
        return logs

    # Cache miss — fetch from RPC and save
    logs = get_logs_chunked(event, from_block, to_block)
    with open(cache_file, "wb") as f:
        pickle.dump(logs, f)
    print(f"  Cached {len(logs):,} {event.event_name} events → {cache_file}")
    return logs

    
def get_logs_chunked_multi(events: dict, from_block: int, to_block: int) -> dict:
    """
    Fetch logs for multiple event types in a single pass over the block range.
    This is 3x more efficient than calling get_logs_chunked separately for each
    event, as it makes one RPC call per chunk instead of three.
    
    Progress is saved every 10 chunks to prevent data loss during long runs.

    Parameters
    ----------
    events : dict
        Mapping of event_name -> contract event object, e.g.:
        {
            "Mint":    pool.events.Mint(),
            "Burn":    pool.events.Burn(),
            "Collect": pool.events.Collect(),
        }
    from_block : int
    to_block   : int

    Returns
    -------
    dict : mapping of event_name -> list of decoded log objects
    """

    topic_map = {}
    for name, event in events.items():
        topic_hash = get_event_topic(name)
        topic_map[topic_hash] = (name, event)

    all_topics = list(topic_map.keys())
    results = {name: [] for name in events}
      
    # Setup for progressive caching
    CHUNKS_BETWEEN_SAVES = 10  # Save every 10 chunks (configurable)
    progress_file = f"cache/cache_multi_{from_block}_{to_block}_progress.pkl"
    chunks_processed_since_save = 0
    
    # Try to load existing progress
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "rb") as f:
                progress_data = pickle.load(f)
                results = progress_data["results"]
                last_processed_chunk_start = progress_data["last_chunk_start"]
                print(f"  Resuming from saved progress (last chunk start: {last_processed_chunk_start})")
        except Exception as e:
            print(f"  Warning: Could not load progress file: {e}")
            last_processed_chunk_start = None
    else:
        last_processed_chunk_start = None

    def save_progress(current_chunk_start: int = None):
        """Save current results to progress file"""
        progress_data = {
            "results": results,
            "last_chunk_start": current_chunk_start,
            "from_block": from_block,
            "to_block": to_block
        }
        with open(progress_file, "wb") as f:
            pickle.dump(progress_data, f)

    def fetch_chunk(chunk_start: int, chunk_end: int, chunk_size: int):
        """
        Recursively fetch logs for [chunk_start, chunk_end].
        If Infura returns a 'query returned more than 10,000 results' error,
        halve the chunk size and retry both halves.
        """
        nonlocal chunks_processed_since_save
        
        try:
            raw_logs = retry_call(
                w3.eth.get_logs,
                {
                    "address":   Web3.to_checksum_address(POOL_ADDRESS),
                    "topics":    [all_topics],
                    "fromBlock": hex(chunk_start),
                    "toBlock":   hex(chunk_end),
                }
            )
            for raw_log in raw_logs:
                log_topic = "0x" + raw_log["topics"][0].hex()
                if log_topic in topic_map:
                    name, event = topic_map[log_topic]
                    decoded = event.process_log(raw_log)
                    results[name].append(decoded)
            
            # Count this chunk as processed
            chunks_processed_since_save += 1
            
            # Auto-save if we've processed enough chunks
            if chunks_processed_since_save >= CHUNKS_BETWEEN_SAVES:
                save_progress(chunk_end)
                print(f"    Auto-saved progress after {CHUNKS_BETWEEN_SAVES} chunks "
                      f"(Total: {sum(len(v) for v in results.values()):,} events collected)")
                chunks_processed_since_save = 0
            
            time.sleep(1.0)  # eth_getLogs costs 255 CUs on Infura, max CU/s is 500

        except Exception as e:
            err_str = str(e).lower()
            if any(x for x in [
                "query returned more than",
                "10000",
                "too many results",
                "bad request",
                "response size",
                "limit exceeded",
            ] if x in err_str):
                new_chunk_size = chunk_size // 2
                if new_chunk_size < 1:
                    raise ValueError(
                        f"Chunk size reached 1 block at [{chunk_start}, {chunk_end}] "
                        f"and still too many logs. Something is wrong."
                    )
                mid = chunk_start + new_chunk_size
                # Recursively fetch the two halves
                fetch_chunk(chunk_start, mid - 1,   new_chunk_size)
                fetch_chunk(mid,         chunk_end,  new_chunk_size)
            else:
                raise  # re-raise anything unrelated to the 10k limit

    # Process chunks, skipping already processed ones if resuming
    chunks = range(from_block, to_block + 1, BLOCK_CHUNK_SIZE)
    
    # Skip chunks if resuming from progress
    skip_until = None
    if last_processed_chunk_start is not None:
        skip_until = last_processed_chunk_start
    
    for chunk_start in tqdm(chunks, desc="  Fetching Mint+Burn+Collect logs"):
        # Skip already processed chunks when resuming
        if skip_until is not None and chunk_start <= skip_until:
            continue
        
        chunk_end = min(chunk_start + BLOCK_CHUNK_SIZE - 1, to_block)
        fetch_chunk(chunk_start, chunk_end, BLOCK_CHUNK_SIZE)
        
        # Save after each chunk as an extra safety measure
        # (but we already auto-save every CHUNKS_BETWEEN_SAVES)
        
    # Final save to ensure everything is captured
    save_progress(to_block)
    
    # Clean up progress file since we're done
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print(f"  Removed temporary progress file")

    for name, logs in results.items():
        print(f"  Fetched {len(logs):,} {name} events")
    return results


def get_logs_multi_cached(events: dict, from_block: int, to_block: int) -> dict:
    """
    Fetch logs for multiple event types with disk caching.
    Uses progressive caching to save intermediate results and allow resumption
    if interrupted mid-execution.
    """
    cache_file = f"cache/cache_multi_{from_block}_{to_block}.pkl"
    progress_file = f"cache/cache_multi_{from_block}_{to_block}_progress.pkl"

    # Try to load from final cache first
    if os.path.exists(cache_file):
        print(f"  Loading Mint+Burn+Collect logs from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            results = pickle.load(f)
        for name, logs in results.items():
            print(f"  Loaded {len(logs):,} {name} events from cache")
        return results
    
    # Check for existing progress file (interrupted run)
    if os.path.exists(progress_file):
        print(f"  Found interrupted session! Resuming from progress file...")
        # Load progress and continue
        with open(progress_file, "rb") as f:
            progress_data = pickle.load(f)
            results = progress_data["results"]
            for name, logs in results.items():
                print(f"  Already fetched {len(logs):,} {name} events before interruption")
        
        # Continue fetching remaining chunks
        print(f"  Resuming from block {progress_data['last_chunk_start'] + 1}...")
        
        # Calculate remaining chunks
        remaining_results = get_logs_chunked_multi(events, from_block, to_block)
        
        # Merge results (should append to existing)
        for name in events:
            if name in remaining_results:
                results[name].extend(remaining_results[name])
        
        # Save final cache
        with open(cache_file, "wb") as f:
            pickle.dump(results, f)
        
        print(f"  Cached final results → {cache_file}")
        return results
    
    # Fresh run - fetch everything
    results = get_logs_chunked_multi(events, from_block, to_block)
    
    # Save final cache
    with open(cache_file, "wb") as f:
        pickle.dump(results, f)
    print(f"  Cached Mint+Burn+Collect logs → {cache_file}")
    return results

def find_block_at_timestamp(target_ts: int, lo: int, hi: int) -> int:
    """
    Binary search for the block number whose timestamp is closest to target_ts.
    target_ts is a Unix timestamp (integer seconds).
    lo and hi are block number bounds.
    """
    while lo < hi:
        mid = (lo + hi) // 2
        block = retry_call(w3.eth.get_block, mid)
        if block["timestamp"] < target_ts:
            lo = mid + 1
        else:
            hi = mid
    return lo

# =============================================================================
# OUTPUT 4 — slot0_snapshots.parquet
# =============================================================================

def extract_slot0_snapshots() -> pd.DataFrame:
    """
    For each day in the study window, call slot0() on the pool contract
    at the block closest to 00:00 UTC that day.

    slot0() returns the current price, tick, and other pool state.
    This requires an archive node because we query historical block states.
    """
    print("\n=== Output 4: slot0_snapshots ===")

    # Build list of daily target timestamps (00:00 UTC each day)
    start_dt = datetime(2025, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_dt   = datetime(2026, 3, 31, 0, 0, 0, tzinfo=timezone.utc)
    target_timestamps = pd.date_range(start=start_dt, end=end_dt, freq="D")

    rows = []
    for dt in tqdm(target_timestamps, desc="  Daily slot0 calls"):
        ts_unix = int(dt.timestamp())

        # Find the block closest to midnight UTC for this day
        block_num = find_block_at_timestamp(ts_unix, STUDY_START_BLOCK, STUDY_END_BLOCK)

        # Call slot0() at that specific historical block (requires archive node)
        result = retry_call(
            pool.functions.slot0().call,
            block_identifier=block_num
        )

        sqrt_price_x96, tick, obs_index, obs_card, obs_card_next, fee_protocol, unlocked = result

        rows.append({
            "snapshot_block":       block_num,
            "snapshot_timestamp":   dt,                                      # UTC datetime
            "sqrtPriceX96":         sqrt_price_x96,                          # raw uint160
            "price_usdc_per_weth":  sqrtPriceX96_to_price(sqrt_price_x96),   # human-readable
            "current_tick":         tick,                                     # int24
            "observation_index":    obs_index,
            "unlocked":             unlocked,
        })

    df = pd.DataFrame(rows)
    
    # sqrtPriceX96 is a uint160 — too large for PyArrow's int64.
	# Cast to string to preserve the exact value without overflow.
    df["sqrtPriceX96"] = df["sqrtPriceX96"].astype(str)
    
    df.to_parquet(OUT_SLOT0_SNAP, index=False)
    print(f"  Saved {len(df):,} rows → {OUT_SLOT0_SNAP}")
    return df

# =============================================================================
# OUTPUT 1 — swap_events.parquet
# =============================================================================
    
def extract_swap_events() -> pd.DataFrame:
    """
    Fetch every Swap event emitted by the pool during the study window.

    The Swap event encodes:
        amount0  (int256) — signed USDC amount (positive = USDC into pool)
        amount1  (int256) — signed WETH amount (positive = WETH into pool)
        sqrtPriceX96     — pool price after the swap
        liquidity        — active liquidity at time of swap
        tick             — active tick after the swap

    Trade direction (from the taker's perspective):
        amount0 > 0 means USDC flowed into the pool → taker received WETH → "buy_weth"
        amount0 < 0 means USDC flowed out of pool   → taker sent WETH    → "sell_weth"
    """
    
    print("\n=== Output 1: swap_events ===")
    
    logs = get_logs_cached(pool.events.Swap(), STUDY_START_BLOCK, STUDY_END_BLOCK)
    print(f"  Fetched {len(logs):,} Swap events")
    
    block_nums = [log["blockNumber"] for log in logs]
    timestamps = fetch_block_timestamps(block_nums)
    
    # Identify any blocks that are still missing from the cache
    missing_blocks = list(set(block_nums) - set(timestamps.keys()))
    if missing_blocks:
        print(f"  Fetching {len(missing_blocks):,} missing blocks individually...")
        for bn in tqdm(missing_blocks, desc="  Missing blocks"):
            block = retry_call(w3.eth.get_block, bn)
            timestamps[bn] = datetime.fromtimestamp(block["timestamp"], tz=timezone.utc)
        # Update the cache with the newly fetched blocks
        with open("cache/cache_timestamps.pkl", "wb") as f:
            pickle.dump(timestamps, f)
        print(f"  Timestamp cache updated with missing blocks")

    # Now build the DataFrame — all blocks guaranteed to be in timestamps
    rows = []
    for log in logs:
        args = log["args"]
        a0   = args["amount0"]
        a1   = args["amount1"]
        sqrt = args["sqrtPriceX96"]

        a0_dec = a0 / (10 ** USDC_DECIMALS)
        a1_dec = a1 / (10 ** WETH_DECIMALS)

        direction    = "buy_weth" if a0 > 0 else "sell_weth"
        notional_usd = abs(a0_dec)

        rows.append({
            "block_number":        log["blockNumber"],
            "block_timestamp":     timestamps[log["blockNumber"]],
            "transaction_hash":    log["transactionHash"].hex(),
            "amount0_raw":         str(a0),
            "amount0_decimal":     a0_dec,
            "amount1_raw":         str(a1),
            "amount1_decimal":     a1_dec,
            "sqrtPriceX96":        str(sqrt),
            "price_usdc_per_weth": sqrtPriceX96_to_price(sqrt),
            "liquidity":           str(args["liquidity"]),
            "tick":                args["tick"],
            "trade_direction":     direction,
            "notional_usd":        notional_usd,
        })

    df = pd.DataFrame(rows)
    df["sqrtPriceX96"] = df["sqrtPriceX96"].astype(str)
    df["liquidity"]    = df["liquidity"].astype(str)
    df["amount0_raw"]  = df["amount0_raw"].astype(str)
    df["amount1_raw"]  = df["amount1_raw"].astype(str)

    df.to_parquet(OUT_SWAP, index=False)
    print(f"  Saved {len(df):,} rows → {OUT_SWAP}")
    return df
    
# =============================================================================
# OUTPUT 2 — mint_burn_events.parquet
# =============================================================================

def extract_mint_burn_events() -> pd.DataFrame:
    """
    Fetch every Mint and Burn event from pool deployment to end of study window.

    We must start from POOL_DEPLOY_BLOCK (not the study window start) because
    the liquidity map at the start of our window is the cumulative result of
    ALL prior LP actions since deployment.

    We also collect Collect events in this same pass (needed for Module 4)
    to avoid a second full historical scan.
    """
    print("\n=== Output 2: mint_burn_events ===")
    print(f"  Scanning from deployment block {POOL_DEPLOY_BLOCK:,} to {STUDY_END_BLOCK:,}")
    print("  (Single-pass scan for Mint + Burn + Collect simultaneously)")
    
    # Single scan for all three event types
    all_logs = get_logs_multi_cached(
        {
            "Mint":    pool.events.Mint(),
            "Burn":    pool.events.Burn(),
            "Collect": pool.events.Collect(),
        },
        from_block=POOL_DEPLOY_BLOCK,
        to_block=STUDY_END_BLOCK,
    )

    mint_logs    = all_logs["Mint"]
    burn_logs    = all_logs["Burn"]
    collect_logs = all_logs["Collect"]

    # --- Save Collect events for Module 4 ---
    collect_rows = []
    for log in collect_logs:
        args = log["args"]
        collect_rows.append({
            "block_number":    log["blockNumber"],
            "transaction_hash": log["transactionHash"].hex(),
            "owner":           args["owner"],
            "recipient":       args["recipient"],
            "tick_lower":      args["tickLower"],
            "tick_upper":      args["tickUpper"],
            "amount0_raw":     str(args["amount0"]),
            "amount1_raw":     str(args["amount1"]),
            "amount0_decimal": args["amount0"] / (10 ** USDC_DECIMALS),
            "amount1_decimal": args["amount1"] / (10 ** WETH_DECIMALS),
        })
    if collect_rows:
        collect_df = pd.DataFrame(collect_rows)
        collect_df.to_parquet("cache/collect_events.parquet", index=False)
        print(f"  Saved {len(collect_df):,} Collect rows → collect_events.parquet")

    # --- Build mint/burn DataFrame ---
    all_block_nums = (
        [log["blockNumber"] for log in mint_logs] +
        [log["blockNumber"] for log in burn_logs]
    )
    timestamps = fetch_block_timestamps(all_block_nums)

    rows = []
    for log in mint_logs:
        args = log["args"]
        rows.append({
            "block_number":     log["blockNumber"],
            "block_timestamp":  timestamps[log["blockNumber"]],
            "transaction_hash": log["transactionHash"].hex(),
            "event_type":       "mint",
            "owner":            args["owner"],
            "tick_lower":       args["tickLower"],
            "tick_upper":       args["tickUpper"],
            "liquidity_raw":    str(args["amount"]),
            "amount0_raw":      str(args["amount0"]),
            "amount1_raw":      str(args["amount1"]),
            "amount0_decimal":  args["amount0"] / (10 ** USDC_DECIMALS),
            "amount1_decimal":  args["amount1"] / (10 ** WETH_DECIMALS),
        })

    for log in burn_logs:
        args = log["args"]
        rows.append({
            "block_number":     log["blockNumber"],
            "block_timestamp":  timestamps[log["blockNumber"]],
            "transaction_hash": log["transactionHash"].hex(),
            "event_type":       "burn",
            "owner":            args["owner"],
            "tick_lower":       args["tickLower"],
            "tick_upper":       args["tickUpper"],
            "liquidity_raw":    str(args["amount"]),
            "amount0_raw":      str(args["amount0"]),
            "amount1_raw":      str(args["amount1"]),
            "amount0_decimal":  args["amount0"] / (10 ** USDC_DECIMALS),
            "amount1_decimal":  args["amount1"] / (10 ** WETH_DECIMALS),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("block_number").reset_index(drop=True)
    df.to_parquet(OUT_MINT_BURN, index=False)
    print(f"  Saved {len(df):,} rows → {OUT_MINT_BURN}")
    return df

# =============================================================================
# OUTPUT 3 — liquidity_snapshots.parquet
# =============================================================================

def extract_liquidity_snapshots(mint_burn_df: pd.DataFrame, slot0_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct the full tick-level liquidity map at each daily snapshot block
    by replaying Mint and Burn events chronologically.

    How Uniswap V3 liquidity works:
    --------------------------------
    Each LP position adds liquidity to a range [tickLower, tickUpper].
    The pool tracks liquidityNet at each tick boundary:
        - At tickLower: +amount is added on Mint, -amount on Burn
        - At tickUpper: -amount is added on Mint, +amount on Burn

    To find the active liquidity at any price:
        Start at MIN_TICK with liquidity = 0
        Walk upward, adding liquidityNet at each initialised tick
        The running total at any tick is the active liquidity in that range

    liquidityGross tracks the total absolute liquidity touching a tick
    (used to know if a tick is "initialised" — i.e. has any active positions).
    """
    print("\n=== Output 3: liquidity_snapshots ===")

    # We will replay events up to each snapshot block
    # Sort snapshot blocks ascending
    snapshot_blocks = sorted(slot0_df["snapshot_block"].tolist())

    # Sort mint/burn events chronologically (should already be sorted, but ensure)
    mb = mint_burn_df.sort_values("block_number").reset_index(drop=True)

    # Running state: liquidityNet and liquidityGross per tick
    # Using defaultdict so missing ticks default to 0
    liquidity_net   = defaultdict(int)   # tick -> net liquidity (signed)
    liquidity_gross = defaultdict(int)   # tick -> gross liquidity (unsigned)

    event_idx = 0          # Pointer into the mint/burn event list
    n_events  = len(mb)

    all_rows = []

    for snap_block in tqdm(snapshot_blocks, desc="  Building snapshots"):

        # Replay all events up to (and including) this snapshot block
        while event_idx < n_events and mb.iloc[event_idx]["block_number"] <= snap_block:
            row   = mb.iloc[event_idx]
            tl    = int(row["tick_lower"])
            tu    = int(row["tick_upper"])
            liq   = int(row["liquidity_raw"])

            if row["event_type"] == "mint":
                # Adding liquidity: tickLower gets +liq, tickUpper gets -liq
                liquidity_net[tl]   += liq
                liquidity_net[tu]   -= liq
                liquidity_gross[tl] += liq
                liquidity_gross[tu] += liq

            else:  # burn
                # Removing liquidity: reverse of mint
                liquidity_net[tl]   -= liq
                liquidity_net[tu]   += liq
                liquidity_gross[tl] -= liq
                liquidity_gross[tu] -= liq

            event_idx += 1

        # Snapshot: record all initialised ticks (liquidityGross > 0)
        # and compute active liquidity by walking from MIN_TICK upward

        # Get sorted list of all initialised ticks
        initialised_ticks = sorted(
            tick for tick, gross in liquidity_gross.items() if gross > 0
        )

        # Walk from MIN_TICK upward, accumulating liquidityNet
        running_liquidity = 0
        tick_active_liq   = {}   # tick -> active liquidity in the range starting at this tick

        for tick in initialised_ticks:
            running_liquidity += liquidity_net[tick]
            tick_active_liq[tick] = running_liquidity

        # Get snapshot metadata
        snap_row = slot0_df[slot0_df["snapshot_block"] == snap_block].iloc[0]

        for tick in initialised_ticks:
            all_rows.append({
                "snapshot_block":      snap_block,
                "snapshot_timestamp":  snap_row["snapshot_timestamp"],
                "tick":                tick,
                "liquidity_net":       liquidity_net[tick],       # signed int
                "liquidity_gross":     liquidity_gross[tick],     # unsigned int
                "active_liquidity":    tick_active_liq[tick],     # running total above this tick
                "price_lower":         tick_to_price(tick),           # USDC/WETH at lower edge
                "price_upper":         tick_to_price(tick + TICK_SPACING),  # USDC/WETH at upper edge
            })

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_LIQ_SNAP, index=False)
    print(f"  Saved {len(df):,} rows → {OUT_LIQ_SNAP}")
    return df

# =============================================================================
# VALIDATION
# =============================================================================

def validate_liquidity_map(snap_df: pd.DataFrame, slot0_df: pd.DataFrame):
    """
    Spot-check the reconstructed liquidity map against live archive node calls.

    For one snapshot block, pick 10 initialised ticks and call pool.ticks(tick)
    at that block via eth_call. Compare liquidityNet and liquidityGross.
    Results must match exactly.

    This validates that our event-replay reconstruction is correct.
    """
    print("\n=== Validation: Liquidity Map Spot-Check ===")

    # Use the first snapshot block as the validation target
    val_block = int(slot0_df["snapshot_block"].iloc[0])
    snap_subset = snap_df[snap_df["snapshot_block"] == val_block]

    # Pick 10 initialised ticks spread across the range
    sample_ticks = snap_subset["tick"].dropna().astype(int).tolist()
    step = max(1, len(sample_ticks) // 10)
    sample_ticks = sample_ticks[::step][:10]

    print(f"  Validating {len(sample_ticks)} ticks at block {val_block:,}")
    print(f"  {'Tick':>10}  {'Our liquidityNet':>18}  {'Chain liquidityNet':>18}  {'Match':>6}")
    print("  " + "-" * 60)

    all_match = True
    for tick in sample_ticks:
        # Call pool.ticks(tick) at the specific historical block
        chain_result = retry_call(
            pool.functions.ticks(tick).call,
            block_identifier=val_block
        )
        chain_gross = chain_result[0]   # liquidityGross (uint128)
        chain_net   = chain_result[1]   # liquidityNet   (int128)

        our_row   = snap_subset[snap_subset["tick"] == tick].iloc[0]
        our_net   = int(our_row["liquidity_net"])
        our_gross = int(our_row["liquidity_gross"])

        net_match   = our_net   == chain_net
        gross_match = our_gross == chain_gross
        match = net_match and gross_match

        if not match:
            all_match = False

        print(f"  {tick:>10}  {our_net:>18}  {chain_net:>18}  {'✓' if match else '✗ MISMATCH':>6}")

    print()
    if all_match:
        print("  ✓ All ticks match. Liquidity reconstruction is correct.")
    else:
        print("  ✗ Some ticks did not match. Investigate event replay logic.")


def validate_swap_volume(swap_df: pd.DataFrame):
    """
    Cross-check: print total notional volume over the study window.
    Compare manually against Uniswap info / Dune Analytics for the same period.
    """
    print("\n=== Validation: Volume Cross-Check ===")
    total_vol = swap_df["notional_usd"].sum()
    n_swaps   = len(swap_df)
    print(f"  Total swaps in study window:  {n_swaps:,}")
    print(f"  Total notional volume (USD):  ${total_vol:,.0f}")
    print("  → Compare this figure against Uniswap Info or Dune Analytics")
    print("    for pool 0x88e6...5640 over Oct 2025 – Mar 2026.")


def validate_slot0_vs_swaps(slot0_df: pd.DataFrame, swap_df: pd.DataFrame):
    """
    Consistency check: for each daily snapshot, compare the price from slot0()
    to the last swap price recorded before that block.
    The two should agree to within one tick spacing (≈ 0.01%).
    """
    print("\n=== Validation: slot0 vs Last Swap Price ===")

    swap_sorted = swap_df.sort_values("block_number")
    discrepancies = 0

    for _, snap in slot0_df.iterrows():
        snap_block  = snap["snapshot_block"]
        slot0_price = snap["price_usdc_per_weth"]

        # Find the last swap at or before this block
        prior_swaps = swap_sorted[swap_sorted["block_number"] <= snap_block]
        if prior_swaps.empty:
            continue

        last_swap_price = prior_swaps.iloc[-1]["price_usdc_per_weth"]

        # Relative difference should be within one tick spacing (0.01% = 0.0001)
        rel_diff = abs(slot0_price - last_swap_price) / last_swap_price
        if rel_diff > 0.0001:
            discrepancies += 1
            print(f"  Block {snap_block:,}: slot0={slot0_price:.2f}, "
                  f"last_swap={last_swap_price:.2f}, diff={rel_diff:.4%}")

    if discrepancies == 0:
        print("  ✓ All snapshot prices agree with last swap price within tick spacing.")
    else:
        print(f"  ✗ {discrepancies} snapshots have price discrepancies > 0.01%. Investigate.")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Uniswap V3 USDC/WETH Data Extraction")
    print(f"Pool:         {POOL_ADDRESS}")
    print(f"Study window: blocks {STUDY_START_BLOCK:,} → {STUDY_END_BLOCK:,}")
    print("=" * 60)

    # Run extractions in dependency order
    #slot0_df    = extract_slot0_snapshots()     # Output 4 (needed by Output 3)
    slot0_df    = pd.read_parquet(OUT_SLOT0_SNAP) 
    #swap_df     = extract_swap_events()          # Output 1
    swap_df     = pd.read_parquet(OUT_SWAP)
    mb_df       = extract_mint_burn_events()     # Output 2
    snap_df     = extract_liquidity_snapshots(mb_df, slot0_df)  # Output 3

    # Run validations
    validate_liquidity_map(snap_df, slot0_df)
    validate_swap_volume(swap_df)
    validate_slot0_vs_swaps(slot0_df, swap_df)

    print("\n✓ All done. Four Parquet files written to current directory.")


if __name__ == "__main__":
    main()
