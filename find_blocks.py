# find_blocks.py
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()
w3 = Web3(Web3.HTTPProvider(os.getenv("RPC_URL")))

def find_block_at_date(year, month, day):
    target_ts = int(datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    
    # Rough bounds — no need to search the entire chain
    lo = 20_000_000
    hi = w3.eth.block_number

    while lo < hi:
        mid = (lo + hi) // 2
        ts = w3.eth.get_block(mid)["timestamp"]
        if ts < target_ts:
            lo = mid + 1
        else:
            hi = mid

    block = w3.eth.get_block(lo)
    dt = datetime.fromtimestamp(block["timestamp"], tz=timezone.utc)
    print(f"{year}-{month:02d}-{day:02d} → block {lo:,}  (actual timestamp: {dt})")
    return lo

find_block_at_date(2025, 10, 1)   # Study start
find_block_at_date(2026,  3, 31)  # Study end
