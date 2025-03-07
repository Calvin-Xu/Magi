import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool

from magi.config import REDIS_CONFIG

#
# Lua scripts for concurrency
#
# ACQUIRE_SCRIPT: checks current concurrency, compares to limit, increments if below.
ACQUIRE_SCRIPT = """
local current = redis.call('GET', KEYS[1])
if not current then
    current = 0
else
    current = tonumber(current)
end

if current < tonumber(ARGV[1]) then
    current = current + 1
    -- Set key with a small expiry as a safeguard to avoid stale concurrency
    redis.call('SET', KEYS[1], current, 'EX', ARGV[2])
    return 1
else
    return 0
end
"""

# RELEASE_SCRIPT: decrement concurrency if > 0
RELEASE_SCRIPT = """
local current = redis.call('GET', KEYS[1])
if not current then
    return 0
end

current = tonumber(current)
if current > 0 then
    current = current - 1
    redis.call('SET', KEYS[1], current, 'EX', ARGV[1])
end
return 1
"""


class RateLimit:
    """Configuration for a rate limit (tokens and concurrency)."""

    def __init__(
        self,
        name: str,
        rpm: int,
        tpm: int,
        window_size: int = 60,
        num_shards: int = 10,
        max_concurrent: int = 20,
    ):
        """
        :param name: Unique name for the rate limit key set.
        :param rpm: Requests per minute limit.
        :param tpm: Tokens per minute limit (e.g., total tokens used).
        :param window_size: Rolling window length in seconds.
        :param num_shards: Number of shards for distributing usage.
        :param max_concurrent: Maximum concurrency allowed at once.
        """
        self.name = name
        self.rpm = rpm
        self.tpm = tpm
        self.window_size = window_size
        self.num_shards = num_shards
        self.max_concurrent = max_concurrent


class RateLimitContext:
    """Context manager wrapper for using DistributedRateLimiter inside an async `with`."""

    def __init__(
        self,
        limiter: "DistributedRateLimiter",
        rate_limit: RateLimit,
        tokens: int,
        reserve: bool = True,
        key: Optional[str] = None,
    ):
        self.limiter = limiter
        self.rate_limit = rate_limit
        self.tokens = tokens
        self.reserve = reserve
        self.key = key
        self.retry_after: Optional[float] = None

    async def __aenter__(self) -> Optional[float]:
        """
        Attempt to acquire. If successful, returns None.
        Otherwise returns a float epoch timestamp at which you may retry.
        """
        self.retry_after = await self.limiter.acquire(
            rate_limit=self.rate_limit,
            tokens=self.tokens,
            reserve=self.reserve,
            key=self.key,
        )
        return self.retry_after

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release concurrency slot if we actually acquired it."""
        if self.retry_after is None:
            await self.limiter.release(self.rate_limit)


class DistributedRateLimiter:
    """
    Distributed rate limiter using Redis.
    - Token bucket for usage (RPM/TPM).
    - Concurrency limit with Lua scripts.
    - Optional "reservation" of future usage if you exceed the limit.
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        jitter_factor: float = 0.1,  # proportion of base wait time to add as random jitter
    ):
        self.redis = redis_client
        self.jitter_factor = jitter_factor
        # Local locks so that calls in the same process do not race when reading/updating usage
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_or_create_lock(self, rate_limit: RateLimit) -> asyncio.Lock:
        """Get a local process lock for usage checks on this rate_limit."""
        if rate_limit.name not in self._locks:
            self._locks[rate_limit.name] = asyncio.Lock()
        return self._locks[rate_limit.name]

    def _get_shard_key(self, rate_limit: RateLimit, shard: int) -> str:
        """Redis sorted set key to store usage events."""
        return f"{rate_limit.name}:shard:{shard}"

    def _get_reservation_key(self, rate_limit: RateLimit, shard: int) -> str:
        """Redis sorted set key for usage reservations (future)."""
        return f"{rate_limit.name}:reservations:shard:{shard}"

    def _get_concurrency_key(self, rate_limit: RateLimit) -> str:
        """Redis key for concurrency counting."""
        return f"{rate_limit.name}:concurrency"

    def _add_jitter(self, base_time: float, initial: bool = False) -> float:
        """
        Add some random jitter (in seconds) to the base_time (which is also an epoch timestamp).
        This helps avoid stampedes at the exact same second.
        """
        if initial:
            # Slightly larger jitter for the first time
            base_jitter = 12.0
        else:
            base_jitter = 6.0
        return base_time + random.random() * base_jitter

    async def _acquire_concurrency(self, rate_limit: RateLimit) -> bool:
        """
        Increment concurrency if below the configured max_concurrent. Returns True if acquired.
        """
        concurrency_key = self._get_concurrency_key(rate_limit)
        # Evaluate the ACQUIRE_SCRIPT in Redis
        result = await self.redis.eval(
            ACQUIRE_SCRIPT,
            1,  # numkeys
            concurrency_key,
            rate_limit.max_concurrent,
            rate_limit.window_size * 2,
        )
        return bool(result)

    async def _release_concurrency(self, rate_limit: RateLimit) -> None:
        """
        Decrement concurrency count (if > 0).
        """
        concurrency_key = self._get_concurrency_key(rate_limit)
        await self.redis.eval(
            RELEASE_SCRIPT,
            1,  # numkeys
            concurrency_key,
            rate_limit.max_concurrent,
            rate_limit.window_size * 2,
        )

    async def _choose_shard(self, rate_limit: RateLimit, key: Optional[str]) -> int:
        """
        'Power of two choices' approach: pick two random shards and choose the one
        with fewer usage events. If `key` is provided, we use a stable hash to pick the shards.
        """
        if not key:
            return random.randrange(rate_limit.num_shards)

        # Stable hashing to pick two shards, then pick the shard with fewer usage
        shard1 = hash(key + ":1") % rate_limit.num_shards
        shard2 = hash(key + ":2") % rate_limit.num_shards
        pipe = self.redis.pipeline(transaction=False)
        pipe.zcard(self._get_shard_key(rate_limit, shard1))
        pipe.zcard(self._get_shard_key(rate_limit, shard2))
        usage_counts = await pipe.execute()
        return shard1 if usage_counts[0] <= usage_counts[1] else shard2

    async def _get_usage(self, rate_limit: RateLimit, now: datetime) -> Tuple[int, int]:
        """
        Calculate how many requests (RPM) and tokens (TPM) have been used
        in the rolling window that ends at `now`.

        - Removes expired usage older than `cutoff` from usage sets.
        - Does NOT add future reservations to 'current usage'.
        """
        cutoff = (now - timedelta(seconds=rate_limit.window_size)).timestamp()
        now_ts = now.timestamp()

        total_rpm = 0
        total_tpm = 0

        pipe = self.redis.pipeline(transaction=False)

        for shard in range(rate_limit.num_shards):
            usage_key = self._get_shard_key(rate_limit, shard)
            reservation_key = self._get_reservation_key(rate_limit, shard)

            # 1) Remove usage older than cutoff
            pipe.zremrangebyscore(usage_key, "-inf", cutoff)
            # 2) Remove reservations that have fully expired (time < now)
            pipe.zremrangebyscore(reservation_key, "-inf", now_ts)

            # Now read usage from [cutoff .. now_ts] only
            pipe.zrange(
                usage_key,
                start=cutoff,
                end=now_ts,
                byscore=True,
                withscores=True,
            )
            # We do NOT sum up future reservations. We'll read them if you need them,
            # but do not add them to total usage here:
            pipe.zrange(
                reservation_key,
                start=now_ts,  # only read reservations at or after 'now'
                end="+inf",
                byscore=True,
                withscores=True,
            )

        results = await pipe.execute()

        idx = 0
        # Each shard uses 4 pipeline commands
        for shard in range(rate_limit.num_shards):
            # after the zremrangebyscore calls, usage_list is next, then reservation_list
            usage_list = results[idx + 2]  # zrange usage
            # reservation_list = results[idx + 3]  # zrange reservations (unused in totals)
            idx += 4

            # Sum usage
            total_rpm += len(usage_list)
            total_tpm += sum(json.loads(u[0])["tokens"] for u in usage_list)

        return total_rpm, total_tpm

    async def acquire(
        self,
        rate_limit: RateLimit,
        tokens: int,
        reserve: bool = True,
        key: Optional[str] = None,
    ) -> Optional[float]:
        """
        Try to acquire both usage capacity (rpm/tpm) and concurrency.

        :return:
          - None if acquisition succeeded immediately (use now).
          - A float epoch timestamp if you must wait until that time (do NOT proceed yet).

        If you get None, you MUST call `release(rate_limit)` later to free concurrency.
        """
        lock = self._get_or_create_lock(rate_limit)
        now = datetime.now()

        async with lock:
            total_rpm, total_tpm = await self._get_usage(rate_limit, now)

            # 1) Check usage-based limit
            if (total_rpm >= rate_limit.rpm) or (total_tpm + tokens > rate_limit.tpm):
                # Over limit
                if not reserve:
                    # Return approximate next time for the caller to wait or back off
                    retry_after = now.timestamp() + (
                        rate_limit.window_size / rate_limit.rpm
                    )
                    return self._add_jitter(retry_after, initial=True)

                # Reserve usage for the future
                # next_slot is an approximate time we can safely re-try
                # e.g. if total_tpm is 10000 and tpm=20000, we wait half the window, etc.
                next_slot = now.timestamp() + (
                    rate_limit.window_size * (total_tpm + tokens) / rate_limit.tpm
                )
                next_slot = self._add_jitter(next_slot, initial=True)

                chosen_shard = await self._choose_shard(rate_limit, key)
                await self.redis.zadd(
                    self._get_reservation_key(rate_limit, chosen_shard),
                    mapping={json.dumps({"tokens": tokens}): next_slot},
                )
                return next_slot

            # 2) Check concurrency limit
            acquired_concurrency = await self._acquire_concurrency(rate_limit)
            if not acquired_concurrency:
                # Concurrency is fully used. Wait a short while, or do exponential backoff, etc.
                retry_after = now.timestamp() + 5.0
                return self._add_jitter(retry_after, initial=True)

            # 3) Record usage event now that concurrency is ours
            chosen_shard = await self._choose_shard(rate_limit, key)
            await self.redis.zadd(
                self._get_shard_key(rate_limit, chosen_shard),
                mapping={json.dumps({"tokens": tokens}): now.timestamp()},
            )

            # Success
            return None

    async def release(self, rate_limit: RateLimit) -> None:
        """Decrement concurrency so another caller can use it."""
        await self._release_concurrency(rate_limit)

    async def close(self):
        """Close Redis resources if needed."""
        await self.redis.close()

    def acquire_context(
        self,
        rate_limit: RateLimit,
        tokens: int,
        reserve: bool = True,
        key: Optional[str] = None,
    ) -> RateLimitContext:
        """
        Return an async context manager for usage:

        async with rate_limiter.acquire_context(rate_limit, tokens) as retry_after:
            if retry_after is not None:
                # We didn't acquire. retry_after is an epoch float. Wait or handle an error.
                wait_seconds = max(0, retry_after - time.time())
                await asyncio.sleep(wait_seconds)
            else:
                # success; do the work
                ...
        """
        return RateLimitContext(self, rate_limit, tokens, reserve, key)


#
# Create a shared Redis client + single instance of the rate limiter.
#
pool = ConnectionPool(
    host=REDIS_CONFIG.host,
    port=REDIS_CONFIG.port,
    max_connections=50,
)
shared_redis_client = aioredis.Redis(connection_pool=pool)
rate_limiter = DistributedRateLimiter(shared_redis_client)
