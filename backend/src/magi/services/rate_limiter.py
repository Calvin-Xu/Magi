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
    """Configuration for a rate limit."""

    def __init__(
        self,
        name: str,
        rpm: int,
        tpm: int,
        window_size: int = 60,
        num_shards: int = 10,
        max_concurrent: int = 20,
    ):
        """Initialize rate limit config."""
        self.name = name
        self.rpm = rpm  # requests per minute
        self.tpm = tpm  # tokens  per minute
        self.window_size = window_size
        self.num_shards = num_shards
        self.max_concurrent = max_concurrent


class RateLimitContext:
    """Context manager for rate limiting."""

    def __init__(
        self,
        limiter: "DistributedRateLimiter",
        rate_limit: RateLimit,
        tokens: int,
        reserve: bool = True,
        key: Optional[str] = None,
    ):
        """Initialize the context manager."""
        self.limiter = limiter
        self.rate_limit = rate_limit
        self.tokens = tokens
        self.reserve = reserve
        self.key = key
        self.retry_after = None

    async def __aenter__(self) -> Optional[float]:
        """Acquire rate limit."""
        self.retry_after = await self.limiter.acquire(
            self.rate_limit, tokens=self.tokens, reserve=self.reserve, key=self.key
        )
        return self.retry_after

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release rate limit if acquired."""
        if self.retry_after is None:  # Only release if we successfully acquired
            await self.limiter.release(self.rate_limit)


class DistributedRateLimiter:
    """Token bucket + distributed concurrency limit using Redis."""

    def __init__(
        self,
        redis_client: aioredis.Redis,
        jitter_factor: float = 0.1,  # 10% jitter by default
    ):
        """
        :param redis_client: A pre-configured redis.asyncio.Redis client with a connection pool.
        :param jitter_factor: proportion of base wait time to add as random jitter.
        """
        self.jitter_factor = jitter_factor
        self.redis = redis_client
        self._locks: Dict[str, asyncio.Lock] = {}
        # Local lock used only to serialize `_get_usage` calls from *this* process
        # (does not protect concurrency globally!)
        # If you want usage checks to be strongly atomic across processes, you’d need
        # a Lua script or watch/multi/exec transaction.
        # For many use cases, best-effort is sufficient.

    def _get_or_create_lock(self, rate_limit: RateLimit) -> asyncio.Lock:
        """Local lock so two coroutines in the same process don't run usage check concurrently."""
        if rate_limit.name not in self._locks:
            self._locks[rate_limit.name] = asyncio.Lock()
        return self._locks[rate_limit.name]

    def _get_shard_key(self, rate_limit: RateLimit, shard: int) -> str:
        """Get Redis key for usage shard."""
        return f"{rate_limit.name}:shard:{shard}"

    def _get_reservation_key(self, rate_limit: RateLimit, shard: int) -> str:
        """Get Redis key for reservations."""
        return f"{rate_limit.name}:reservations:shard:{shard}"

    def _get_concurrency_key(self, rate_limit: RateLimit) -> str:
        """Key for global concurrency tracking."""
        return f"{rate_limit.name}:concurrency"

    def _add_jitter(self, base_time: float, initial: bool = False) -> float:
        """Add randomized jitter to the future wait time (in epoch seconds)."""
        if initial:
            # Larger jitter on the first time
            base_jitter = 12.0
            return base_time + random.random() * base_jitter
        else:
            max_jitter = 6.0
            return base_time + random.random() * max_jitter

    async def _acquire_concurrency(self, rate_limit: RateLimit) -> bool:
        concurrency_key = self._get_concurrency_key(rate_limit)
        # Use positional arguments instead of keys=..., args=...
        result = await self.redis.eval(
            ACQUIRE_SCRIPT,
            1,  # numkeys
            concurrency_key,
            rate_limit.max_concurrent,
            rate_limit.window_size * 2,
        )
        return bool(result)

    async def _release_concurrency(self, rate_limit: RateLimit) -> None:
        concurrency_key = self._get_concurrency_key(rate_limit)
        await self.redis.eval(
            RELEASE_SCRIPT,
            1,  # numkeys
            concurrency_key,
            rate_limit.window_size * 2,
        )

    async def _choose_shard(self, rate_limit: RateLimit, key: Optional[str]) -> int:
        """Use the 'power of two choices' to pick a shard with presumably lower usage."""
        if not key:
            return random.randrange(rate_limit.num_shards)

        shard1 = hash(key + ":1") % rate_limit.num_shards
        shard2 = hash(key + ":2") % rate_limit.num_shards

        # Instead of 'async with self.redis.pipeline()', do this:
        pipe = self.redis.pipeline(transaction=False)
        pipe.zcard(self._get_shard_key(rate_limit, shard1))
        pipe.zcard(self._get_shard_key(rate_limit, shard2))
        usage_counts = await pipe.execute()

        return shard1 if usage_counts[0] <= usage_counts[1] else shard2

    async def _get_usage(self, rate_limit: RateLimit, now: datetime) -> Tuple[int, int]:
        """Aggregate usage across all shards + reservations."""
        cutoff = (now - timedelta(seconds=rate_limit.window_size)).timestamp()

        total_rpm = 0
        total_tpm = 0
        reserved_rpm = 0
        reserved_tpm = 0

        # Similarly here, remove the 'async with' usage
        pipe = self.redis.pipeline(transaction=False)
        for shard in range(rate_limit.num_shards):
            usage_key = self._get_shard_key(rate_limit, shard)
            reservation_key = self._get_reservation_key(rate_limit, shard)

            pipe.zremrangebyscore(usage_key, "-inf", cutoff)
            pipe.zremrangebyscore(reservation_key, "-inf", now.timestamp())
            pipe.zrange(usage_key, 0, -1, withscores=True)
            pipe.zrange(reservation_key, 0, -1, withscores=True)

        results = await pipe.execute()
        # Now parse 'results' the same way as before
        idx = 0
        for shard in range(rate_limit.num_shards):
            usage_list = results[idx + 2]
            reservation_list = results[idx + 3]

            total_rpm += len(usage_list)
            total_tpm += sum(json.loads(u[0])["tokens"] for u in usage_list)

            reserved_rpm += len(reservation_list)
            reserved_tpm += sum(json.loads(r[0])["tokens"] for r in reservation_list)
            idx += 4

        return (total_rpm + reserved_rpm), (total_tpm + reserved_tpm)

    async def acquire(
        self,
        rate_limit: RateLimit,
        tokens: int,
        reserve: bool = True,
        key: Optional[str] = None,
    ) -> Optional[float]:
        """
        Attempt to acquire rate-limit capacity *and* a concurrency slot.

        Returns:
          - None if acquisition succeeded (you may proceed immediately).
          - A float (epoch time) if you must wait until that time before trying again.

        Once you get None (success), you **must** call `release(rate_limit)` later
        to free up the concurrency slot (e.g., in a `finally:` block).
        """
        # return None
        lock = self._get_or_create_lock(rate_limit)
        now = datetime.now()

        async with lock:
            total_rpm, total_tpm = await self._get_usage(rate_limit, now)

            # 1) Check usage-based limit
            if (total_rpm >= rate_limit.rpm) or (total_tpm + tokens > rate_limit.tpm):
                # Over limit: either reserve or return
                if not reserve:
                    # Return approximate next time
                    retry_after = now.timestamp() + (
                        rate_limit.window_size / rate_limit.rpm
                    )
                    return self._add_jitter(retry_after, initial=True)

                # Reserve usage in future
                next_slot = now.timestamp() + (
                    rate_limit.window_size * (total_tpm + tokens) / rate_limit.tpm
                )
                next_slot = self._add_jitter(next_slot)

                # Put reservation into whichever shard is less loaded
                chosen_shard = await self._choose_shard(rate_limit, key)
                await self.redis.zadd(
                    self._get_reservation_key(rate_limit, chosen_shard),
                    {json.dumps({"tokens": tokens}): next_slot},
                )
                return next_slot

            # 2) Check concurrency limit
            acquired_concurrency = await self._acquire_concurrency(rate_limit)
            if not acquired_concurrency:
                # Concurrency is fully used. Return an approximate wait time.
                # (This is somewhat naive; you might use a fixed backoff or exponential approach.)
                retry_after = now.timestamp() + 5.0  # e.g., wait 5s before retry
                return self._add_jitter(retry_after, initial=True)

            # 3) Record usage event now that concurrency is ours
            chosen_shard = await self._choose_shard(rate_limit, key)
            await self.redis.zadd(
                self._get_shard_key(rate_limit, chosen_shard),
                {json.dumps({"tokens": tokens}): now.timestamp()},
            )

            # Return success (None means proceed)
            return None

    async def release(self, rate_limit: RateLimit) -> None:
        """
        Decrement concurrency when your actual request/work completes.
        You should call this in a finally-block or equivalent.
        """
        await self._release_concurrency(rate_limit)

    async def close(self):
        """Close Redis connection if needed."""
        await self.redis.close()

    def acquire_context(
        self,
        rate_limit: RateLimit,
        tokens: int,
        reserve: bool = True,
        key: Optional[str] = None,
    ) -> RateLimitContext:
        """
        Get a context manager for rate limiting.

        Usage:
            async with rate_limiter.acquire_context(rate_limit, tokens) as retry_after:
                if retry_after is not None:
                    # Wait or handle rate limiting
                    await asyncio.sleep(max(0, retry_after - time.time()))
                else:
                    # Proceed with the actual work
                    result = await do_work()
        """
        return RateLimitContext(self, rate_limit, tokens, reserve, key)


pool = ConnectionPool(
    host=REDIS_CONFIG.host, port=REDIS_CONFIG.port, max_connections=50
)
shared_redis_client = aioredis.Redis(connection_pool=pool)
rate_limiter = DistributedRateLimiter(shared_redis_client)
