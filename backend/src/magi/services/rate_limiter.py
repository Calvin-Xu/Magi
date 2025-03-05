"""Rate limiting utilities using Redis."""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import redis.asyncio as redis

from magi.config import REDIS_CONFIG


class RateLimit:
    """Configuration for a rate limit."""

    def __init__(
        self,
        name: str,
        rpm: int,
        tpm: int,
        window_size: int = 60,
        num_shards: int = 10,
        max_concurrent: int = 5,
    ):
        """Initialize rate limit config."""
        self.name = name
        self.rpm = rpm
        self.tpm = tpm
        self.window_size = window_size
        self.num_shards = num_shards
        self.max_concurrent = max_concurrent


class DistributedRateLimiter:
    """Token bucket rate limiter using Redis for distributed coordination."""

    def __init__(
        self,
        jitter_factor: float = 0.1,  # 10% jitter by default
    ):
        """Initialize rate limiter."""
        self.jitter_factor = jitter_factor
        self.redis = redis.from_url(
            f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}"
        )
        self._locks: Dict[str, asyncio.Lock] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}

    def _get_or_create_lock(self, rate_limit: RateLimit) -> asyncio.Lock:
        """Get or create lock for a rate limit."""
        if rate_limit.name not in self._locks:
            self._locks[rate_limit.name] = asyncio.Lock()
        return self._locks[rate_limit.name]

    def _get_or_create_semaphore(self, rate_limit: RateLimit) -> asyncio.Semaphore:
        """Get or create semaphore for a rate limit."""
        if rate_limit.name not in self._semaphores:
            self._semaphores[rate_limit.name] = asyncio.Semaphore(
                rate_limit.max_concurrent
            )
        return self._semaphores[rate_limit.name]

    def _get_shard_key(self, rate_limit: RateLimit, shard: int) -> str:
        """Get Redis key for a shard."""
        return f"{rate_limit.name}:shard:{shard}"

    def _get_reservation_key(self, rate_limit: RateLimit, shard: int) -> str:
        """Get Redis key for reservations."""
        return f"{rate_limit.name}:reservations:shard:{shard}"

    def _add_jitter(self, retry_after: float, initial: bool = False) -> float:
        """Add randomized jitter to retry time."""
        if initial:
            # Add more jitter for initial attempts to better spread requests
            base_jitter = 30  # 30 seconds base jitter
            return retry_after + (random.random() * base_jitter)

        # Normal jitter for retries
        max_jitter = 6  # 6 seconds max jitter
        return retry_after + (random.random() * max_jitter)

    async def _get_shard(self, rate_limit: RateLimit, key: str) -> int:
        """Use power of two choices for better load balancing."""
        # Pick two random shards
        shard1 = hash(f"{key}:1") % rate_limit.num_shards
        shard2 = hash(f"{key}:2") % rate_limit.num_shards

        # Get usage from both shards
        async with self.redis.pipeline() as pipe:
            pipe.zcard(self._get_shard_key(rate_limit, shard1))
            pipe.zcard(self._get_shard_key(rate_limit, shard2))
            counts = await pipe.execute()

        # Return shard with lower usage
        return shard1 if counts[0] <= counts[1] else shard2

    async def _get_usage(self, rate_limit: RateLimit, now: datetime) -> Tuple[int, int]:
        """Get current RPM and TPM usage across all shards."""
        cutoff = (now - timedelta(seconds=rate_limit.window_size)).timestamp()

        total_rpm = 0
        total_tpm = 0
        reserved_rpm = 0
        reserved_tpm = 0

        async with self.redis.pipeline() as pipe:
            for shard in range(rate_limit.num_shards):
                key = self._get_shard_key(rate_limit, shard)
                res_key = self._get_reservation_key(rate_limit, shard)

                # Clean up old entries
                pipe.zremrangebyscore(key, "-inf", cutoff)
                pipe.zremrangebyscore(res_key, "-inf", now.timestamp())

                # Get current usage and reservations
                pipe.zrange(key, 0, -1, withscores=True)
                pipe.zrange(res_key, 0, -1, withscores=True)

            results = await pipe.execute()

            for i in range(2, len(results), 4):
                requests = results[i]
                total_rpm += len(requests)
                total_tpm += sum(json.loads(req[0])["tokens"] for req in requests)

                reservations = results[i + 1]
                reserved_rpm += len(reservations)
                reserved_tpm += sum(
                    json.loads(res[0])["tokens"] for res in reservations
                )

        return total_rpm + reserved_rpm, total_tpm + reserved_tpm

    async def acquire(
        self,
        rate_limit: RateLimit,
        tokens: int,
        reserve: bool = True,
        key: Optional[str] = None,
    ) -> Optional[float]:
        """Acquire rate limit approval with concurrency control."""
        semaphore = self._get_or_create_semaphore(rate_limit)
        lock = self._get_or_create_lock(rate_limit)

        async with semaphore:
            now = datetime.now()

            async with lock:
                total_rpm, total_tpm = await self._get_usage(rate_limit, now)

                if total_rpm >= rate_limit.rpm or total_tpm + tokens > rate_limit.tpm:
                    if not reserve:
                        retry_after = (
                            now.timestamp() + rate_limit.window_size / rate_limit.rpm
                        )
                        return self._add_jitter(retry_after, initial=True)

                    next_slot = now.timestamp() + (
                        rate_limit.window_size * (total_tpm + tokens) / rate_limit.tpm
                    )
                    next_slot = self._add_jitter(next_slot)

                    shard = (
                        await self._get_shard(rate_limit, key)
                        if key
                        else random.randrange(rate_limit.num_shards)
                    )
                    await self.redis.zadd(
                        self._get_reservation_key(rate_limit, shard),
                        {
                            json.dumps({"tokens": tokens}): next_slot,
                        },
                    )
                    return next_slot

                shard = (
                    await self._get_shard(rate_limit, key)
                    if key
                    else random.randrange(rate_limit.num_shards)
                )
                await self.redis.zadd(
                    self._get_shard_key(rate_limit, shard),
                    {
                        json.dumps({"tokens": tokens}): now.timestamp(),
                    },
                )
                return None

    async def close(self):
        """Close Redis connection."""
        await self.redis.close()
