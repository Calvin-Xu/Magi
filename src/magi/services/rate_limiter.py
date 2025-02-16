"""Rate limiting utilities using Redis."""

import asyncio
from datetime import datetime, timedelta
import json
from typing import Optional, Tuple
import redis.asyncio as redis
from ..config import REDIS_CONFIG


class DistributedRateLimiter:
    """Token bucket rate limiter using Redis for distributed coordination."""

    def __init__(
        self,
        window_size: int = 60,
        num_shards: int = 10,
    ):
        """Initialize rate limiter."""
        self.window_size = window_size
        self.num_shards = num_shards
        self.redis = redis.from_url(
            f"redis://{REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}"
        )
        self._lock = asyncio.Lock()

    async def _get_usage(self, now: datetime) -> Tuple[int, int]:
        """Get current RPM and TPM usage across all shards."""
        cutoff = (now - timedelta(seconds=self.window_size)).timestamp()

        total_rpm = 0
        total_tpm = 0

        # Clean up and get usage from all shards atomically
        async with self.redis.pipeline() as pipe:
            # Queue cleanup and range queries for all shards
            for shard in range(self.num_shards):
                key = f"gemini_requests:shard:{shard}"
                pipe.zremrangebyscore(key, "-inf", cutoff)
                pipe.zrange(key, 0, -1, withscores=True)

            # Execute all commands
            results = await pipe.execute()

            # Process results (every other result is a zrange result)
            for i in range(1, len(results), 2):
                requests = results[i]  # List of (data, timestamp) tuples
                total_rpm += len(requests)
                total_tpm += sum(json.loads(req[0])["tokens"] for req in requests)

        return total_rpm, total_tpm

    async def acquire(
        self,
        tokens: int,
        rpm_limit: int,
        tpm_limit: int,
        reserve: bool = False,
    ) -> Optional[float]:
        """Wait until rate limits allow the request."""
        async with self._lock:
            now = datetime.now()

            # Check global usage first
            total_rpm, total_tpm = await self._get_usage(now)

            # If we're already over global limit, fail fast
            if total_rpm >= rpm_limit or total_tpm + tokens > tpm_limit:
                if not reserve:
                    # Calculate retry time with jitter
                    base_wait = self.window_size / rpm_limit
                    jitter = base_wait * 0.1  # 10% jitter
                    wait_time = (
                        base_wait + (hash(str(now.timestamp())) % 100) * jitter / 100
                    )
                    return now.timestamp() + wait_time

                # If reserving, calculate when tokens will be available
                tokens_needed = max(0, total_tpm + tokens - tpm_limit)
                return now.timestamp() + (
                    tokens_needed / (tpm_limit / self.window_size)
                )

            # Pick a shard using power of two choices
            shard1 = hash(str(now.timestamp())) % self.num_shards
            shard2 = hash(str(now.timestamp() + 1)) % self.num_shards

            # Get usage from both shards
            usage1 = len(
                await self.redis.zrange(f"gemini_requests:shard:{shard1}", 0, -1)
            )
            usage2 = len(
                await self.redis.zrange(f"gemini_requests:shard:{shard2}", 0, -1)
            )

            # Use the less loaded shard
            shard = shard1 if usage1 <= usage2 else shard2

            # Record the request
            await self.redis.zadd(
                f"gemini_requests:shard:{shard}",
                {
                    json.dumps({"tokens": tokens}): now.timestamp(),
                },
            )
            return None

    async def close(self):
        """Close Redis connection."""
        await self.redis.close()
