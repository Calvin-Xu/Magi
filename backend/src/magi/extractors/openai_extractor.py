"""OpenAI-based relationship extractor."""

from datetime import datetime
from typing import List

from magi.config import OPENAI_CONFIG
from magi.extractors.base import (
    ExtractionMetrics,
    RelationshipExtractor,
)
from magi.services.models import ExtractedRelationship
from magi.services.openai import (
    call_openai_with_backoff,
    count_tokens,
    create_openai_client,
    get_model_limits,
    get_tokenizer,
)
from magi.services.rate_limiter import RateLimit, rate_limiter
from magi.utils import get_logger
from pydantic import BaseModel

logger = get_logger(__name__)


class RelationshipItem(BaseModel):
    subject: str
    subject_description: str
    object: str
    object_description: str
    predicate: str
    predicate_description: str
    constraint_condition: str
    reason: str
    is_causal: bool


class RelationshipList(BaseModel):
    """Schema for the list of extracted relationships."""

    relationships: List[RelationshipItem]


class OpenAIError(Exception):
    """Base error for OpenAI API calls."""

    pass


class OpenAIExtractor(RelationshipExtractor):
    """Relationship extractor using OpenAI's ChatCompletion API and JSON Schema."""

    def __init__(
        self,
        model: str,
        openai_api_key: str = OPENAI_CONFIG.api_key,
        max_concurrent_requests: int = 10000,
        max_retries: int = 5,
    ):
        """
        Initialize the OpenAI-based extractor.

        Args:
            model: Model name (e.g. "gpt-4o-2024-08-06") that supports JSON schema.
            openai_api_key: Your OpenAI API key.
            max_concurrent_requests: Upper bound on concurrency for this class (extra layer).
            max_retries: Number of times to retry on transient errors.
        """

        # Validate model is supported in our service
        try:
            limits = get_model_limits(model)
        except KeyError:
            logger.error(f"Unsupported model or missing config: {model}")
            raise ValueError(f"Unsupported model or missing config: {model}")

        super().__init__(
            model=model,
            max_input_tokens=limits.input_token_limit,
            max_concurrent_requests=max_concurrent_requests,
        )

        # For structured outputs
        self._client = create_openai_client(openai_api_key)

        # A simple distributed rate limiter from your codebase
        self._rate_limiter = rate_limiter
        self._rate_limit = RateLimit(
            name=model,
            rpm=limits.rpm,
            tpm=limits.tpm,
            window_size=60,
            num_shards=10,
            max_concurrent=limits.max_concurrent,
        )

        self.max_retries = max_retries

        # Use the shared tokenizer
        self._tokenizer = get_tokenizer(model)

        logger.info(
            f"Initialized OpenAIExtractor with model: {model}, max_concurrent: {max_concurrent_requests}"
        )

    async def close(self):
        """Clean up any async resources (e.g. distributed rate limiter)."""
        pass

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the shared token counter."""
        return count_tokens(text, self.model)

    def _count_tokens_sync(self, text: str) -> int:
        """Synchronously count tokens in `text` using tiktoken."""
        return len(self._tokenizer.encode(text))

    async def _extract_relationships_raw(
        self,
        text: str,
        **kwargs,
    ) -> List[ExtractedRelationship]:
        """Extract relationships from a single chunk of text."""
        if not text.strip():
            logger.warning("Empty text chunk, skipping extraction")
            return []

        start_time = datetime.now()
        text_hash = hash(text)

        # Construct the prompt
        # Ideally, you'd have a templating system here, but for simplicity...
        prompt = f"""Extract all entity relationships from the following text and explain your reasoning. Focus on causes, effects, associations, and connections.

TEXT:
{text}

Your task is to identify relationships where one entity relates to another entity in a specific way. For each relationship, provide:
1. The source entity (subject)
2. A clear, unique description of the source entity
3. The target entity (object)
4. A clear, unique description of the target entity
5. The relationship type between them
6. A clear description of the relationship type
7. Any constraint or condition under which the relationship holds (if applicable)
8. Your reasoning for identifying this relationship
9. Whether the relationship is causal (true if one entity causes an effect on the other)

Examples of relationships: 
- X increases Y
- A is part of B
- C inhibits D
- E is associated with F

Provide comprehensive, unique descriptions for each entity and relationship type that would distinguish them from similar entities or relationships.
"""

        try:
            # Prepare message for OpenAI API
            messages = [{"role": "user", "content": prompt}]

            # Call OpenAI API with backoff and rate limiting using the shared service
            response = await call_openai_with_backoff(
                client=self._client,
                model=self.model,
                messages=messages,
                response_format=RelationshipList,
                rate_limit=self._rate_limit,
                max_retries=self.max_retries,
            )

            if not response:
                logger.warning(f"No valid response for text hash: {text_hash}")
                return []

            # Extract the parsed result from the response
            # which should be a RelationshipList pydantic object, or a refusal.
            result = response.choices[0].message

            if result.refusal:
                # The model refused for safety reasons or other policy
                logger.warning(
                    f"Model refused to answer for text hash: {text_hash}. Reason: {result.refusal}"
                )
                return []

            if not result.parsed:
                logger.warning(
                    f"No parsed object from the model for text hash: {text_hash}"
                )
                return []

            # Everything looks good. `parsed` is our RelationshipList
            extracted_list: RelationshipList = result.parsed
            logger.info(
                f"Successfully extracted {len(extracted_list.relationships)} relationships from text hash: {text_hash}"
            )

            # Record usage metrics
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            # The usage object has token counts for prompt & completion
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            logger.debug(
                f"Extraction metrics - Duration: {duration_ms:.2f}ms, "
                f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}"
            )

            self._metrics.append(
                ExtractionMetrics(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    duration_ms=duration_ms,
                    timestamp=start_time,
                )
            )

            # Convert RelationshipItem -> ExtractedRelationship
            relationships_out: List[ExtractedRelationship] = []
            for item in extracted_list.relationships:
                relationships_out.append(
                    ExtractedRelationship(
                        from_entity=item.subject,
                        from_entity_description=item.subject_description,
                        to_entity=item.object,
                        to_entity_description=item.object_description,
                        relationship_type=item.predicate,
                        relationship_description=item.predicate_description,
                        constraint_condition=(
                            ""
                            if item.constraint_condition.lower() == "none"
                            else item.constraint_condition
                        ),
                        reason=item.reason,
                        is_causal=item.is_causal,
                    )
                )

            return relationships_out

        except Exception as e:
            logger.exception(
                f"Error extracting relationships for text hash {text_hash}: {str(e)}"
            )
            return []
