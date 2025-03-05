"""
OpenAI implementation of the Resolver class for entity and relationship type resolution.
This resolver uses OpenAI's LLM capabilities to verify if objects are the same.
"""

from typing import List, Optional, TypeVar
import asyncio
from pydantic import BaseModel, Field
from openai import OpenAI
import tiktoken

from ..config import OPENAI_CONFIG
from .base import Resolver
from ..services.models import Entity, RelationshipType
from .models import (
    ObjectPair,
    VerificationResult as ModelVerificationResult,
)

T = TypeVar("T", Entity, RelationshipType)


class VerificationResult(BaseModel):
    """Model for verification results from OpenAI."""

    input_index: int = Field(
        ..., description="Index of the input object in the provided list"
    )
    is_same: bool = Field(
        ..., description="Whether the objects refer to the same entity or concept"
    )
    updated_name: Optional[str] = Field(
        None,
        description="Updated name combining information from both objects (if they are the same)",
    )
    updated_description: Optional[str] = Field(
        None,
        description="Updated description combining information from both objects (if they are the same)",
    )


class VerificationResponse(BaseModel):
    """Model for the complete verification response from OpenAI."""

    results: List[VerificationResult] = Field(
        ..., description="List of verification results for each pair of objects"
    )


class OpenAIResolver(Resolver[T]):
    """
    OpenAI implementation of the Resolver class.

    This resolver uses OpenAI's LLM capabilities to verify if objects are the same
    and to generate updated names and descriptions when needed.
    """

    def __init__(
        self,
        conn,
        embedding_provider,
        table_name: str,
        reference_column: str = "id",
        similarity_threshold: float = 0.8,
        max_tokens_per_batch: int = 4000,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        api_key: str = OPENAI_CONFIG["api_key"],
    ):
        """
        Initialize the OpenAI resolver.

        Args:
            conn: asyncpg connection
            embedding_provider: Provider for computing embeddings
            table_name: Name of the table to search in (e.g., 'entities', 'relationship_types')
            reference_column: Name of the column that serves as a reference (e.g., 'id')
            similarity_threshold: Threshold for considering objects as similar (0-1)
            max_tokens_per_batch: Maximum number of tokens per LLM batch
            model: OpenAI model to use
            temperature: Temperature for LLM generation (0-1)
            api_key: OpenAI API key
        """
        super().__init__(
            conn,
            embedding_provider,
            table_name,
            reference_column,
            similarity_threshold,
            max_tokens_per_batch,
        )
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key)
        # Initialize the tokenizer for the specified model
        self.tokenizer = (
            tiktoken.encoding_for_model(model)
            if model.startswith("gpt")
            else tiktoken.get_encoding("cl100k_base")
        )
        # Reserve tokens for system message, response format, and some overhead
        self.reserved_tokens = 500

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    async def _create_verification_batches(
        self, object_pairs: List[ObjectPair]
    ) -> List[List[ObjectPair]]:
        """
        Group object pairs into batches to avoid token limit.

        Args:
            object_pairs: List of ObjectPair instances

        Returns:
            List of batches, where each batch is a list of object pairs
        """
        batches = []
        current_batch = []
        current_token_count = 0

        # Create a sample prompt to estimate the base token count (instructions, etc.)
        base_prompt = self._create_verification_prompt([])
        base_token_count = self._count_tokens(base_prompt)

        # Available tokens for actual object pairs
        available_tokens = (
            self.max_tokens_per_batch - self.reserved_tokens - base_token_count
        )

        for pair in object_pairs:
            # Skip pairs where no similar object was found
            if pair.similar_object is None:
                current_batch.append(pair)
                continue

            # Create a sample prompt with just this pair to count tokens
            sample_pair_prompt = self._create_verification_prompt([pair])
            pair_token_count = self._count_tokens(sample_pair_prompt) - base_token_count

            # If adding this pair would exceed the token limit, start a new batch
            if (
                current_token_count + pair_token_count > available_tokens
                and current_batch
            ):
                batches.append(current_batch)
                current_batch = [pair]
                current_token_count = pair_token_count
            else:
                current_batch.append(pair)
                current_token_count += pair_token_count

        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)

        return batches

    async def _verify_objects_batch(
        self, batch: List[ObjectPair]
    ) -> List[ModelVerificationResult]:
        """
        Verify if the retrieved objects are the same as the input objects using OpenAI.

        Args:
            batch: List of ObjectPair instances

        Returns:
            List of verification results
        """
        # Filter out pairs where no similar object was found
        pairs_to_verify = [pair for pair in batch if pair.similar_object is not None]

        # If there are no pairs to verify, return results immediately
        if not pairs_to_verify:
            return [
                ModelVerificationResult(
                    input_object=pair.input_object,
                    db_object=None,
                    is_same=False,
                    is_from_input_list=pair.is_from_input_list,
                )
                for pair in batch
            ]

        # Prepare the prompt for the LLM
        prompt = self._create_verification_prompt(pairs_to_verify)

        # Count tokens to ensure we're within limits
        token_count = self._count_tokens(prompt)
        if token_count + self.reserved_tokens > self.max_tokens_per_batch:
            print(
                f"Warning: Prompt token count ({token_count}) is close to or exceeds the limit. Consider reducing batch size."
            )

        try:
            # Call the OpenAI API with structured output
            response = await asyncio.to_thread(
                self.client.beta.chat.completions.parse,
                model=self.model,
                temperature=self.temperature,
                response_format=VerificationResponse,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that analyzes objects to determine if they are the same entity or concept.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            # Process the results
            verification_results = response.choices[0].message.parsed
            processed_results = []

            # Keep track of which input objects have been processed
            processed_input_objects = set()

            # Process verified pairs
            for result in verification_results.results:
                input_idx = result.input_index
                if 0 <= input_idx < len(pairs_to_verify):
                    pair = pairs_to_verify[input_idx]
                    processed_input_objects.add(id(pair.input_object))

                    processed_results.append(
                        ModelVerificationResult(
                            input_object=pair.input_object,
                            db_object=pair.similar_object,
                            is_same=result.is_same,
                            updated_name=result.updated_name,
                            updated_description=result.updated_description,
                            is_from_input_list=pair.is_from_input_list,
                        )
                    )

            # Add results for input objects that were not processed
            for pair in batch:
                if (
                    pair.similar_object is None
                    or id(pair.input_object) not in processed_input_objects
                ):
                    processed_results.append(
                        ModelVerificationResult(
                            input_object=pair.input_object,
                            db_object=pair.similar_object,
                            is_same=False,
                            is_from_input_list=pair.is_from_input_list,
                        )
                    )

            return processed_results
        except Exception as e:
            # Handle errors gracefully
            print(f"Error processing LLM response: {e}")

            # Return a default response indicating objects are different
            return [
                ModelVerificationResult(
                    input_object=pair.input_object,
                    db_object=pair.similar_object,
                    is_same=False,
                    is_from_input_list=pair.is_from_input_list,
                )
                for pair in batch
            ]

    def _create_verification_prompt(self, pairs: List[ObjectPair]) -> str:
        """
        Create a prompt for the LLM to verify if objects are the same.

        Args:
            pairs: List of ObjectPair instances

        Returns:
            Prompt string for the LLM
        """
        prompt = """
I need you to analyze pairs of objects and determine if they refer to the same entity or relationship type.
For each pair, determine:
1. If they are the same entity/relationship type (is_same: true/false)
2. If they are the same, provide an updated name and description that combines the best information from both

Here are the pairs to analyze:

"""

        for i, pair in enumerate(pairs):
            input_obj = pair.input_object
            db_obj = pair.similar_object

            if db_obj is None:
                continue

            prompt += f"\nPair {i}:\n"
            prompt += f"Input Object:\n- Name: {input_obj.name}\n- Description: {input_obj.description}\n"
            prompt += f"Retrieved Object:\n- Name: {db_obj.name}\n- Description: {db_obj.description}\n"

        prompt += "\nPlease analyze each pair and determine if they represent the same entity or concept."

        return prompt
