import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import asyncpg
from magi.augmenters.base import GraphAugmenter
from magi.augmenters.models import (
    ResearchResponse,
    SchemaContext,
    SchemaEntityInfo,
)
from magi.augmenters.prompts import (
    research_system_prompt,
    research_user_prompt,
)
from magi.config import PERPLEXITY_CONFIG
from magi.services.models import ExtractedRelationship
from magi.services.openai import count_tokens
from magi.services.rate_limiter import rate_limiter, PERPLEXITY_RATE_LIMIT
from magi.utils import get_logger

logger = get_logger(__name__)


class PerplexityAugmenter(GraphAugmenter):
    """GraphAugmenter implementation using Perplexity API for deep research.

    This augmenter uses Perplexity's API to conduct domain-specific research
    based on the dataset schema, discovering new entities and relationships
    to create a hybrid schema-knowledge graph.
    """

    def __init__(
        self,
        api_key: str = PERPLEXITY_CONFIG.api_key,
        model: str = "sonar-pro",
    ):
        """Initialize the Perplexity augmenter.

        Args:
            api_key: Perplexity API key
            model: Perplexity model to use
        """
        self.api_key = api_key
        self.model = model
        self.rate_limiter = rate_limiter
        self.rate_limit = PERPLEXITY_RATE_LIMIT

    async def create_context(self, conn: asyncpg.Connection) -> str:
        """Create a context string by analyzing the schema graph.

        Queries the database for entities and relationships created from
        imported schema and formats them into a structured context string.

        Args:
            conn: Database connection to fetch schema entities and relationships

        Returns:
            A formatted string describing the schema for use as context
        """
        # Create schema context object
        schema_context = SchemaContext()

        # Fetch entities with from_imported_schema=True
        query = "SELECT * FROM entities WHERE from_imported_schema = true"
        entities_rows = await conn.fetch(query)
        entities = []

        # Convert rows to Entity objects
        for row in entities_rows:
            entity = {
                "name": row["name"],
                "description": row["description"],
                "postgres_reference": row["id"],
            }
            entities.append(entity)

        # Separate tables and properties
        tables = {}
        properties = {}

        for entity in entities:
            # Properties have names in format "table.property"
            if "." in entity["name"]:
                properties[entity["name"]] = entity

                # Extract parent table
                parent_table = entity["name"].split(".")[0]
                entity_info = SchemaEntityInfo(
                    name=entity["name"],
                    description=entity["description"],
                    is_property=True,
                    parent_table=parent_table,
                )
                schema_context.properties.append(entity_info)
            else:
                tables[entity["name"]] = entity
                entity_info = SchemaEntityInfo(
                    name=entity["name"],
                    description=entity["description"],
                    is_property=False,
                )
                schema_context.tables.append(entity_info)

        # Fetch relationships with from_imported_schema=True
        rels_query = """
            SELECT r.*, rt.name as relationship_type_name 
            FROM relationships r
            JOIN relationship_types rt ON r.relationship_type = rt.id
            WHERE r.from_imported_schema = true
        """
        relationship_rows = await conn.fetch(rels_query)

        # Format relationships for context
        for row in relationship_rows:
            # Get entity names
            from_entity_name = next(
                (
                    e["name"]
                    for e in entities
                    if e["postgres_reference"] == row["from_entity"]
                ),
                "Unknown",
            )
            to_entity_name = next(
                (
                    e["name"]
                    for e in entities
                    if e["postgres_reference"] == row["to_entity"]
                ),
                "Unknown",
            )

            # Add formatted relationship to context
            formatted_rel = f"{from_entity_name} -> {row['relationship_type_name']} -> {to_entity_name}"
            schema_context.relationships.append(formatted_rel)

        # Generate the context string
        return schema_context.format_context()

    async def _call_perplexity_api(
        self,
        prompt: str,
        context: str,
    ) -> Dict[str, Any]:
        """Call the Perplexity API with rate limiting.

        Args:
            prompt: User instruction for research focus
            context: Schema context to guide research

        Returns:
            Raw API response

        Raises:
            Exception: If the API call fails
        """
        api_url = "https://api.perplexity.ai/chat/completions"

        # Prepare the messages for the API
        messages = [
            {"role": "system", "content": research_system_prompt()},
            {"role": "user", "content": research_user_prompt(context, prompt)},
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": ResearchResponse.model_json_schema()},
            },
            # Additional parameters for better research quality
            "max_tokens": 4000,
            "top_p": 0.9,
            "frequency_penalty": 1.0,
            "presence_penalty": 0.2,
            # Enable web search with high context for thorough research
            "web_search_options": {"search_context_size": "high"},
        }

        # Calculate token usage using tiktoken
        tokens_needed = sum(
            count_tokens(m.get("content", ""), "gpt-4o") for m in messages
        )

        # Use rate limiter to manage API calls
        async with self.rate_limiter.acquire_context(
            rate_limit=self.rate_limit,
            tokens=int(tokens_needed),
            reserve=True,
        ) as retry_after:
            if retry_after:
                # In case we need to wait until a specific time
                wait_seconds = max(0.0, retry_after - datetime.now().timestamp())
                logger.debug(f"Rate limited, waiting for {wait_seconds:.2f} seconds")
                await asyncio.sleep(wait_seconds)
                # Recursive call to try again after waiting
                return await self._call_perplexity_api(api_url, messages)

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        api_url, headers=headers, json=payload
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(
                                f"Perplexity API error: {response.status} - {error_text}"
                            )
                            raise Exception(
                                f"Perplexity API call failed: {response.status}"
                            )

                        result = await response.json()
                        return result
            except Exception as e:
                logger.error(f"Error calling Perplexity API: {str(e)}")
                raise

    async def _parse_research_response(
        self, api_response: Dict[str, Any]
    ) -> ResearchResponse:
        """Parse the raw API response into a structured format.

        Args:
            api_response: Raw response from the Perplexity API

        Returns:
            Structured research response

        Raises:
            ValueError: If the response cannot be parsed
        """
        try:
            # Extract the content from the API response
            content = api_response["choices"][0]["message"]["content"]
            
            # Log a sample of the content for debugging
            content_preview = content[:500] + "..." if len(content) > 500 else content
            logger.debug(f"API response content preview: {content_preview}")
            
            try:
                # First try standard JSON parsing
                parsed_content = json.loads(content)
            except json.JSONDecodeError as json_err:
                # Log the specific error and attempt recovery
                logger.warning(f"JSON parse error at position {json_err.pos}: {json_err.msg}")
                
                # Try to recover from common JSON issues
                try:
                    # If we have partial JSON, try to fix it
                    if '"relationships":' in content:
                        # Find where the relationships array is and take valid part
                        start_idx = content.find('{"relationships":')
                        if start_idx >= 0:
                            # Try to find matching closing brace
                            content = self._find_balanced_json(content[start_idx:])
                            logger.info(f"Attempted to recover JSON structure, length: {len(content)}")
                            parsed_content = json.loads(content)
                        else:
                            raise ValueError("Could not locate relationships in malformed JSON")
                    else:
                        raise ValueError("Could not recover JSON structure")
                except Exception as recovery_err:
                    logger.error(f"Recovery attempt failed: {str(recovery_err)}")
                    # Reraise the original error
                    raise json_err

            # Create a ResearchResponse object
            return ResearchResponse(**parsed_content)
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse Perplexity API response: {str(e)}")
            content_snippet = api_response.get("choices", [{}])[0].get("message", {}).get("content", "")[:100]
            logger.error(f"Content snippet: {content_snippet}...")
            raise ValueError(f"Failed to parse API response: {str(e)}")
    
    def _find_balanced_json(self, content: str) -> str:
        """Attempt to extract a balanced JSON object from potentially malformed content.
        
        Args:
            content: String that may contain a JSON object
            
        Returns:
            Extracted balanced JSON string
        """
        # Basic approach: count opening and closing braces
        stack = []
        for i, char in enumerate(content):
            if char == '{':
                stack.append(i)
            elif char == '}':
                if stack:
                    stack.pop()
                    # If we've closed all open braces, we might have a complete object
                    if not stack:
                        return content[:i+1]
        
        # If we couldn't find balance, return the original content 
        # (other error handling will catch this)
        return content

    async def get_augmented_relationships(
        self,
        context: Optional[str] = None,
        user_instruction: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, List[ExtractedRelationship]]:
        """Returns new relationships discovered through research.

        This method performs domain-specific research based on the schema context
        to discover new entities and relationships.

        Args:
            context: Schema context string created by create_context
            user_instruction: Optional user guidance for research focus

        Returns:
            List of new relationships discovered through research
        """

        if not context:
            context = await self.create_context(None)

        # Default research instruction if none provided
        default_instruction = (
            "Conduct research to identify relationships between the entities in this dataset. "
            "Focus on causal relationships where possible."
        )

        # Use user instruction if provided, otherwise use default
        instruction = user_instruction or default_instruction

        # Call the Perplexity API
        logger.info("Calling Perplexity API for domain research")
        api_response = await self._call_perplexity_api(instruction, context)

        # Parse the response
        research_response = await self._parse_research_response(api_response)

        logger.info(f"Research response: {research_response}")

        # Convert to ExtractedRelationship objects
        relationships = []
        for item in research_response.relationships:
            relationship = ExtractedRelationship(
                from_entity=item.subject,
                from_entity_description=item.subject_description,
                to_entity=item.object,
                to_entity_description=item.object_description,
                relationship_type=item.predicate,
                relationship_description=item.predicate_description,
                constraint_condition=item.constraint_condition,
                reason=item.reason,
                is_causal=item.is_causal,
                source_uri=item.source_uri,
                confidence=item.confidence,
            )
            relationships.append(relationship)

        return research_response.summary, relationships
