"""
Concrete OpenAI-based resolver, chunking by max_objects_per_batch.
"""

import asyncio
from datetime import datetime
from typing import Literal, Dict, List, Optional, Tuple

import tiktoken
from openai import OpenAI

from magi.config import OPENAI_CONFIG
from magi.resolvers.base import MergedEntity, ObjectWithEmbedding, Resolver, T
from magi.resolvers.models import (
    LLMIntraBatchMergeResponse,
    VerificationBatchResponse,
)
from magi.services.rate_limiter import O3_MINI_RATE_LIMIT, rate_limiter
from magi.utils import get_logger
from magi.services import db_operations

logger = get_logger(__name__)


class OpenAIResolver(Resolver[T]):
    def __init__(
        self,
        conn,
        embedding_provider,
        table_name: Literal["entities", "relationship_types"],
        reference_column: str = "id",
        similarity_threshold: float = 0.4,
        max_objects_per_batch: int = 50,
        model: str = "o3-mini-2025-01-31",
        api_key: str = OPENAI_CONFIG.api_key,
        max_retries: int = 5,
        **kwargs,
    ):
        super().__init__(
            conn,
            embedding_provider,
            table_name,
            reference_column,
            similarity_threshold,
            max_objects_per_batch,
            **kwargs,
        )
        self.model = model
        self.max_retries = max_retries

        # Initialize OpenAI
        self.client = OpenAI(api_key=api_key)

        # We use a token encoder for possible length checks or logging
        self.tokenizer = (
            tiktoken.encoding_for_model(model)
            if model.startswith("gpt-")
            else tiktoken.get_encoding("cl100k_base")
        )
        self.reserved_tokens = 500

        # Rate limiter configuration
        self._rate_limiter = rate_limiter
        self._rate_limit = O3_MINI_RATE_LIMIT

    # --------------------------------------------------------------
    # 1) _merge_intra_batch
    # --------------------------------------------------------------
    async def _merge_intra_batch(
        self, objects: List[ObjectWithEmbedding]
    ) -> List[MergedEntity]:
        """
        Single LLM call for the entire sub-batch. Then log a table of merges.
        Using structured outputs => LLMIntraBatchMergeResponse
        """
        if not objects:
            return []

        # Build ID mapping to hide the hash_key from the LLM
        idx_to_hash = {i: obj.hash_key for i, obj in enumerate(objects)}
        prompt_text = self._build_intra_batch_merge_prompt(objects)

        # Attempt the call up to self.max_retries times if we get an error or refusal
        merged_entities: List[MergedEntity] = []
        attempt_count = 0
        while attempt_count < self.max_retries:
            attempt_count += 1
            logger.info(
                f"[OpenAIResolver] Attempt {attempt_count} to get merges via structured output..."
            )

            try:
                # Acquire rate limit token
                if not await self._acquire_rate_limit(token_count=1000):
                    continue

                # Do the structured-output call
                completion = await asyncio.to_thread(
                    self.client.beta.chat.completions.parse,
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert AI system in semantic coreference resolution for knowledge graph construction."
                                " Always return valid JSON that conforms to the given schema."
                            ),
                        },
                        {"role": "user", "content": prompt_text},
                    ],
                    response_format=LLMIntraBatchMergeResponse,  # pydantic model
                )

                # Check for refusal
                if completion.choices[0].message.refusal:
                    logger.warning(
                        "[OpenAIResolver._merge_intra_batch] Model refused to answer."
                    )
                    # We can either break or retry
                    continue

                # All good: parse the pydantic object
                parsed_obj: LLMIntraBatchMergeResponse = completion.choices[
                    0
                ].message.parsed
                merged_entities = self._build_merged_entities_from_llm(
                    parsed_obj, objects, idx_to_hash
                )
                break  # success, break out
            except Exception as e:
                logger.warning(
                    f"[OpenAIResolver._merge_intra_batch] Failed attempt {attempt_count}: {e}"
                )
                await asyncio.sleep(2**attempt_count)

        # If we never got a successful parse, fallback => 1-1 merges
        if not merged_entities:
            logger.error(
                "[OpenAIResolver._merge_intra_batch] All attempts failed or refused. Fallback => 1-1 merges."
            )
            merged_entities = [
                MergedEntity(
                    temp_id=obj.hash_key,
                    name=obj.name,
                    description=obj.description,
                    member_hash_keys=[obj.hash_key],
                    embedding=obj.embedding,
                )
                for obj in objects
            ]

        # Log a table: each row => entity => members
        self._log_merged_table(merged_entities, objects)
        return merged_entities

    def _build_intra_batch_merge_prompt(
        self, objects: List[ObjectWithEmbedding]
    ) -> str:
        """
        We pass integer IDs to the LLM, not the actual hash_key. The LLM's output
        must conform to LLMIntraBatchMergeResponse Pydantic schema.
        """
        prompt = (
            "We have a batch of objects, each with an integer ID, name, and description. "
            "Identify duplicates and merge them into a single object. Two objects are duplicates "
            "if they refer to the same entity or concept (aliases, synonyms, etc.).\n\n"
            "Return JSON matching this schema:\n"
            "LLMIntraBatchMergeResponse =>\n"
            "{\n"
            '  "merged_entities": [\n'
            "    {\n"
            '      "merged_id": "string",\n'
            '      "merged_name": "string",\n'
            '      "merged_description": "string",\n'
            '      "member_ids": [0,1]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Constraints:\n"
            "- Each input object must appear exactly once in exactly one merged group.\n"
            "- 'merged_name' is a single canonical name that the object is best known by.\n"
            "- 'merged_description' is the best globally-identifying description, merging info.\n\n"
            "Objects:\n"
        )

        for i, obj in enumerate(objects):
            prompt += f"Object {i}:\n"
            prompt += f"  ID: {i}\n"
            prompt += f"  name: {obj.name}\n"
            prompt += f"  description: {obj.description}\n\n"

        prompt += "Return valid JSON only."
        return prompt

    def _build_merged_entities_from_llm(
        self,
        response_obj: LLMIntraBatchMergeResponse,
        objects: List[ObjectWithEmbedding],
        idx_to_hash: Dict[int, str],
    ) -> List[MergedEntity]:
        """
        Convert the LLM's structured output back to a list of MergedEntity, mapping integer IDs
        to real hash_keys.
        """
        obj_map = {o.hash_key: o for o in objects}
        merged_entities: List[MergedEntity] = []

        for me in response_obj.merged_entities:
            if len(me.member_ids) == 1:
                single_hk = idx_to_hash[me.member_ids[0]]
                source_obj = obj_map.get(single_hk)
                init_emb = source_obj.embedding if source_obj else []
            else:
                init_emb = []

            member_hashes = [
                idx_to_hash[m_id] for m_id in me.member_ids if m_id in idx_to_hash
            ]

            merged_entities.append(
                MergedEntity(
                    temp_id=me.merged_id,
                    name=me.merged_name,
                    description=me.merged_description,
                    member_hash_keys=member_hashes,
                    embedding=init_emb,
                )
            )

        return merged_entities

    def _log_merged_table(
        self, merged_entities: List[MergedEntity], objects: List[ObjectWithEmbedding]
    ):
        """
        Log a table with columns:
        MergedEntityName  temp_id  =>  [ (orig name1, hk1), (orig name2, hk2), ... ]
        """
        obj_map = {o.hash_key: o for o in objects}
        logger.info("[OpenAIResolver._merge_intra_batch] Merged Entities Table:")
        for entity in merged_entities:
            row_left = f"({entity.name[:25]}...), {entity.temp_id}"
            member_texts = []
            for hk in entity.member_hash_keys:
                orig = obj_map.get(hk)
                if orig:
                    member_texts.append(f"({orig.name}, {hk})")
                else:
                    member_texts.append(f"(UNKNOWN, {hk})")
            row_right = "; ".join(member_texts)
            logger.info(f"  {row_left} => {row_right}")

    # --------------------------------------------------------------
    # 2) _match_or_insert_merged_entities
    # --------------------------------------------------------------
    async def _match_or_insert_merged_entities(
        self,
        merged_entities: List[MergedEntity],
        batch_dict: Dict[str, T],
    ):
        """
        For each entity, get top DB candidate => single LLM call verifying all pairs => unify or insert.
        Log side-by-side input & output from the LLM.
        """
        # Step A: see if any references exist from the original objects
        to_resolve = []
        for entity in merged_entities:
            existing_ref = self._any_existing_ref(entity, batch_dict)
            if existing_ref is not None:
                entity.reference_id = existing_ref
                # optional DB update
                await self._update_object_in_db(
                    object_id=existing_ref,
                    updates={"name": entity.name, "description": entity.description},
                )
            else:
                to_resolve.append(entity)

        if not to_resolve:
            return

        # Step B: gather top DB candidates
        all_embeddings = [m.embedding for m in to_resolve]

        # Use a larger batch size for more efficient database queries
        # This significantly reduces the number of database round trips
        batch_size = min(100, len(all_embeddings))  # Use up to 100 embeddings per batch
        logger.info(
            f"[_match_or_insert_merged_entities] Finding similar embeddings for {len(all_embeddings)} entities with batch_size={batch_size}"
        )

        candidates_batch = await self.find_similar_by_embeddings_batch(
            self.conn,
            self.table_name,
            all_embeddings,
            self.similarity_threshold,
            limit_per_query=1,
            batch_size=batch_size,
        )

        pairs: List[Tuple[MergedEntity, Optional[dict]]] = []
        for idx, entity in enumerate(to_resolve):
            cands = candidates_batch[idx]
            if cands:
                pairs.append((entity, cands[0]))
            else:
                pairs.append((entity, None))

        # Step C: single LLM call to verify each pair
        to_insert = await self._verify_db_pairs_single_call(pairs)

        # Step D: for anything the LLM says "not same", insert new
        for entity in to_insert:
            new_id = await self._safe_insert(entity)
            entity.reference_id = new_id

    def _any_existing_ref(
        self, entity: MergedEntity, batch_dict: Dict[str, T]
    ) -> Optional[int]:
        distinct_refs = set()
        for hk in entity.member_hash_keys:
            if hk not in batch_dict:
                continue
            obj = batch_dict[hk]
            ref_id = getattr(obj, "postgres_reference", None)
            if ref_id is not None:
                distinct_refs.add(ref_id)

        if not distinct_refs:
            return None
        if len(distinct_refs) > 1:
            logger.warning(
                f"Multiple DB references in same merged entity: {distinct_refs}"
            )
        return list(distinct_refs)[0]

    async def _verify_db_pairs_single_call(
        self, pairs: List[Tuple[MergedEntity, Optional[dict]]]
    ) -> List[MergedEntity]:
        """
        Single LLM call. For each pair => are_same => unify or insert new.
        Return the list of entities that must be inserted new.
        """
        no_candidate = [(e, None) for (e, c) in pairs if c is None]
        verify_pairs = [(e, c) for (e, c) in pairs if c is not None]

        # immediate insertion for those that have no DB candidate
        to_insert: List[MergedEntity] = []
        for entity, _ in no_candidate:
            to_insert.append(entity)

        if not verify_pairs:
            return to_insert

        # Log LLM input side-by-side
        logger.info("[_verify_db_pairs_single_call] LLM input pairs:")
        for idx, (entity, candidate) in enumerate(verify_pairs):
            logger.info(f" Pair {idx}:")
            logger.info(
                f"   - MergedEntity: name='{entity.name}', desc='{entity.description}'"
            )
            logger.info(
                f"   - DB Candidate: name='{candidate['name']}', desc='{candidate['description']}'"
            )

        prompt_text = self._build_verification_batch_prompt(verify_pairs)

        # Attempt call with structured outputs => VerificationBatchResponse
        attempt_count = 0
        verify_results = []
        while attempt_count < self.max_retries:
            attempt_count += 1
            logger.info(
                f"[_verify_db_pairs_single_call] Attempt #{attempt_count} for verification."
            )

            try:
                if not await self._acquire_rate_limit(token_count=1000):
                    continue

                completion = await asyncio.to_thread(
                    self.client.beta.chat.completions.parse,
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert AI system in semantic coreference resolution. "
                                "Return valid JSON exactly matching VerificationBatchResponse."
                            ),
                        },
                        {"role": "user", "content": prompt_text},
                    ],
                    response_format=VerificationBatchResponse,
                )

                # Check refusal
                if completion.choices[0].message.refusal:
                    logger.warning(
                        "[_verify_db_pairs_single_call] Model refused to answer."
                    )
                    continue

                results_obj = completion.choices[0].message.parsed
                verify_results = results_obj.results
                break
            except Exception as e:
                logger.warning(
                    f"[_verify_db_pairs_single_call] Attempt {attempt_count} failed: {e}"
                )
                await asyncio.sleep(2**attempt_count)

        if not verify_results:
            logger.error(
                "[_verify_db_pairs_single_call] All attempts failed or refused. Defaulting are_same=False."
            )
            # fallback => all false
            verify_results = []
            for i in range(len(verify_pairs)):
                verify_results.append((False, None, None))

            # We'll just manually shape them into a suitable structure
            # to keep code simpler
            from magi.resolvers.models import VerificationResult

            vrs = []
            for i in range(len(verify_pairs)):
                vrs.append(
                    VerificationResult(
                        pair_index=i,
                        are_same=False,
                        updated_name=None,
                        updated_description=None,
                    )
                )
            verify_results = vrs

        # Now apply results
        # results is a list of VerificationResult in the same order as the prompt
        # "pair_index" indexes into verify_pairs
        for vr in verify_results:
            idx = vr.pair_index
            if idx < 0 or idx >= len(verify_pairs):
                continue
            entity, candidate = verify_pairs[idx]
            are_same = vr.are_same
            updated_name = vr.updated_name
            updated_desc = vr.updated_description

            logger.info(
                f" Pair {idx} => are_same={are_same}, updated_name='{updated_name}', "
                f"updated_desc='{(updated_desc[:50] + '...') if updated_desc else None}'"
            )
            if are_same and candidate is not None:
                entity.reference_id = candidate["reference_id"]
                updates = {}
                if updated_name:
                    updates["name"] = updated_name
                    entity.name = updated_name
                if updated_desc:
                    updates["description"] = updated_desc
                    entity.description = updated_desc
                if updates:
                    await self._update_object_in_db(candidate["reference_id"], updates)
            else:
                to_insert.append(entity)

        return to_insert

    def _build_verification_batch_prompt(
        self, pairs: List[Tuple[MergedEntity, dict]]
    ) -> str:
        """
        The LLM must return:
        {
          "results": [
            {
              "pair_index": 0,
              "are_same": true,
              "updated_name": "X",
              "updated_description": "Y"
            },
            ...
          ]
        }
        This matches VerificationBatchResponse exactly.
        """
        prompt = (
            "We have several pairs of (new_merged_entity, existing_db_object). For each pair:\n"
            "- are_same: bool (whether they refer to the same entity)\n"
            "- if are_same=true, updated_name and updated_description can unify or refine the entity.\n\n"
            "Return JSON conforming to VerificationBatchResponse =>\n"
            "{\n"
            '  "results": [\n'
            "    {\n"
            '      "pair_index": 0,\n'
            '      "are_same": true,\n'
            '      "updated_name": "...",\n'
            '      "updated_description": "..." \n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Pairs:\n"
        )

        for idx, (entity, candidate) in enumerate(pairs):
            prompt += f"Pair {idx}:\n"
            prompt += " New Merged Object:\n"
            prompt += f"   name: {entity.name}\n"
            prompt += f"   description: {entity.description}\n\n"
            prompt += " Existing DB Object:\n"
            prompt += f"   name: {candidate['name']}\n"
            prompt += f"   description: {candidate['description']}\n\n"

        prompt += "Return only valid JSON."
        return prompt

    # --------------------------------------------------------------
    # Internal / Helper
    # --------------------------------------------------------------
    async def _acquire_rate_limit(self, token_count: int = 1000):
        """
        Acquire a rate limit token with the specified token count.

        Args:
            token_count: Number of tokens to count against the rate limit
        """
        async with self._rate_limiter.acquire_context(
            rate_limit=self._rate_limit,
            tokens=token_count,
            reserve=True,
        ) as retry_after:
            if retry_after:
                wait_seconds = max(0.0, retry_after - datetime.now().timestamp())
                if wait_seconds > 0:
                    logger.debug(f"Rate-limited. Sleeping {wait_seconds:.2f} seconds.")
                    await asyncio.sleep(wait_seconds)
                    return False  # Indicate we should retry
            return True  # Indicate we can proceed

    # --------------------------------------------------------------
    # DB Insert / Update / Retrieve
    # --------------------------------------------------------------
    async def _find_duplicate_in_db(self, entity: MergedEntity) -> Optional[int]:
        if self.table_name == "entities":
            result = await db_operations.find_entity_by_name(self.conn, entity.name)
            if result:
                return result["id"]
        elif self.table_name == "relationship_types":
            result = await db_operations.find_relationship_type_by_name(
                self.conn, entity.name
            )
            if result:
                return result["id"]
        else:
            raise ValueError(f"Unknown table name: {self.table_name}")

        return None

    async def _insert_object_into_db(self, entity: MergedEntity) -> int:
        # Convert MergedEntity to the appropriate model based on table_name
        if self.table_name == "entities":
            from magi.services.models import Entity

            obj = Entity(
                name=entity.name,
                description=entity.description,
                embedding=entity.embedding,
                from_imported_schema=False,  # Resolver-created objects are not from imported schema
            )

            return await db_operations.insert_entity(self.conn, obj)

        elif self.table_name == "relationship_types":
            from magi.services.models import RelationshipType

            obj = RelationshipType(
                name=entity.name,
                description=entity.description,
                embedding=entity.embedding,
                from_imported_schema=False,  # Resolver-created objects are not from imported schema
            )

            return await db_operations.insert_relationship_type(self.conn, obj)

        else:
            raise ValueError(f"Unknown table name: {self.table_name}")

    async def _update_object_in_db(self, object_id: int, updates: dict) -> None:
        if self.table_name == "entities":
            await db_operations.update_entity(self.conn, object_id, updates)
        elif self.table_name == "relationship_types":
            await db_operations.update_relationship_type(self.conn, object_id, updates)

    async def find_similar_by_embeddings_batch(
        self,
        conn,
        table_name: str,
        query_embeddings: List[List[float]],
        threshold: float,
        limit_per_query: int = 1,
        batch_size: int = 50,
    ) -> List[List[dict]]:
        """
        Find similar embeddings for a batch of query embeddings.

        Args:
            conn: Database connection
            table_name: Table to search in
            query_embeddings: List of embedding vectors to search for
            threshold: Similarity threshold (0-1)
            limit_per_query: Maximum number of results per query embedding
            batch_size: Number of embeddings to process in a single database query

        Returns:
            List of lists of matching records
        """
        return await db_operations.find_similar_by_embeddings_batch(
            conn, table_name, query_embeddings, threshold, limit_per_query, batch_size
        )

    async def close(self):
        """Clean up resources."""
        logger.debug("Closing OpenAIResolver resources")
        await self._rate_limiter.close()
