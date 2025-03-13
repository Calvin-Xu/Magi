"""
Abstract base class for entity resolvers.
Implements a multi-batch approach based on max_objects_per_batch.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TypeVar, Generic, Tuple

import asyncpg

from magi.embedders.base import EmbeddingProvider
from magi.resolvers.models import ObjectWithEmbedding, MergedEntity

logger = logging.getLogger(__name__)

T = TypeVar("T")


def chunk_dict(input_dict: Dict[str, T], chunk_size: int) -> List[Dict[str, T]]:
    """
    Utility to split a dictionary into sub-dicts of up to chunk_size items each.
    The order is based on the dict's iteration order.
    """
    items = list(input_dict.items())
    chunks = []
    for i in range(0, len(items), chunk_size):
        sub_chunk = items[i : i + chunk_size]
        chunks.append(dict(sub_chunk))
    return chunks


class Resolver(ABC, Generic[T]):
    """
    Abstract base class for resolvers.

    We implement the resolve(...) method to chunk input objects
    into sub-batches (max_objects_per_batch).
    Each sub-batch is processed *independently* (no merging across sub-batches).
    """

    EMBED_PROMPT = "Represent the following object description for retrieval: "

    def __init__(
        self,
        conn: asyncpg.Connection,
        embedding_provider: EmbeddingProvider,
        table_name: str,
        reference_column: str,
        similarity_threshold: float = 0.4,
        max_objects_per_batch: int = 50,
        candidate_epsilon: float = 0.05,
        db_candidate_limit: int = 1,
        max_concurrent_requests: int = 40,
        max_resolve_retries: int = 5,
    ):
        self.conn = conn
        self.embedding_provider = embedding_provider
        self.table_name = table_name
        self.reference_column = reference_column
        self.similarity_threshold = similarity_threshold
        self.candidate_epsilon = candidate_epsilon
        self.db_candidate_limit = db_candidate_limit
        self.max_objects_per_batch = max_objects_per_batch
        self.max_concurrent_requests = max_concurrent_requests
        self.max_resolve_retries = max_resolve_retries

    async def resolve(self, objects_dict: Dict[str, T]) -> Dict[str, T]:
        """
        Public entry point. We chunk the entire input dict into sub-batches of
        at most max_objects_per_batch.

        Each sub-batch is processed in isolation. Then we combine all results.
        """
        if not objects_dict:
            logger.info("No objects to resolve, returning empty dictionary.")
            return {}

        # Break into sub-batches
        batches = chunk_dict(objects_dict, self.max_objects_per_batch)
        logger.info(
            f"[Resolver] Splitting {len(objects_dict)} objects into {len(batches)} batch(es), "
            f"each up to size={self.max_objects_per_batch}."
        )

        final_resolved: Dict[str, T] = {}

        for batch_index, batch_dict in enumerate(batches, start=1):
            logger.info(
                f"[Resolver] Processing batch {batch_index}/{len(batches)} "
                f"with {len(batch_dict)} objects..."
            )

            try:
                resolved_subdict = await self._process_single_batch(batch_dict)
            except Exception as e:
                logger.exception(
                    f"[Resolver] Error processing batch {batch_index}: {e}. "
                    "Falling back to original objects in this batch."
                )
                resolved_subdict = batch_dict

            # Combine into final results
            final_resolved.update(resolved_subdict)

        logger.info(
            f"[Resolver] Done processing all {len(batches)} batch(es). "
            f"Returning {len(final_resolved)} resolved objects."
        )
        return final_resolved

    async def _process_single_batch(self, batch_dict: Dict[str, T]) -> Dict[str, T]:
        """
        Process one batch with up to self.max_resolve_retries attempts at merging and matching.
        Any objects still unresolved after max_retries are fallback-inserted so that all
        eventually have a reference ID.
        """
        unresolved_dict = dict(batch_dict)  # start with everything
        resolved_dict: Dict[str, T] = {}
        attempt = 0

        while attempt < self.max_resolve_retries and unresolved_dict:
            attempt += 1
            logger.info(
                f"[Resolver] Attempt #{attempt} for batch of {len(unresolved_dict)} objects."
            )

            partial_resolved, leftover = await self._process_single_batch_pass(
                unresolved_dict
            )

            # Add the newly resolved to final
            resolved_dict.update(partial_resolved)

            # leftover will be retried
            unresolved_dict = leftover

        # After exhausting attempts, fallback-insert for any leftover
        if unresolved_dict:
            logger.warning(
                f"[Resolver] {len(unresolved_dict)} objects still unresolved after "
                f"{self.max_resolve_retries} attempts. Fallback-inserting."
            )
            for hk, obj in unresolved_dict.items():
                logger.info(
                    f"  Fallback insert for hash_key={hk}, name={getattr(obj, 'name', '')}"
                )
                new_id = await self._safe_insert(self._object_to_model(obj))
                setattr(obj, "postgres_reference", new_id)
                resolved_dict[hk] = obj

        return resolved_dict

    async def _process_single_batch_pass(
        self, batch_dict: Dict[str, T]
    ) -> Tuple[Dict[str, T], Dict[str, T]]:
        """
        Single pipeline pass for the sub-batch:
          1) Convert + compute embeddings
          2) LLM-based merging
          3) Possibly re-embed merges
          4) Match or insert => references assigned
          5) Build partial_resolved (those with reference_id) vs leftover (missing references or missing from LLM)

        Returns:
          (partial_resolved_dict, leftover_dict)
        """
        # 1) Convert + compute embeddings
        object_models = self._convert_and_embed(batch_dict)
        object_models = await self._compute_embeddings(object_models)

        # 2) Merge all in one step
        merged_entities = await self._merge_intra_batch(object_models)

        # 3) Re-embed merges if needed
        merged_entities = await self._compute_embeddings_for_merged(merged_entities)

        # 4) Match or insert => references assigned
        await self._match_or_insert_merged_entities(merged_entities, batch_dict)

        # 5) separate partial resolved from leftover
        partial_resolved: Dict[str, T] = {}
        leftover: Dict[str, T] = {}

        # Gather all hash_keys that the LLM mentioned in merges
        mentioned_keys = set()
        for entity in merged_entities:
            for hk in entity.member_hash_keys:
                mentioned_keys.add(hk)

        for entity in merged_entities:
            # If no reference_id after matching, we consider leftover
            if entity.reference_id is None:
                # leftover
                for hk in entity.member_hash_keys:
                    original_obj = batch_dict[hk]
                    leftover[hk] = original_obj
            else:
                # reference_id found => partial_resolved
                for hk in entity.member_hash_keys:
                    original_obj = batch_dict[hk]
                    updated_obj = self._model_to_object(entity, original_obj)
                    setattr(updated_obj, "postgres_reference", entity.reference_id)
                    partial_resolved[hk] = updated_obj

        # Any objects not mentioned at all by the LLM => leftover
        not_mentioned = set(batch_dict.keys()) - mentioned_keys
        for hk in not_mentioned:
            leftover[hk] = batch_dict[hk]

        return partial_resolved, leftover

    @abstractmethod
    async def _merge_intra_batch(
        self, objects: List[ObjectWithEmbedding]
    ) -> List[MergedEntity]:
        """
        Single pass merging of all objects in the sub-batch. Return the MergedEntities.
        Also includes logging in a table format: each row => {merged entity name, temp_id} => {names + hashes of members}.
        """
        pass

    async def _compute_embeddings(
        self, objects: List[ObjectWithEmbedding]
    ) -> List[ObjectWithEmbedding]:
        """
        Compute embeddings for objects that don't have them.
        """
        need_embedding = [obj for obj in objects if not obj.embedding]
        if not need_embedding:
            return objects

        texts = []
        for obj in need_embedding:
            prompt = f"{obj.name}: {obj.description}" if obj.description else obj.name
            texts.append(prompt)

        embeddings = await self.embedding_provider.embed(
            texts=texts, truncation=True, embed_prompt=self.EMBED_PROMPT
        )

        idx = 0
        for obj in objects:
            if not obj.embedding:
                obj.embedding = embeddings[idx]
                idx += 1

        return objects

    async def _compute_embeddings_for_merged(
        self, merged_entities: List[MergedEntity]
    ) -> List[MergedEntity]:
        """
        Recompute embeddings for merges that lack them after LLM merges.
        """
        need_embedding = [m for m in merged_entities if not m.embedding]
        if not need_embedding:
            return merged_entities

        texts = []
        for m in need_embedding:
            prompt = f"{m.name}: {m.description}" if m.description else m.name
            texts.append(prompt)

        embeddings = await self.embedding_provider.embed(
            texts=texts, truncation=True, embed_prompt=self.EMBED_PROMPT
        )

        idx = 0
        for m in merged_entities:
            if not m.embedding:
                m.embedding = embeddings[idx]
                idx += 1

        return merged_entities

    @abstractmethod
    async def _match_or_insert_merged_entities(
        self, merged_entities: List[MergedEntity], batch_dict: Dict[str, T]
    ):
        """
        Single pass to get DB matches for each merged entity, LLM verification,
        then unify or insert. Must log side-by-side input & output from the LLM.
        """
        pass

    def _convert_and_embed(self, batch_dict: Dict[str, T]) -> List[ObjectWithEmbedding]:
        """
        Convert T -> ObjectWithEmbedding. Synchronous step. We pass to _compute_embeddings later.
        """
        object_models: List[ObjectWithEmbedding] = []
        for hash_key, obj in batch_dict.items():
            model = self._object_to_model(obj)
            model.hash_key = hash_key
            object_models.append(model)
        return object_models

    def _object_to_model(self, obj: T) -> ObjectWithEmbedding:
        reference_id = getattr(obj, "postgres_reference", None)
        embedding = getattr(obj, "embedding", []) or []
        return ObjectWithEmbedding(
            name=getattr(obj, "name", ""),
            description=getattr(obj, "description", ""),
            embedding=embedding,
            reference_id=reference_id,
            hash_key="",  # assigned later
        )

    def _model_to_object(self, entity: MergedEntity, original_obj: T) -> T:
        if hasattr(original_obj, "name"):
            original_obj.name = entity.name
        if hasattr(original_obj, "description"):
            original_obj.description = entity.description
        if hasattr(original_obj, "embedding"):
            original_obj.embedding = entity.embedding
        return original_obj

    # -----------------------------------------------------------
    # DB Insert / Update
    # -----------------------------------------------------------
    async def _safe_insert(self, entity: MergedEntity) -> int:
        """
        Insert with concurrency check.
        """
        maybe_id = await self._find_duplicate_in_db(entity)
        if maybe_id is not None:
            return maybe_id
        return await self._insert_object_into_db(entity)

    @abstractmethod
    async def _find_duplicate_in_db(self, entity: MergedEntity) -> Optional[int]:
        pass

    @abstractmethod
    async def _insert_object_into_db(self, entity: MergedEntity) -> int:
        pass

    @abstractmethod
    async def _update_object_in_db(self, object_id: int, updates: dict) -> None:
        pass

    # -----------------------------------------------------------
    # Utility for DB retrieval
    # -----------------------------------------------------------

    @abstractmethod
    async def find_similar_by_embeddings_batch(
        self,
        conn,
        table_name: str,
        query_embeddings: List[List[float]],
        threshold: float,
        limit_per_query: int = 1,
    ) -> List[List[dict]]:
        pass

    @abstractmethod
    async def close(self):
        pass
