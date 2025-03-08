"""
Abstract base class for entity resolvers.
Implements a multi-batch approach based on max_objects_per_batch, with no sub-batching inside each batch.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TypeVar, Generic

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

    async def resolve(self, objects_dict: Dict[str, T]) -> Dict[str, T]:
        """
        Public entry point. We chunk the entire input dict into sub-batches of at most max_objects_per_batch.

        Each sub-batch is processed fully (merge, match, insert) in isolation.
        Then we combine all results into one dictionary (but no cross-batch merges).

        If you want a single giant batch, set max_objects_per_batch >= len(objects_dict).
        """
        if not objects_dict:
            logger.info("No objects to resolve, returning empty dictionary")
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
                f"[Resolver] Processing batch {batch_index}/{len(batches)} with {len(batch_dict)} objects..."
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
        Process one batch in a single pipeline pass (no sub-batching).
        """
        # 1) Convert + compute embeddings
        object_models = self._convert_and_embed(batch_dict)

        # But _convert_and_embed is synchronous, let's gather them properly
        # Actually we might do it asynchronously
        object_models = await self._compute_embeddings(object_models)

        # 2) Merge all in one step
        merged_entities = await self._merge_intra_batch(object_models)

        # 3) Re-embed merges if needed
        merged_entities = await self._compute_embeddings_for_merged(merged_entities)

        # 4) Match or insert => references assigned
        await self._match_or_insert_merged_entities(merged_entities, batch_dict)

        # 5) Build final map
        logger.info(
            f"[Resolver] Building final map from {len(merged_entities)} merged entities"
        )
        for entity in merged_entities:
            logger.info(
                f"  Entity: name='{entity.name}', reference_id={entity.reference_id}, member_hash_keys={entity.member_hash_keys}"
            )

        resolved_dict: Dict[str, T] = {}
        for entity in merged_entities:
            if entity.reference_id is None:
                # Insert as fallback
                logger.warning(
                    f"[Resolver] Entity missing reference_id: {entity.name}. Performing fallback insert."
                )
                new_id = await self._safe_insert(entity)
                entity.reference_id = new_id
                logger.info(
                    f"[Resolver] Fallback insert complete. New reference_id: {new_id}"
                )

            for hk in entity.member_hash_keys:
                original_obj = batch_dict.get(hk)
                if original_obj is None:
                    # LLM might have introduced unknown keys
                    logger.warning(
                        f"[Resolver] LLM introduced unknown hash_key={hk}; skipping."
                    )
                    continue
                updated_obj = self._model_to_object(entity, original_obj)
                setattr(updated_obj, "postgres_reference", entity.reference_id)
                resolved_dict[hk] = updated_obj

        # For any objects not included by the LLM, fallback
        missing_keys = set(batch_dict.keys()) - set(resolved_dict.keys())
        if missing_keys:
            logger.warning(
                f"[Resolver] Found {len(missing_keys)} objects not included in LLM results. Adding fallbacks."
            )
            for hk in missing_keys:
                logger.info(f"[Resolver] Adding fallback for hash_key={hk}")
                resolved_dict[hk] = batch_dict[hk]

        return resolved_dict

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

    def _object_to_model(self, obj: T) -> ObjectWithEmbedding:
        reference_id = getattr(obj, "postgres_reference", None)
        embedding = getattr(obj, "embedding", []) or []
        return ObjectWithEmbedding(
            name=getattr(obj, "name", ""),
            description=getattr(obj, "description", ""),
            embedding=embedding,
            reference_id=reference_id,
            hash_key="",
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
    async def find_similar_by_embedding(
        self,
        conn,
        table_name: str,
        query_embedding: List[float],
        threshold: float,
        limit: int,
    ) -> List[dict]:
        pass

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
