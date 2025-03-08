"""
Concrete OpenAI-based resolver, chunking by max_objects_per_batch.
Logs:
- Intra-batch merges as a table
- Side-by-side LLM inputs & outputs for DB verification
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TypeVar

import tiktoken
from openai import OpenAI

from magi.config import OPENAI_CONFIG
from magi.services.rate_limiter import O3_MINI_RATE_LIMIT, rate_limiter
from magi.utils import get_logger

from .base import Resolver
from .models import ObjectWithEmbedding, MergedEntity, LLMIntraBatchMergeResponse

logger = get_logger(__name__)
T = TypeVar("T")


class OpenAIResolver(Resolver[T]):
    def __init__(
        self,
        conn,
        embedding_provider,
        table_name: str,
        reference_column: str = "id",
        similarity_threshold: float = 0.4,
        max_objects_per_batch: int = 50,
        model: str = "o3-mini-2025-01-31",
        api_key: str = OPENAI_CONFIG.api_key,
        max_retries: int = 5,
    ):
        super().__init__(
            conn,
            embedding_provider,
            table_name,
            reference_column,
            similarity_threshold,
            max_objects_per_batch,
        )
        self.model = model
        self.max_retries = max_retries

        # Initialize OpenAI
        self.client = OpenAI(api_key=api_key)

        # We use a simple token encoder
        self.tokenizer = (
            tiktoken.encoding_for_model(model)
            if model.startswith("gpt-")
            else tiktoken.get_encoding("cl100k_base")
        )
        self.reserved_tokens = 500

        # Rate limiter
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
        """
        if not objects:
            return []

        prompt_text = self._build_intra_batch_merge_prompt(objects)
        # We ignore token count, because we rely on max_objects_per_batch for safety
        response_text = await self._call_openai(prompt_text)

        merged_entities = self._parse_intra_batch_merge_result(response_text, objects)

        #  Logging a table: each row => {merged entity name, temp_id} => {names + hashes of members}
        self._log_merged_table(merged_entities, objects)

        return merged_entities

    def _build_intra_batch_merge_prompt(
        self, objects: List[ObjectWithEmbedding]
    ) -> str:
        prompt = (
            "You are an expert AI system in semantic coreference resolution. We have a batch of objects, each with "
            "a name, description, and a 'hash_key'. Identify duplicates and merge them into a single object. "
            "Two objects are duplicates if they refer to the same entity, concept, or idea, such as aliases.\n\n"
            "Output JSON:\n"
            "{\n"
            '  "merged_entities": [\n'
            "    {\n"
            '      "merged_id": "string",\n'
            '      "merged_name": "string",\n'
            '      "merged_description": "string",\n'
            '      "member_hash_keys": ["hashA","hashB"]\n'
            "    }, ...\n"
            "  ]\n"
            "}\n\n"
            "Constraints:\n"
            "- Each input object must appear exactly once in exactly one merged group.\n"
            "- 'merged_name' is a single canonical name that the object is best known by.\n"
            "- 'merged_description' is the best globally-identifying description of the object "
            "   synthesized from existing descriptions. If you learn aliases and names the object is known by, add them.\n\n"
            "Here are the objects:\n"
        )
        for i, obj in enumerate(objects):
            prompt += (
                f"Object {i}:\n"
                f"  hash_key: {obj.hash_key}\n"
                f"  name: {obj.name}\n"
                f"  description: {obj.description}\n\n"
            )
        prompt += "\nReturn valid JSON only."
        return prompt

    def _parse_intra_batch_merge_result(
        self, response_text: str, objects: List[ObjectWithEmbedding]
    ) -> List[MergedEntity]:
        try:
            data = json.loads(response_text)
            resp = LLMIntraBatchMergeResponse(**data)
        except Exception as e:
            logger.warning(
                f"[OpenAIResolver._parse_intra_batch_merge_result] JSON parse fail: {e}"
            )
            # fallback
            return [
                MergedEntity(
                    temp_id=obj.hash_key,
                    name=obj.name,
                    description=obj.description,
                    member_hash_keys=[obj.hash_key],
                    embedding=obj.embedding,
                )
                for obj in objects
            ]

        obj_map = {o.hash_key: o for o in objects}
        merged_entities = []
        for me in resp.merged_entities:
            # If only one member, reuse embedding
            if len(me.member_hash_keys) == 1:
                single_hk = me.member_hash_keys[0]
                source_obj = obj_map.get(single_hk)
                init_emb = source_obj.embedding if source_obj else []
            else:
                init_emb = []
            merged_entities.append(
                MergedEntity(
                    temp_id=me.merged_id,
                    name=me.merged_name,
                    description=me.merged_description,
                    member_hash_keys=me.member_hash_keys,
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
        candidates_batch = await self.find_similar_by_embeddings_batch(
            self.conn, self.table_name, all_embeddings, self.similarity_threshold, 1
        )

        pairs = []
        for idx, entity in enumerate(to_resolve):
            cands = candidates_batch[idx]
            if cands:
                pairs.append((entity, cands[0]))
            else:
                # no DB match => we will just insert
                pairs.append((entity, None))

        # Step C: single LLM call
        to_insert = await self._verify_db_pairs_single_call(pairs)

        # Step D: insert needed
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
        self, pairs: List[Tuple[MergedEntity, dict]]
    ) -> List[MergedEntity]:
        """
        Single LLM call. For each pair => are_same => unify or insert => we produce logging side-by-side.
        Return the list of entities that must be inserted new.
        """
        # Collect pairs that have no DB candidate
        no_candidate = [(e, None) for (e, c) in pairs if c is None]
        verify_pairs = [(e, c) for (e, c) in pairs if c is not None]

        # Log LLM input side-by-side
        if verify_pairs:
            logger.info("[_verify_db_pairs_single_call] LLM input pairs:")
            for idx, (entity, candidate) in enumerate(verify_pairs):
                logger.info(f" Pair {idx}:")
                logger.info(
                    f"   - MergedEntity: name='{entity.name}', desc='{entity.description}'"
                )
                logger.info(
                    f"   - DB Candidate: name='{candidate['name']}', desc='{candidate['description']}'"
                )

        to_insert = []
        # immediate insertion for those that have no DB candidate
        for entity, _ in no_candidate:
            to_insert.append(entity)

        if not verify_pairs:
            return to_insert

        # Build the prompt
        prompt_text = self._build_verification_batch_prompt(verify_pairs)
        response_text = await self._call_openai(prompt_text)

        # Parse
        results = self._parse_verification_batch_output(response_text, verify_pairs)

        # Now log side-by-side with LLM results
        logger.info("[_verify_db_pairs_single_call] LLM verification results:")
        for idx, (are_same, updated_name, updated_desc) in enumerate(results):
            entity, candidate = verify_pairs[idx]
            logger.info(
                f" Pair {idx} => are_same={are_same}, updated_name='{updated_name}', "
                f"updated_desc='{(updated_desc[:50] + '...') if updated_desc else None}'"
            )
            if are_same:
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
        We want a single JSON response:
          {
            "results": [
              {
                "pair_index": 0,
                "are_same": true,
                "updated_name": "...",
                "updated_description": "..."
              },
              ...
            ]
          }
        """
        prompt = (
            "You are an expert AI system in semantic coreference resolution. "
            "We have several pairs of (new_object, existing_object). For each pair, determine "
            "if they new object refers to the same entity, concept, or idea as the existing object."
            "If they are the same, compose a possibly updated canonical name that the object is best known by, "
            "and possibly augment the globally identifying description of the object. "
            " - are_same (true/false)\n"
            " - updated_name (if are_same)\n"
            " - updated_description (if are_same)\n\n"
            "Return valid JSON:\n"
            "{\n"
            '  "results": [\n'
            "    {\n"
            '      "pair_index": <int>,\n'
            '      "are_same": <bool>,\n'
            '      "updated_name": <string or null>,\n'
            '      "updated_description": <string or null>\n'
            "    }, ...\n"
            "  ]\n"
            "}\n\n"
            "Here are the pairs:\n"
        )
        for idx, (entity, candidate) in enumerate(pairs):
            prompt += f"Pair {idx}:\n"
            prompt += " New Object:\n"
            prompt += f"   name: {entity.name}\n"
            prompt += f"   description: {entity.description}\n\n"
            prompt += " Existing Object:\n"
            prompt += f"   name: {candidate['name']}\n"
            prompt += f"   description: {candidate['description']}\n\n"
        prompt += "Return only JSON."
        return prompt

    def _parse_verification_batch_output(
        self,
        response_text: str,
        pairs: List[Tuple[MergedEntity, dict]],
    ) -> List[Tuple[bool, Optional[str], Optional[str]]]:
        """
        Expects:
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
        Return list of (are_same, updated_name, updated_desc) in same order as pairs.
        """
        out = [(False, None, None)] * len(pairs)
        try:
            data = json.loads(response_text)
            items = data.get("results", [])
            for item in items:
                pair_index = item.get("pair_index")
                if pair_index is not None and 0 <= pair_index < len(pairs):
                    are_same = item.get("are_same", False)
                    updated_name = item.get("updated_name")
                    updated_desc = item.get("updated_description")
                    if not isinstance(are_same, bool):
                        are_same = False
                    if not isinstance(updated_name, (str, type(None))):
                        updated_name = None
                    if not isinstance(updated_desc, (str, type(None))):
                        updated_desc = None
                    out[pair_index] = (are_same, updated_name, updated_desc)
        except Exception as e:
            logger.warning(f"Failed to parse LLM verification JSON: {e}")
            # fallback => all false
        return out

    # --------------------------------------------------------------
    # Internal OpenAI call
    # --------------------------------------------------------------
    async def _call_openai(self, prompt_text: str) -> str:
        """
        Single function to call the LLM with a prompt (ignoring token count).
        We rely on max_objects_per_batch instead of tokens here.
        """
        async with self._rate_limiter.acquire_context(
            rate_limit=self._rate_limit,
            tokens=1000,  # or any constant cost
            reserve=True,
        ) as retry_after:
            if retry_after:
                wait_seconds = max(0.0, retry_after - datetime.now().timestamp())
                logger.debug(f"Rate-limited. Sleeping {wait_seconds} seconds.")
                await asyncio.sleep(wait_seconds)

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert AI system in semantic coreference resolution for knowledge graph construction. "
                                "Return valid JSON exactly as requested."
                            ),
                        },
                        {"role": "user", "content": prompt_text},
                    ],
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"[OpenAIResolver] call failed (attempt {attempt}): {e}")
                last_exc = e
                await asyncio.sleep(2**attempt)

        logger.error("[OpenAIResolver] All attempts failed, returning empty JSON.")
        if last_exc:
            raise last_exc
        return "{}"

    # --------------------------------------------------------------
    # DB Insert / Update / Retrieve
    # --------------------------------------------------------------
    async def _find_duplicate_in_db(self, entity: MergedEntity) -> Optional[int]:
        query = f"SELECT id FROM {self.table_name} WHERE name=$1 LIMIT 1"
        row = await self.conn.fetchrow(query, entity.name)
        if row:
            return row["id"]
        return None

    async def _insert_object_into_db(self, entity: MergedEntity) -> int:
        fields = ["name", "description", "embedding"]
        values = [entity.name, entity.description, str(entity.embedding)]
        placeholders = [f"${i + 1}" for i in range(len(values))]
        placeholders[-1] += "::vector"
        query = f"""
        INSERT INTO {self.table_name} ({", ".join(fields)})
        VALUES ({", ".join(placeholders)})
        RETURNING id
        """
        new_id = await self.conn.fetchval(query, *values)
        return new_id

    async def _update_object_in_db(self, object_id: int, updates: dict) -> None:
        if not updates:
            return
        set_clauses = []
        values = [object_id]
        param_index = 2
        for field, value in updates.items():
            if field == "embedding" and isinstance(value, list):
                set_clauses.append(f"{field} = ${param_index}::vector")
                values.append(str(value))
            else:
                set_clauses.append(f"{field} = ${param_index}")
                values.append(value)
            param_index += 1
        set_clause = ", ".join(set_clauses)
        query = f"UPDATE {self.table_name} SET {set_clause} WHERE id=$1"
        await self.conn.execute(query, *values)

    async def find_similar_by_embedding(
        self,
        conn,
        table_name: str,
        query_embedding: List[float],
        threshold: float,
        limit: int,
    ) -> List[dict]:
        emb_str = str(query_embedding)
        query = f"""
        SELECT id, name, description, 1 - (embedding <=> $1::vector) AS similarity
        FROM {table_name}
        WHERE 1 - (embedding <=> $1::vector) > $2
        ORDER BY similarity DESC
        LIMIT $3
        """
        rows = await conn.fetch(query, emb_str, threshold, limit)
        results = []
        for r in rows:
            results.append(
                {
                    "reference_id": r["id"],
                    "name": r["name"],
                    "description": r["description"],
                    "similarity": r["similarity"],
                }
            )
        return results

    async def find_similar_by_embeddings_batch(
        self,
        conn,
        table_name: str,
        query_embeddings: List[List[float]],
        threshold: float,
        limit_per_query: int = 1,
    ) -> List[List[dict]]:
        # We'll do it sequentially for clarity
        out = []
        for emb in query_embeddings:
            row = await self.find_similar_by_embedding(
                conn, table_name, emb, threshold, limit_per_query
            )
            out.append(row)
        return out

    async def close(self):
        await self._rate_limiter.close()
