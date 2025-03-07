"""
Abstract base class for entity and relationship type resolvers.
These resolvers handle the resolution of entities and relationship types
against existing database entries using embedding similarity and LLM verification.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

import asyncpg
import numpy as np

from magi.embedders.base import EmbeddingProvider
from magi.resolvers.models import (
    ObjectPair,
    ObjectWithEmbedding,
    ProcessedObject,
    VerificationResult,
)

logger = logging.getLogger(__name__)

# Generic type for the object (entity or relationship type)
T = TypeVar("T")


class Resolver(Generic[T], ABC):
    """
    Abstract base class for resolvers of objects (entities or relationship types).

    Resolvers are responsible for:
    1. Finding similar objects in the database using embedding similarity
    2. Verifying if the retrieved objects are the same as the input objects using an LLM
    3. Updating existing objects with new information or creating new objects
    4. Returning resolved objects with database references
    """

    # Prompt for embedding objects for retrieval
    EMBED_PROMPT = "Represent the following unique object description for retrieval: "
    RETRIEVAL_PROMPT = "Represent the following unique object description for retrieving a object with that description: "

    def __init__(
        self,
        conn: asyncpg.Connection,
        embedding_provider: EmbeddingProvider,
        table_name: str,
        reference_column: str,
        similarity_threshold: float = 0.4,
        max_tokens_per_batch: int = 4000,
        candidate_epsilon: float = 0.05,
        db_candidate_limit: int = 1,
        max_concurrent_requests: int = 40,
    ):
        """
        Initialize the resolver with database connection and parameters.

        Args:
            conn: asyncpg connection
            embedding_provider: Provider for computing embeddings
            table_name: Name of the table to search in (e.g., 'entities', 'relationship_types')
            reference_column: Name of the column that serves as a reference (e.g., 'id')
            similarity_threshold: Threshold for considering objects as similar (0-1)
            max_tokens_per_batch: Maximum number of tokens per LLM batch
        """
        self.conn = conn
        self.embedding_provider = embedding_provider
        self.table_name = table_name
        self.reference_column = reference_column
        self.similarity_threshold = similarity_threshold
        self.max_tokens_per_batch = max_tokens_per_batch
        self.candidate_epsilon = candidate_epsilon
        self.db_candidate_limit = db_candidate_limit
        self.max_concurrent_requests = max_concurrent_requests

    async def resolve(self, objects_dict: Dict[str, T]) -> Dict[str, T]:
        """
        Resolve a dictionary of objects against themselves & existing database entries.

        This method:
        1. Computes embeddings for each object
        2. Finds similar objects in the database using embedding similarity
        3. Verifies if the retrieved objects are the same using an LLM
        4. Updates existing objects or creates new ones
        5. Returns resolved objects with database references

        Args:
            objects_dict: Dictionary mapping from hash to objects (Entity or RelationshipType) to resolve

        Returns:
            Dictionary mapping from the same hash keys to resolved objects with database references
        """
        if not objects_dict:
            logger.info("No objects to resolve, returning empty dictionary")
            return {}

        try:
            logger.info(f"Resolving {len(objects_dict)} objects")

            # Convert objects to models and include hash keys
            object_models = []
            for hash_key, obj in objects_dict.items():
                model = self._object_to_model(obj)
                model.hash_key = hash_key  # Assign the hash key to the model
                object_models.append(model)

            logger.debug(
                f"Converted {len(object_models)} objects to models with hash keys"
            )

            # Compute embeddings
            object_models_with_embeddings = await self._compute_embeddings(
                object_models
            )
            logger.debug(
                f"Computed embeddings for {len(object_models_with_embeddings)} objects"
            )

            # Find similar objects
            object_pairs = await self._find_similar_objects(
                object_models_with_embeddings
            )
            logger.debug(f"Found {len(object_pairs)} object pairs")

            # Create batches for LLM verification
            batches = await self._create_verification_batches(object_pairs)
            logger.debug(f"Created {len(batches)} verification batches")

            semaphore = asyncio.Semaphore(self.max_concurrent_requests)

            async def verify_batch_with_semaphore(batch, batch_index):
                async with semaphore:
                    logger.debug(
                        f"Verifying batch {batch_index + 1}/{len(batches)} with {len(batch)} pairs"
                    )
                    return await self._verify_objects_batch(batch)

            # Create tasks for all batches
            verification_tasks = [
                verify_batch_with_semaphore(batch, i) for i, batch in enumerate(batches)
            ]

            # Execute all tasks concurrently and gather results
            batch_results = await asyncio.gather(*verification_tasks)

            # Flatten the results
            verification_results = [
                result for batch_result in batch_results for result in batch_result
            ]

            logger.debug(f"Verified {len(verification_results)} object pairs")

            # Process verification results - now returns Dict[str, T] directly
            resolved_dict = await self._process_verification_results(
                object_models_with_embeddings, verification_results, objects_dict
            )
            logger.debug(
                f"Processed verification results into {len(resolved_dict)} resolved objects"
            )

            logger.info(f"Successfully resolved {len(resolved_dict)} objects")
            return resolved_dict

        except Exception as e:
            logger.exception(f"Error in Resolver.resolve: {str(e)}")
            # Return the original objects if there's an error
            return objects_dict

    def _create_verification_prompt(self, pairs: List[ObjectPair]) -> str:
        """
        Create a prompt for the LLM to verify if objects are the same.

        Args:
            pairs: List of ObjectPair instances

        Returns:
            Prompt string for the LLM
        """
        prompt = """
                As an expert AI assistant in entity resolution, you analyze pairs of objects and determine if they refer to the same entity or concept.

                For each pair:
                1. Determine if Object A and Object B refer to the same entity or concept.
                2. If they are the same, create an updated name and description that combines the information from both.
                3. If they are not the same, leave the updated fields as null.

                Respond with a JSON object containing your analysis for each pair.

                Here are the pairs to analyze:
                """.strip()

        for i, pair in enumerate(pairs):
            prompt += f"""
                        Pair {i}:
                        Object A:
                        - Name: {pair.input_object.name}
                        - Description: {pair.input_object.description}

                        Object B:
                        - Name: {pair.similar_object.name}
                        - Description: {pair.similar_object.description}

                        """

        prompt += """
                Use the following criteria to determine if two objects are the same:
                - Do they refer to the same real-world entity, concept, or relationship?
                - Are they aliases, nicknames, synonyms or different ways of expressing the same idea?
                - Would merging them provide a more complete understanding without introducing contradictions?

                For each pair, provide:
                1. Whether they are the same (true/false)
                2. If same, a canonical name that best represents the object and is identifiable in a global context
                3. If same, a description that combines information from both descriptions

                The name field should contain one canonical name the object is best known by.
                Any aliases or other names the object is known as should be included in the description.

                Your response should be a JSON object with the following structure:
                ```json
                {
                "results": [
                    {
                    "pair_id": 0,
                    "are_same": true,
                    "updated_name": "A canonical name that is identifiable globally",
                    "updated_description": "Revised globally-identifying description of the object, combining information from both entries"
                    },
                    {
                    "pair_id": 1,
                    "are_same": false,
                    "updated_name": null,
                    "updated_description": null
                    }
                ]
                }
                ```
                """.strip()

        return prompt

    def _object_to_model(self, obj: T) -> ObjectWithEmbedding:
        """
        Convert an object to a model for processing.

        Args:
            obj: Entity or RelationshipType object

        Returns:
            Model representation of the object
        """
        # Extract reference ID based on object type
        reference_id = None
        if hasattr(obj, "postgres_reference"):
            reference_id = obj.postgres_reference

        # Create the model
        return ObjectWithEmbedding(
            name=obj.name,
            description=obj.description,
            embedding=obj.embedding if hasattr(obj, "embedding") else [],
            reference_id=reference_id,
            hash_key=obj.hash_key,
            metadata={},  # Additional fields could be added here if needed
        )

    def _model_to_object(self, model: ObjectWithEmbedding, original_obj: T) -> T:
        """
        Convert a model back to an object.

        Args:
            model: Model representation of the object
            original_obj: Original object to use as a template

        Returns:
            Updated object with resolved fields
        """
        # Create a new object of the same type as the original
        if hasattr(original_obj, "postgres_reference"):
            # Track if an object is missing a reference_id
            if model.reference_id is None:
                logger.warning(
                    f"Object {model.name} does not have a reference_id after resolution"
                )

            # Assuming Entity or similar
            updated_obj = type(original_obj)(
                name=model.name,
                description=model.description,
                embedding=model.embedding,
                postgres_reference=model.reference_id,
            )
            return updated_obj
        else:
            # Generic fallback
            for key, value in vars(model).items():
                if key != "metadata":  # Skip the metadata dictionary
                    setattr(original_obj, key, value)
            return original_obj

    async def _compute_embeddings(
        self, objects: List[ObjectWithEmbedding]
    ) -> List[ObjectWithEmbedding]:
        """
        Compute embeddings for a list of objects.

        Args:
            objects: List of objects

        Returns:
            List of objects with added embeddings
        """
        # Skip objects that already have embeddings
        objects_to_embed = []
        for obj in objects:
            if not obj.embedding:
                objects_to_embed.append(obj)

        if not objects_to_embed:
            return objects

        # Prepare descriptions for embedding
        descriptions = []
        for obj in objects_to_embed:
            if obj.name and obj.description:
                prompt = f"{obj.name}: {obj.description}"
            elif obj.name:
                prompt = f"{obj.name}"
            else:
                prompt = f"{obj.description}"
            descriptions.append(prompt)

        # Compute embeddings in batch
        embeddings = await self.embedding_provider.embed(
            texts=descriptions,
            truncation=True,
            embed_prompt=self.EMBED_PROMPT,
        )

        # Assign embeddings to objects
        embed_idx = 0
        for obj in objects:
            if not obj.embedding:
                obj.embedding = embeddings[embed_idx]
                embed_idx += 1

        return objects

    async def _find_similar_objects(
        self, objects: List[ObjectWithEmbedding]
    ) -> List[ObjectPair[ObjectWithEmbedding]]:
        """
        Find similar objects in both the input list and the database for each input object.

        For each valid object (with a non-empty embedding), we:
        1. Compute pairwise similarities with other input objects.
        2. Retrieve candidate matches from the database (up to a configurable limit).
        3. For both input and DB candidates, we include all matches with similarity >= threshold
            and within an epsilon margin of the top candidate.

        Returns:
            List of ObjectPair containing input_object, candidate similar_object, and a flag indicating
            if the candidate came from the input list.
        """
        if not objects:
            return []

        # Only consider objects with valid embeddings.
        valid_objects = [obj for obj in objects if obj.embedding]
        if not valid_objects:
            logger.warning("No objects with valid embeddings found")
            return []

        # Convert embeddings to a numpy array.
        embeddings = np.array(
            [obj.embedding for obj in valid_objects], dtype=np.float32
        )

        # Configuration parameters.
        similarity_threshold = self.similarity_threshold
        epsilon = self.candidate_epsilon  # margin below the top similarity
        db_candidate_limit = self.db_candidate_limit  # how many DB candidates to fetch

        # Special handling if there's only one valid object: only check DB.
        if len(valid_objects) == 1:
            db_results = await self.find_similar_by_embedding(
                self.conn,
                self.table_name,
                valid_objects[0].embedding,
                similarity_threshold,
                limit=db_candidate_limit,
            )
            candidate_pairs = []
            if db_results:
                # Find top DB similarity.
                top_db_similarity = db_results[0]["similarity"]
                for db_obj in db_results:
                    if db_obj["similarity"] >= max(
                        similarity_threshold, top_db_similarity - epsilon
                    ):
                        candidate = ObjectWithEmbedding(
                            name=db_obj["name"],
                            description=db_obj["description"],
                            reference_id=db_obj["reference_id"],
                            embedding=[],  # not needed here
                            hash_key=valid_objects[
                                0
                            ].hash_key,  # use same hash key as input object
                            metadata={},
                        )
                        candidate_pairs.append(
                            ObjectPair(
                                input_object=valid_objects[0],
                                similar_object=candidate,
                                is_from_input_list=False,
                            )
                        )
            return candidate_pairs

        # Normalize embeddings for cosine similarity.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / np.maximum(norms, 1e-10)

        # Compute the similarity matrix for input objects.
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        # Exclude self-comparisons.
        np.fill_diagonal(similarity_matrix, -1)

        # Retrieve DB candidates in batch (using a limit higher than 1).
        db_results_batch = await self.find_similar_by_embeddings_batch(
            self.conn,
            self.table_name,
            [obj.embedding for obj in valid_objects],
            similarity_threshold,
            limit_per_query=db_candidate_limit,
        )

        candidate_pairs = []  # final list of ObjectPair

        # For each input object, collect candidates from both sources.
        for i, obj in enumerate(valid_objects):
            # --- Input-list candidates ---
            input_sims = similarity_matrix[i]
            # Identify candidates with similarity above threshold.
            input_candidate_indices = np.where(input_sims >= similarity_threshold)[0]
            input_candidates = []
            if input_candidate_indices.size > 0:
                top_input_similarity = np.max(input_sims[input_candidate_indices])
                # Include all input neighbors whose similarity is within epsilon of the top.
                for j in input_candidate_indices:
                    if input_sims[j] >= max(
                        similarity_threshold, top_input_similarity - epsilon
                    ):
                        # Avoid self-pairing (should be already excluded via diag=-1, but safe-check)
                        if j == i:
                            continue
                        input_candidates.append((j, input_sims[j]))

            # --- Database candidates ---
            db_candidates = []
            db_result_list = db_results_batch[i]
            if db_result_list:
                top_db_similarity = db_result_list[0]["similarity"]
                for db_candidate in db_result_list:
                    if db_candidate["similarity"] >= max(
                        similarity_threshold, top_db_similarity - epsilon
                    ):
                        db_candidates.append((db_candidate, db_candidate["similarity"]))

            # --- Aggregate candidate pairs for this object ---
            # For each candidate from input list:
            for j, sim_score in input_candidates:
                candidate_pairs.append(
                    ObjectPair(
                        input_object=obj,
                        similar_object=valid_objects[j],
                        is_from_input_list=True,
                    )
                )

            # For each candidate from the DB:
            for db_candidate, sim_score in db_candidates:
                # Create a candidate object from DB result.
                candidate_obj = ObjectWithEmbedding(
                    name=db_candidate["name"],
                    description=db_candidate["description"],
                    reference_id=db_candidate["reference_id"],
                    embedding=[],  # not required here
                    hash_key=obj.hash_key,  # use same hash_key as input object
                    metadata={},
                )
                candidate_pairs.append(
                    ObjectPair(
                        input_object=obj,
                        similar_object=candidate_obj,
                        is_from_input_list=False,
                    )
                )

        return candidate_pairs

    @abstractmethod
    async def _create_verification_batches(
        self, object_pairs: List[ObjectPair[ObjectWithEmbedding]]
    ) -> List[List[ObjectPair[ObjectWithEmbedding]]]:
        """
        Group object pairs into batches to avoid token limit.

        Args:
            object_pairs: List of ObjectPair containing input_object, similar_object, and is_from_input_list flag

        Returns:
            List of batches, where each batch is a list of object pairs
        """
        pass

    @abstractmethod
    async def _verify_objects_batch(
        self, batch: List[ObjectPair[ObjectWithEmbedding]]
    ) -> List[VerificationResult[ObjectWithEmbedding]]:
        """
        Verify if the retrieved objects are the same as the input objects using an LLM.

        Args:
            batch: List of ObjectPair containing input_object, similar_object, and is_from_input_list flag

        Returns:
            List of verification results with input_object, db_object, are_same flag,
            updated_name, updated_description, and is_from_input_list flag
        """
        pass

    async def _process_verification_results(
        self,
        objects: List[ObjectWithEmbedding],
        verification_results: List[VerificationResult[ObjectWithEmbedding]],
        objects_dict: Dict[str, T],
    ) -> Dict[str, T]:
        """
        Process verification results and update/create objects with correct merges.

        Steps:
        1) Build adjacency graph for all pairs (are_same=True) -> group objects that are identical.
        2) Find connected components (each group is one real-world entity).
        3) For each group, either:
            - Link to an existing database row (if any member has reference_id),
            - or insert exactly one new row for the entire group.
        4) Mark all items in that group with the chosen reference_id so they point to the same DB record.
        5) Recompute embeddings if any group members' descriptions were updated by the LLM.
        6) Convert them back to original object types for the final return.
        """

        # ----- 1) Prepare Data Structures -----

        # We map each hash_key -> the ObjectWithEmbedding for easy reference
        hash_to_object = {obj.hash_key: obj for obj in objects}

        # Our final result data: hash_key -> ProcessedObject
        hash_to_processed: Dict[str, ProcessedObject[ObjectWithEmbedding]] = {}

        # Step 1a: We'll store partial LLM updates (updated_name / updated_description)
        # keyed by the input_obj's hash_key. If multiple pairs updated the same object,
        # you might want more sophisticated merging logic. For simplicity, the last one
        # would overwrite. You can refine as needed.
        partial_updates: Dict[str, Dict[str, str]] = {}

        # Step 1b: Build adjacency dict for objects that are "the same"
        adjacency = {hk: set() for hk in hash_to_object.keys()}

        # We only mark adjacency if result.are_same is True
        for result in verification_results:
            if result.are_same and result.pair.similar_object is not None:
                in_key = result.pair.input_object.hash_key
                sim_key = result.pair.similar_object.hash_key
                adjacency[in_key].add(sim_key)
                adjacency[sim_key].add(in_key)

                # If LLM suggested updated fields, stash them:
                if result.updated_name or result.updated_description:
                    partial_updates.setdefault(in_key, {})
                    if result.updated_name:
                        partial_updates[in_key]["name"] = result.updated_name
                    if result.updated_description:
                        partial_updates[in_key]["description"] = (
                            result.updated_description
                        )

        # ----- 2) Find Connected Components (Groups) -----
        visited = set()

        def dfs(start, group):
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                group.add(node)
                for nei in adjacency[node]:
                    if nei not in visited:
                        stack.append(nei)

        groups = []
        for hk in adjacency:
            if hk not in visited:
                comp = set()
                dfs(hk, comp)
                groups.append(comp)

        # ----- 3) For each group, unify under a single DB reference -----
        for group in groups:
            # If all items in this group are already processed, skip
            if all(hk in hash_to_processed for hk in group):
                continue

            # Check if any object in this group has an existing reference_id
            any_db_obj = None
            for hk in group:
                if hash_to_object[hk].reference_id:
                    any_db_obj = hash_to_object[hk]
                    break

            # We'll pick one "representative" from the group for insertion or linking
            rep_key = next(iter(group))  # just pick the first hash_key
            rep_obj = hash_to_object[rep_key]

            # If the LLM gave partial updates for rep_obj, apply them
            if rep_key in partial_updates:
                pu = partial_updates[rep_key]
                if "name" in pu:
                    rep_obj.name = pu["name"]
                if "description" in pu:
                    rep_obj.description = pu["description"]

            if any_db_obj:
                # If any object is already in the DB, unify everything under that ID
                rep_obj.reference_id = any_db_obj.reference_id
                # Optionally do an update if you want to merge info into the existing DB row
                # We'll skip that detail or mark for later
            else:
                # Insert exactly one row for the entire group
                rep_id = await self._safe_insert(rep_obj)
                rep_obj.reference_id = rep_id

            # Now assign the rep_obj's reference_id to every item in the group
            for hk in group:
                if hk in hash_to_processed:
                    continue  # already done

                obj = hash_to_object[hk]

                # If partial updates exist for this specific object, apply them
                if hk in partial_updates and obj is not rep_obj:
                    pu = partial_updates[hk]
                    if "name" in pu:
                        obj.name = pu["name"]
                    if "description" in pu:
                        obj.description = pu["description"]

                # Link to the representative's reference
                obj.reference_id = rep_obj.reference_id

                # Mark it as processed
                hash_to_processed[hk] = ProcessedObject(
                    resolved=obj,
                    reference_id=rep_obj.reference_id,
                    hash_key=hk,
                    is_new=(any_db_obj is None),  # new only if we just inserted
                    has_updates=False,
                )

        # 4) Handle any objects not in adjacency (i.e. no pairs or no matches).
        #    The DFS covers them as singletons, but let's be safe in case something was missed.
        for obj in objects:
            if obj.hash_key not in hash_to_processed:
                # This is truly isolated; create or unify
                if obj.reference_id:
                    # Already in DB, so just store it
                    hash_to_processed[obj.hash_key] = ProcessedObject(
                        resolved=obj,
                        reference_id=obj.reference_id,
                        hash_key=obj.hash_key,
                        is_new=False,
                        has_updates=False,
                    )
                else:
                    # Insert brand new
                    if obj.hash_key in partial_updates:
                        pu = partial_updates[obj.hash_key]
                        if "name" in pu:
                            obj.name = pu["name"]
                        if "description" in pu:
                            obj.description = pu["description"]
                    new_id = await self._safe_insert(obj)
                    obj.reference_id = new_id
                    hash_to_processed[obj.hash_key] = ProcessedObject(
                        resolved=obj,
                        reference_id=new_id,
                        hash_key=obj.hash_key,
                        is_new=True,
                        has_updates=False,
                    )

        # ----- 5) Recompute embeddings if any object got an updated description -----

        # We check partial_updates for "description" changes. If the LLM proposed a new description,
        # we embed it. Note: you could also track changes from merges, but let's keep it straightforward.
        updated_objs = []
        for hk, obj in hash_to_object.items():
            # if partial_updates had a new "description" for hk, let's assume we need a new embedding
            if hk in partial_updates and "description" in partial_updates[hk]:
                updated_objs.append((obj, hk))

        if updated_objs:
            # 5a) gather the updated descriptions
            new_texts = [o.description for (o, _) in updated_objs]

            # 5b) do a batch embed
            new_embeddings = await self.embedding_provider.embed(
                texts=new_texts,
                truncation=True,
                embed_prompt=self.EMBED_PROMPT,
            )

            # 5c) update each object's embedding in-memory & in the DB
            for i, (obj_to_update, hk) in enumerate(updated_objs):
                obj_to_update.embedding = new_embeddings[i]
                # update DB
                await self._update_object_in_db(
                    obj_to_update.reference_id,
                    {"embedding": new_embeddings[i]},
                )

                # Mark that we have effectively updated the DB
                processed = hash_to_processed[hk]
                hash_to_processed[hk] = ProcessedObject(
                    resolved=obj_to_update,
                    reference_id=obj_to_update.reference_id,
                    hash_key=processed.hash_key,
                    is_new=processed.is_new,
                    has_updates=True,  # or True if partial updates were applied
                )

        # ----- 6) Convert processed objects back to original types in the final dictionary -----
        resolved_dict = {}
        for hk, original_obj in objects_dict.items():
            if hk in hash_to_processed:
                processed = hash_to_processed[hk]
                # Convert the processed model back to the original object type
                resolved_dict[hk] = self._model_to_object(
                    processed.resolved, original_obj
                )
            else:
                logger.warning(
                    f"Object with hash {hk} was not processed at all, using original"
                )
                resolved_dict[hk] = original_obj

        return resolved_dict

    async def _safe_insert(self, obj: ObjectWithEmbedding) -> int:
        """
        Insert the object into the DB with concurrency check.
        This tries to prevent duplicates if another parallel task
        inserted a matching item simultaneously.

        In practice, you should enforce a unique constraint in the DB
        (e.g., on a canonical name or an embedding-based fingerprint)
        and handle IntegrityError with a fallback SELECT.
        """
        # Optionally search the DB to see if a row with the same canonical name
        # or embedding signature just appeared. If found, reuse that ID.
        maybe_id = await self._find_duplicate_in_db(obj)
        if maybe_id is not None:
            return maybe_id

        # Otherwise, proceed with an insert
        new_id = await self._insert_object_into_db(obj)
        return new_id

    async def _find_duplicate_in_db(self, obj: ObjectWithEmbedding) -> Optional[int]:
        """
        Example concurrency check: see if a row with the same name
        or other signature was just inserted.

        For robust concurrency handling:
        - Add a UNIQUE constraint in the DB for the name or an embedding hash
        - Catch IntegrityError
        - If so, SELECT the row to find its ID
        """
        # For demonstration, we just do a quick name check:
        query = f"""
            SELECT id
            FROM {self.table_name}
            WHERE name = $1
            LIMIT 1
        """
        row = await self.conn.fetchrow(query, obj.name)
        if row:
            return row["id"]
        return None

    async def _update_object_in_db(
        self, object_id: int, updates: Dict[str, Any]
    ) -> None:
        """
        Update an object in the database.

        Args:
            object_id: ID of the object to update
            updates: Dictionary of field-value pairs to update
        """
        if not updates:
            return

        # Build the SET clause
        set_clauses = []
        values = [object_id]  # First parameter is the ID
        param_index = 2  # Start from $2 since $1 is the ID

        for field, value in updates.items():
            if field == "embedding":
                # Handle embedding specially - cast to vector type
                set_clauses.append(f"{field} = ${param_index}::vector")
                # Convert embedding to string representation
                embedding_str = str(value)
                values.append(embedding_str)
            else:
                set_clauses.append(f"{field} = ${param_index}")
                values.append(value)
            param_index += 1

        set_clause = ", ".join(set_clauses)

        # Execute the update
        query = f"""
        UPDATE {self.table_name}
        SET {set_clause}
        WHERE id = $1
        """

        await self.conn.execute(query, *values)

    async def _insert_object_into_db(self, obj: ObjectWithEmbedding) -> int:
        """
        Insert a new object into the database.

        Args:
            obj: Object to insert

        Returns:
            ID of the inserted object
        """
        # Extract fields and values
        fields = ["name", "description", "embedding"]
        values = [obj.name, obj.description]

        # Convert embedding to string representation and ensure it's cast to vector type
        embedding_str = str(obj.embedding)
        values.append(embedding_str)

        # Build the query with proper casting to vector type
        placeholders = [f"${i + 1}" for i in range(len(fields))]
        # Explicitly cast the embedding to vector type
        placeholders[-1] = f"${len(fields)}::vector"

        fields_str = ", ".join(fields)
        placeholders_str = ", ".join(placeholders)

        query = f"""
        INSERT INTO {self.table_name} ({fields_str})
        VALUES ({placeholders_str})
        RETURNING id
        """

        # Execute the query and get the ID
        result = await self.conn.fetchval(query, *values)
        return result

    async def find_similar_by_embedding(
        self,
        conn,
        table_name: str,
        query_embedding: List[float],
        threshold: float,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Find similar objects by cosine similarity using pgvector.

        Args:
            conn: asyncpg connection
            table_name: Name of the table to search in
            query_embedding: Embedding vector to compare against
            threshold: Similarity threshold (higher means more similar)
            limit: Maximum number of results to return

        Returns:
            List of similar objects with similarity scores
        """
        # Convert the embedding to a string representation for SQL
        embedding_str = str(query_embedding)

        # Query for similar objects using cosine similarity
        # 1 - cosine_distance gives us cosine similarity
        query = f"""
        SELECT id, name, description, 1 - (embedding <=> $1::vector) AS similarity
        FROM {table_name}
        WHERE 1 - (embedding <=> $1::vector) > $2
        ORDER BY similarity DESC
        LIMIT $3
        """

        rows = await conn.fetch(query, embedding_str, threshold, limit)

        # Convert rows to dictionaries
        results = []
        for row in rows:
            similar_obj = {
                "name": row["name"],
                "description": row["description"],
                "reference_id": row["id"],
                "similarity": row["similarity"],
                # We don't retrieve the embedding here to save bandwidth
                "embedding": [],
            }
            results.append(similar_obj)

        return results

    async def find_similar_by_embeddings_batch(
        self,
        conn,
        table_name: str,
        query_embeddings: List[List[float]],
        threshold: float,
        limit_per_query: int = 1,
    ) -> List[List[Dict[str, Any]]]:
        """
        Find similar objects by cosine similarity using pgvector in batch.

        Args:
            conn: asyncpg connection
            table_name: Name of the table to search in
            query_embeddings: List of embedding vectors to compare against
            threshold: Similarity threshold (higher means more similar)
            limit_per_query: Maximum number of results to return per query

        Returns:
            List of lists of similar objects with similarity scores
        """
        if not query_embeddings:
            return []

        # Prepare the query template with a placeholder for the embedding
        query_template = f"""
        SELECT id, name, description, 1 - (embedding <=> $1::vector) AS similarity
        FROM {table_name}
        WHERE 1 - (embedding <=> $1::vector) > $2
        ORDER BY similarity DESC
        LIMIT $3
        """

        # Execute queries sequentially but prepare them all at once
        results = []
        for embedding in query_embeddings:
            # Convert the embedding to a string representation for SQL
            embedding_str = str(embedding)

            # Execute the query
            rows = await conn.fetch(
                query_template, embedding_str, threshold, limit_per_query
            )

            # Convert rows to dictionaries
            batch_results = []
            for row in rows:
                similar_obj = {
                    "name": row["name"],
                    "description": row["description"],
                    "reference_id": row["id"],
                    "similarity": row["similarity"],
                    # We don't retrieve the embedding here to save bandwidth
                    "embedding": [],
                }
                batch_results.append(similar_obj)

            results.append(batch_results)

        return results
