"""
Abstract base class for entity and relationship type resolvers.
These resolvers handle the resolution of entities and relationship types
against existing database entries using embedding similarity and LLM verification.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

import asyncpg
import numpy as np

from magi.resolvers.models import (
    ObjectPair,
    ObjectWithEmbedding,
    ProcessedObject,
    VerificationResult,
)

from magi.embedders.base import EmbeddingProvider

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
        similarity_threshold: float = 0.2,
        max_tokens_per_batch: int = 4000,
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

            # Verify each batch
            verification_results = []
            for i, batch in enumerate(batches):
                logger.debug(
                    f"Verifying batch {i + 1}/{len(batches)} with {len(batch)} pairs"
                )
                batch_results = await self._verify_objects_batch(batch)
                verification_results.extend(batch_results)
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

                        """.strip()

        prompt += """
                Use the following criteria to determine if two objects are the same:
                - Do they refer to the same real-world entity, concept, or relationship?
                - Are they synonyms or different ways of expressing the same idea?
                - Would merging them provide a more complete understanding without introducing contradictions?

                For each pair, provide:
                1. Whether they are the same (true/false)
                2. If same, a name that best represents the entity and is identifiable in a global context
                3. If same, a description that combines information from both descriptions

                Your response should be a JSON object with the following structure:
                ```json
                {
                "results": [
                    {
                    "pair_id": 0,
                    "is_same": true,
                    "updated_name": "Best name that represents both objects",
                    "updated_description": "Combined description that includes information from both objects"
                    },
                    {
                    "pair_id": 1,
                    "is_same": false,
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

        Args:
            objects: List of objects with embeddings

        Returns:
            List of ObjectPair containing input_object, similar_object, and is_from_input_list flag
        """
        if not objects:
            return []

        # Filter out objects with empty embeddings
        valid_objects = [obj for obj in objects if obj.embedding]
        if not valid_objects:
            logger.warning("No objects with valid embeddings found")
            return []

        # Convert all embeddings to a numpy array for vectorized operations
        embeddings = np.array(
            [obj.embedding for obj in valid_objects], dtype=np.float32
        )

        # Early exit if we only have one object
        if len(valid_objects) == 1:
            # Only check database for similar objects
            db_results = await self.find_similar_by_embedding(
                self.conn,
                self.table_name,
                valid_objects[0].embedding,
                self.similarity_threshold,
                limit=1,
            )

            if not db_results:
                return []

            db_obj = db_results[0]
            # Create ObjectWithEmbedding from the dictionary
            similar_obj = ObjectWithEmbedding(
                name=db_obj["name"],
                description=db_obj["description"],
                reference_id=db_obj["reference_id"],
                embedding=[],  # Empty embedding for now
                hash_key=valid_objects[0].hash_key,  # Use same hash key as input object
                metadata={},
            )

            return [
                ObjectPair(
                    input_object=valid_objects[0],
                    similar_object=similar_obj,
                    is_from_input_list=False,
                )
            ]

        # Compute all pairwise similarities in one batch
        # Normalize the embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / np.maximum(
            norms, 1e-10
        )  # Avoid division by zero

        # Compute similarity matrix - dot product of normalized vectors is cosine similarity
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

        # Set diagonal to -1 to exclude self-comparisons
        np.fill_diagonal(similarity_matrix, -1)

        # Batch find similar objects in the database
        all_embeddings = [obj.embedding for obj in valid_objects]
        db_results_batch = await self.find_similar_by_embeddings_batch(
            self.conn,
            self.table_name,
            all_embeddings,
            self.similarity_threshold,
            limit_per_query=1,
        )

        # Process results and create ObjectPairs - exactly one per input object
        result_pairs = []
        for i, obj in enumerate(valid_objects):
            # Find the most similar object in the input list
            input_similarities = similarity_matrix[i]
            max_input_similarity = np.max(input_similarities)
            max_input_idx = np.argmax(input_similarities)

            # Check database results for this object
            db_obj_results = db_results_batch[i]
            db_similarity = 0.0

            if db_obj_results:
                db_obj = db_obj_results[0]
                db_similarity = db_obj["similarity"]

            # Compare similarities and choose the most similar object
            if max_input_similarity >= self.similarity_threshold and (
                not db_obj_results or max_input_similarity >= db_similarity
            ):
                # Input list object is more similar
                most_similar_input_obj = valid_objects[max_input_idx]
                result_pairs.append(
                    ObjectPair(
                        input_object=obj,
                        similar_object=most_similar_input_obj,
                        is_from_input_list=True,
                    )
                )
            elif db_obj_results and db_similarity >= self.similarity_threshold:
                # Database object is more similar
                db_obj = db_obj_results[0]
                similar_db_obj = ObjectWithEmbedding(
                    name=db_obj["name"],
                    description=db_obj["description"],
                    reference_id=db_obj["reference_id"],
                    embedding=[],  # Empty embedding for now
                    hash_key=obj.hash_key,  # Use same hash key as input object
                    metadata={},
                )
                result_pairs.append(
                    ObjectPair(
                        input_object=obj,
                        similar_object=similar_db_obj,
                        is_from_input_list=False,
                    )
                )
            # If neither is above threshold, no pair is created for this object

        return result_pairs

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
            List of verification results with input_object, db_object, is_same flag,
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
        Process verification results and update/create objects.

        Args:
            objects: Original list of objects with embeddings
            verification_results: Results from LLM verification
            objects_dict: Original dictionary of objects

        Returns:
            Dictionary mapping from the same hash keys to resolved objects with database references
        """
        # Create a mapping from hash_key to the original object for direct updates
        hash_to_object = {obj.hash_key: obj for obj in objects}

        # Dictionary to track processed objects by their hash key
        hash_to_processed: Dict[str, ProcessedObject[ObjectWithEmbedding]] = {}

        # Track objects that need embedding updates
        objects_to_update = []
        descriptions_to_embed = []

        # Process verification results
        for result in verification_results:
            input_obj = result.pair.input_object
            similar_obj = result.pair.similar_object
            hash_key = input_obj.hash_key

            # Skip if we've already processed this input object
            if hash_key in hash_to_processed:
                continue

            if result.is_same and similar_obj:
                # Check if the similar object has a reference_id
                if similar_obj.reference_id is None:
                    # No reference_id means we need to create a new object
                    obj_to_update = hash_to_object[hash_key]

                    # Apply any updates from verification
                    if result.updated_name:
                        obj_to_update.name = result.updated_name
                    if result.updated_description:
                        obj_to_update.description = result.updated_description

                    # Create a new database entry
                    db_id = await self._insert_object_into_db(obj_to_update)

                    # Set the reference_id directly
                    obj_to_update.reference_id = db_id

                    hash_to_processed[hash_key] = ProcessedObject(
                        resolved=obj_to_update,
                        reference_id=db_id,
                        hash_key=hash_key,
                        is_new=True,
                        has_updates=False,
                    )

                    continue

                # Objects are the same - either link or update
                updated_name = result.updated_name
                updated_description = result.updated_description

                # Determine if we need to update the existing object
                if updated_name or updated_description:
                    # Get the original object to update
                    obj_to_update = hash_to_object[hash_key]

                    # Update the object
                    if updated_name:
                        obj_to_update.name = updated_name
                    if updated_description:
                        obj_to_update.description = updated_description

                    # Set the reference_id
                    obj_to_update.reference_id = similar_obj.reference_id

                    # Update the object in the database
                    await self._update_object_in_db(
                        similar_obj.reference_id,
                        {
                            "name": obj_to_update.name,
                            "description": obj_to_update.description,
                        },
                    )

                    # Queue for embedding update if description changed
                    if updated_description:
                        objects_to_update.append((obj_to_update, hash_key))
                        descriptions_to_embed.append(obj_to_update.description)

                    # Store in our results
                    hash_to_processed[hash_key] = ProcessedObject(
                        resolved=obj_to_update,
                        reference_id=similar_obj.reference_id,
                        hash_key=hash_key,
                        is_new=False,
                        has_updates=True,
                    )
                else:
                    # No updates needed, just link to existing object
                    obj_to_update = hash_to_object[hash_key]
                    obj_to_update.reference_id = similar_obj.reference_id

                    hash_to_processed[hash_key] = ProcessedObject(
                        resolved=obj_to_update,
                        reference_id=similar_obj.reference_id,
                        hash_key=hash_key,
                        is_new=False,
                        has_updates=False,
                    )
            else:
                # Objects are different or no similar object - create new one
                obj_to_update = hash_to_object[hash_key]
                db_id = await self._insert_object_into_db(obj_to_update)

                # Set the reference_id directly
                obj_to_update.reference_id = db_id

                hash_to_processed[hash_key] = ProcessedObject(
                    resolved=obj_to_update,
                    reference_id=db_id,
                    hash_key=hash_key,
                    is_new=True,
                    has_updates=False,
                )

        # Process any remaining objects that weren't in verification results
        for obj in objects:
            hash_key = obj.hash_key
            if hash_key not in hash_to_processed:
                # Create a new database entry
                db_id = await self._insert_object_into_db(obj)

                # Set the reference_id directly
                obj.reference_id = db_id

                hash_to_processed[hash_key] = ProcessedObject(
                    resolved=obj,
                    reference_id=db_id,
                    hash_key=hash_key,
                    is_new=True,
                    has_updates=False,
                )

        # Compute new embeddings in batch if needed
        if descriptions_to_embed:
            new_embeddings = await self.embedding_provider.embed(
                texts=descriptions_to_embed,
                truncation=True,
                embed_prompt=self.EMBED_PROMPT,
            )

            # Update objects with new embeddings sequentially instead of in parallel
            for i, (obj_to_update, hash_key) in enumerate(objects_to_update):
                # Update the embedding in the object directly
                obj_to_update.embedding = new_embeddings[i]

                # Update the object in the database
                await self._update_object_in_db(
                    obj_to_update.reference_id,
                    {"embedding": new_embeddings[i]},
                )

        # Convert processed objects back to original type and maintain dictionary structure
        resolved_dict = {}
        for hash_key, obj in objects_dict.items():
            if hash_key in hash_to_processed:
                processed = hash_to_processed[hash_key]
                # Convert the processed model back to the original object type
                resolved_dict[hash_key] = self._model_to_object(processed.resolved, obj)
            else:
                # This should not happen, but just in case
                logger.warning(
                    f"Object with hash {hash_key} was not processed, using original"
                )
                resolved_dict[hash_key] = obj

        return resolved_dict

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
