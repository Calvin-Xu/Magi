"""
Abstract base class for entity and relationship type resolvers.
These resolvers handle the resolution of entities and relationship types
against existing database entries using embedding similarity and LLM verification.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

import asyncpg
import numpy as np

from magi.resolvers.models import ObjectWithEmbedding

from magi.embedders.base import EmbeddingProvider
from magi.utils import get_logger, log_async_function_call
from magi.resolvers.models import (
    ObjectPair,
    ProcessedObject,
    SimilarObject,
    VerificationResult,
)

# Create a logger for this module
logger = get_logger(__name__)

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

    @log_async_function_call()
    async def resolve(self, objects: List[T]) -> List[T]:
        """
        Resolve a list of objects against existing database entries.

        This method:
        1. Computes embeddings for each object
        2. Finds similar objects in the database using embedding similarity
        3. Verifies if the retrieved objects are the same using an LLM
        4. Updates existing objects or creates new ones
        5. Returns resolved objects with database references

        Args:
            objects: List of objects (Entity or RelationshipType) to resolve

        Returns:
            List of resolved objects with database references and updated fields
        """
        if not objects:
            logger.info("No objects to resolve, returning empty list")
            return []

        try:
            logger.info(f"Resolving {len(objects)} objects")

            # Convert objects to dictionaries
            object_models = [self._object_to_model(obj) for obj in objects]
            logger.debug(f"Converted {len(object_models)} objects to models")

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

            # Process verification results
            processed_objects = await self._process_verification_results(
                object_models_with_embeddings, verification_results
            )
            logger.debug(f"Processed {len(processed_objects)} verification results")

            # Convert back to original objects
            resolved_objects = [
                self._model_to_object(processed.resolved, objects[i])
                for i, processed in enumerate(processed_objects)
            ]
            logger.info(f"Successfully resolved {len(resolved_objects)} objects")

            return resolved_objects
        except Exception as e:
            logger.exception(f"Error in Resolver.resolve: {str(e)}")
            # Return the original objects if there's an error
            return objects

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
        result = []
        n_objects = len(objects)

        if n_objects == 0:
            return result

        # Convert all embeddings to a numpy array for vectorized operations
        embeddings = np.array([obj.embedding for obj in objects], dtype=np.float32)

        # Compute all pairwise similarities in one go
        # Normalize the embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / np.maximum(
            norms, 1e-10
        )  # Avoid division by zero

        # Compute similarity matrix using dot product of normalized vectors
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

        # Set diagonal to -1 to exclude self-comparisons
        np.fill_diagonal(similarity_matrix, -1)

        # Find the most similar object for each input object
        for i, obj in enumerate(objects):
            # Get similarities for this object with all other input objects
            input_similarities = similarity_matrix[i]

            # Find the index of the most similar object and its similarity
            most_similar_idx = np.argmax(input_similarities)
            highest_input_similarity = input_similarities[most_similar_idx]

            # Only consider it if above threshold
            most_similar_input_obj = None
            if highest_input_similarity > self.similarity_threshold:
                similar_obj = objects[most_similar_idx]
                most_similar_input_obj = SimilarObject(
                    name=similar_obj.name,
                    description=similar_obj.description,
                    embedding=similar_obj.embedding,
                    reference_id=similar_obj.reference_id,
                    metadata=similar_obj.metadata,
                    similarity=float(highest_input_similarity),
                )

            # Find similar objects in the database
            similar_db_objects = await self.find_similar_by_embedding(
                self.conn,
                self.table_name,
                obj.embedding,
                self.similarity_threshold,
                limit=1,  # Only need the most similar one
            )

            # Get the most similar database object if any
            most_similar_db_obj = None
            highest_db_similarity = -1.0

            if similar_db_objects:
                db_obj = similar_db_objects[0]
                highest_db_similarity = db_obj.similarity
                most_similar_db_obj = db_obj

            # Determine which object is more similar
            if most_similar_input_obj and (
                highest_input_similarity > highest_db_similarity
            ):
                # Input object is more similar
                result.append(
                    ObjectPair(
                        input_object=obj,
                        similar_object=most_similar_input_obj,
                        is_from_input_list=True,
                    )
                )
            elif most_similar_db_obj:
                # Database object is more similar
                result.append(
                    ObjectPair(
                        input_object=obj,
                        similar_object=most_similar_db_obj,
                        is_from_input_list=False,
                    )
                )
            else:
                # No similar object found
                result.append(
                    ObjectPair(
                        input_object=obj, similar_object=None, is_from_input_list=False
                    )
                )

        return result

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
    ) -> List[ProcessedObject[ObjectWithEmbedding]]:
        """
        Process verification results and update/create objects.

        Args:
            objects: Original list of objects with embeddings
            verification_results: Results from LLM verification

        Returns:
            List of processed objects with database references
        """
        resolved_objects = []
        objects_to_update_embeddings = []
        descriptions_to_embed = []

        # Track which input objects have been processed
        processed_input_objects = set()

        # Create a mapping from input object name to its index in the resolved_objects list
        input_name_to_index = {}

        for result in verification_results:
            input_obj = result.input_object
            similar_obj = result.db_object  # This could be from DB or input list
            is_same = result.is_same
            is_from_input_list = result.is_from_input_list

            # Skip if this input object has already been processed
            if input_obj.name in processed_input_objects:
                continue

            processed_input_objects.add(input_obj.name)

            if is_same and similar_obj is not None:
                if is_from_input_list:
                    # Both objects are from the input list and are the same
                    # We need to add one to the database and link the other to it

                    # Check if the similar object has already been processed
                    if (
                        similar_obj.name in processed_input_objects
                        and similar_obj.name in input_name_to_index
                    ):
                        # The similar object has already been processed and added to the database
                        # Just link this object to the same database entry
                        similar_obj_index = input_name_to_index[similar_obj.name]
                        processed_obj = resolved_objects[similar_obj_index]
                        db_id = processed_obj.reference_id

                        resolved_obj = self._model_to_object(input_obj, input_obj)
                        # Update the object with the reference
                        resolved_obj_model = self._object_to_model(resolved_obj)
                        resolved_obj_model.reference_id = db_id

                        processed_obj = ProcessedObject(
                            original=input_obj,
                            resolved=resolved_obj_model,
                            reference_id=db_id,
                            is_new=False,
                            has_updates=False,
                        )
                        resolved_objects.append(processed_obj)
                        input_name_to_index[input_obj.name] = len(resolved_objects) - 1
                    else:
                        # Neither object has been processed yet
                        # Create a merged object with updated name/description if provided
                        updated_name = result.updated_name or input_obj.name
                        updated_description = (
                            result.updated_description or input_obj.description
                        )

                        merged_obj = ObjectWithEmbedding(
                            name=updated_name,
                            description=updated_description,
                            embedding=input_obj.embedding,
                            metadata=input_obj.metadata.copy(),
                        )

                        # Insert the merged object into the database
                        db_id = await self._insert_object_into_db(merged_obj)

                        # Update both objects with the reference
                        resolved_obj1 = ObjectWithEmbedding(
                            name=updated_name,
                            description=updated_description,
                            embedding=input_obj.embedding,
                            reference_id=db_id,
                            metadata=input_obj.metadata.copy(),
                        )

                        processed_obj = ProcessedObject(
                            original=input_obj,
                            resolved=resolved_obj1,
                            reference_id=db_id,
                            is_new=True,
                            has_updates=updated_name != input_obj.name
                            or updated_description != input_obj.description,
                        )

                        resolved_objects.append(processed_obj)
                        input_name_to_index[input_obj.name] = len(resolved_objects) - 1

                        # Mark the similar object as processed
                        processed_input_objects.add(similar_obj.name)

                        # If we need to update embeddings based on the merged data
                        if (
                            updated_name != input_obj.name
                            or updated_description != input_obj.description
                        ):
                            objects_to_update_embeddings.append(
                                (merged_obj, len(resolved_objects) - 1)
                            )
                            descriptions_to_embed.append(
                                f"{merged_obj.name}: {merged_obj.description}"
                            )
                else:
                    # Object from input list matches one from the database
                    # Objects are the same, update the existing object if needed
                    updated_name = result.updated_name
                    updated_description = result.updated_description

                    if updated_name or updated_description:
                        # Update the object in the database
                        updates = {}
                        if updated_name:
                            updates["name"] = updated_name
                        if updated_description:
                            updates["description"] = updated_description

                        # Track objects that need updated embeddings
                        if updated_name or updated_description:
                            # Store the object and its index for later embedding update
                            obj_to_update = ObjectWithEmbedding(
                                name=updated_name if updated_name else input_obj.name,
                                description=updated_description
                                if updated_description
                                else input_obj.description,
                                embedding=input_obj.embedding,
                                reference_id=similar_obj.reference_id,
                                metadata=input_obj.metadata.copy(),
                            )

                            objects_to_update_embeddings.append(
                                (obj_to_update, len(resolved_objects))
                            )
                            descriptions_to_embed.append(
                                f"{obj_to_update.name}: {obj_to_update.description}"
                            )

                        # Create a resolved object with the reference and updated values
                        resolved_obj = ObjectWithEmbedding(
                            name=updated_name if updated_name else input_obj.name,
                            description=updated_description
                            if updated_description
                            else input_obj.description,
                            embedding=input_obj.embedding,
                            reference_id=similar_obj.reference_id,
                            metadata=input_obj.metadata.copy(),
                        )

                        processed_obj = ProcessedObject(
                            original=input_obj,
                            resolved=resolved_obj,
                            reference_id=similar_obj.reference_id,
                            is_new=False,
                            has_updates=updated_name is not None
                            or updated_description is not None,
                        )

                        # We'll update the embedding later in batch
                        resolved_objects.append(processed_obj)
                        input_name_to_index[input_obj.name] = len(resolved_objects) - 1
                    else:
                        # No updates needed, just link to the existing object
                        resolved_obj = ObjectWithEmbedding(
                            name=input_obj.name,
                            description=input_obj.description,
                            embedding=input_obj.embedding,
                            reference_id=similar_obj.reference_id,
                            metadata=input_obj.metadata.copy(),
                        )

                        processed_obj = ProcessedObject(
                            original=input_obj,
                            resolved=resolved_obj,
                            reference_id=similar_obj.reference_id,
                            is_new=False,
                            has_updates=False,
                        )

                        resolved_objects.append(processed_obj)
                        input_name_to_index[input_obj.name] = len(resolved_objects) - 1
            else:
                # Objects are different or no similar object was found, create a new one
                db_id = await self._insert_object_into_db(input_obj)

                # Update the object with the reference
                resolved_obj = ObjectWithEmbedding(
                    name=input_obj.name,
                    description=input_obj.description,
                    embedding=input_obj.embedding,
                    reference_id=db_id,
                    metadata=input_obj.metadata.copy(),
                )

                processed_obj = ProcessedObject(
                    original=input_obj,
                    resolved=resolved_obj,
                    reference_id=db_id,
                    is_new=True,
                    has_updates=False,
                )

                resolved_objects.append(processed_obj)
                input_name_to_index[input_obj.name] = len(resolved_objects) - 1

        # Compute new embeddings in batch if needed
        if descriptions_to_embed:
            new_embeddings = await self.embedding_provider.embed(
                texts=descriptions_to_embed,
                truncation=True,
                embed_prompt=self.EMBED_PROMPT,
            )

            # Update objects with new embeddings
            for i, (obj_to_update, obj_index) in enumerate(
                objects_to_update_embeddings
            ):
                # Update the embedding in the resolved object
                resolved_objects[obj_index].resolved.embedding = new_embeddings[i]

                # Update the object in the database
                await self._update_object_in_db(
                    resolved_objects[obj_index].reference_id,
                    {"embedding": new_embeddings[i]},
                )

        # Check for any unprocessed input objects
        for obj in objects:
            if obj.name not in processed_input_objects:
                # Create a new database entry for this object
                db_id = await self._insert_object_into_db(obj)

                # Add to resolved objects
                resolved_obj = ObjectWithEmbedding(
                    name=obj.name,
                    description=obj.description,
                    embedding=obj.embedding,
                    reference_id=db_id,
                    metadata=obj.metadata.copy(),
                )

                processed_obj = ProcessedObject(
                    original=obj,
                    resolved=resolved_obj,
                    reference_id=db_id,
                    is_new=True,
                    has_updates=False,
                )

                resolved_objects.append(processed_obj)

        return resolved_objects

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
    ) -> List[SimilarObject]:
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

        # Convert rows to SimilarObject instances
        results = []
        for row in rows:
            similar_obj = SimilarObject(
                name=row["name"],
                description=row["description"],
                reference_id=row["id"],
                similarity=row["similarity"],
                # We don't retrieve the embedding here to save bandwidth
                embedding=[],
            )
            results.append(similar_obj)

        return results
