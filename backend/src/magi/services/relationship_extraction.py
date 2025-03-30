"""Relationship extraction service for document batches."""

import asyncio
import json
from typing import AsyncIterator

from magi.extractors.base import RelationshipExtractor
from magi.services.schemas import DocumentBatch, TextDocument
from magi.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


class RelationshipExtractionService:
    """Service for extracting relationships from document batches."""

    def __init__(self, extractor: RelationshipExtractor, max_concurrent: int = 100):
        """
        Initialize the relationship extraction service.

        Args:
            extractor: The relationship extractor to use
        """
        self.extractor = extractor
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def extract_relationships_from_text(self, text: str) -> list[dict]:
        """
        Extract relationships from text using the provided extractor.

        Args:
            text: The text to extract relationships from

        Returns:
            A list of relationship dictionaries
        """
        relationships = []
        async for rel in self.extractor.extract_relationships(text):
            relationships.append(rel.model_dump())
        return relationships

    async def extract_relationships_from_document(
        self, document: TextDocument
    ) -> TextDocument:
        """
        Extract relationships from a single document.

        Args:
            document: The document to extract relationships from

        Returns:
            The document with relationships_json field populated
        """
        async with self.semaphore:
            try:
                relationships = await self.extract_relationships_from_text(
                    document.content
                )
                document.relationships_json = json.dumps(relationships)
            except Exception as e:
                logger.error(
                    f"Error extracting relationships from {document.uri}: {str(e)}"
                )
                document.relationships_json = json.dumps([])
        return document

    async def process_batch(self, batch: DocumentBatch) -> DocumentBatch:
        """
        Process a batch of documents to extract relationships.

        Args:
            batch: The batch of documents to process

        Returns:
            The batch with relationships extracted
        """
        if not batch.documents:
            return batch

        # Longer documents start processing first
        docs_by_length = sorted(
            batch.documents, key=lambda doc: len(doc.content), reverse=True
        )

        tasks = {}
        for doc in docs_by_length:
            task = asyncio.create_task(self.extract_relationships_from_document(doc))
            tasks[doc.uri] = (doc, task)

        processed_documents = []
        for completed_task in asyncio.as_completed(
            [task for _, task in tasks.values()]
        ):
            processed_doc = await completed_task
            processed_documents.append(processed_doc)

            logger.debug(
                f"Processed {len(processed_documents)}/{len(batch.documents)} documents"
            )

        return DocumentBatch(documents=processed_documents)

    async def process_batches(
        self, batches: AsyncIterator[DocumentBatch]
    ) -> AsyncIterator[DocumentBatch]:
        """
        Process document batches to extract relationships.

        Args:
            batches: An async iterator of document batches

        Yields:
            Processed document batches with relationships extracted
        """
        async for batch in batches:
            processed_batch = await self.process_batch(batch)
            yield processed_batch
