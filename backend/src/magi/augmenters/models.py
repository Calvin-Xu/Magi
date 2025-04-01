"""Models for knowledge graph augmentation.

This module defines data models used by the graph augmentation process, including
representations of API responses and intermediate data structures.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ResearchRelationship(BaseModel):
    """A relationship discovered through research."""

    subject: str
    subject_description: str
    object: str
    object_description: str
    predicate: str
    predicate_description: str
    constraint_condition: Optional[str] = None
    reason: str
    is_causal: bool
    confidence: float = 0.0
    source_uri: str


class ResearchResponse(BaseModel):
    """Response from research containing discovered relationships."""

    relationships: List[ResearchRelationship] = []
    summary: Optional[str] = None


class SchemaEntityInfo(BaseModel):
    """Information about an entity in the schema graph."""

    name: str = Field(..., description="Name of the entity")
    description: str = Field(..., description="Description of the entity")
    is_property: bool = Field(
        False, description="Whether this entity represents a property"
    )
    parent_table: Optional[str] = Field(
        None, description="Parent table if this is a property"
    )

    @property
    def formatted(self) -> str:
        """Format the entity info as a string for context."""
        if self.is_property and self.parent_table:
            return f"Property: {self.name}\nParent Table: {self.parent_table}\nDescription: {self.description}"
        else:
            return f"Table: {self.name}\nDescription: {self.description}"


class SchemaContext(BaseModel):
    """Context about the schema for research purposes."""

    tables: List[SchemaEntityInfo] = Field(default_factory=list)
    properties: List[SchemaEntityInfo] = Field(default_factory=list)
    relationships: List[str] = Field(default_factory=list)
    domain_description: Optional[str] = Field(None)

    def format_context(self) -> str:
        """Format the schema context as a string for the LLM."""
        lines = ["# DATASET SCHEMA INFORMATION"]

        if self.domain_description:
            lines.append("\n## Domain Description")
            lines.append(self.domain_description)

        # Group tables with their properties
        lines.append("\n## Tables and Properties")

        # Create a mapping of tables to their properties
        table_properties = {}
        for prop in self.properties:
            if prop.parent_table not in table_properties:
                table_properties[prop.parent_table] = []
            table_properties[prop.parent_table].append(prop)

        # Output tables with their properties
        for table in self.tables:
            lines.append(f"\n### Table: {table.name}")
            lines.append(f"Description: {table.description}")

            # Add properties for this table
            if table.name in table_properties:
                lines.append("\nProperties:")
                for prop in table_properties[table.name]:
                    lines.append(f"- {prop.name.split('.')[-1]}: {prop.description}")

        if self.relationships:
            lines.append("\n## Known Relationships")
            for rel in self.relationships:
                lines.append(f"- {rel}")

        return "\n".join(lines)
