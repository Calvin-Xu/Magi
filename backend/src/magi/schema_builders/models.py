"""Pydantic models for schema extraction from datasets."""

from pydantic import BaseModel, Field
from typing import Dict, Optional, Union


class PropertySchema(BaseModel):
    """Schema for a table property."""

    description: str = Field(
        ..., description="Detailed description of what this property represents"
    )
    reference: Optional[Union[str, bool]] = Field(
        None,
        description="Name of the referenced table if this is a foreign key, or false if it's not",
    )
    type: str = Field(
        "string", description="Data type of the property (string, number, boolean, etc.)"
    )
    is_primary_key: bool = Field(
        False, description="Indicates whether this property is a primary key"
    )
    
    def get_reference_name(self) -> Optional[str]:
        """Get the reference name if it's a string, otherwise return None."""
        return self.reference if isinstance(self.reference, str) else None
    
    @property
    def references(self) -> Optional[str]:
        """Alias for reference that returns a string or None."""
        return self.get_reference_name()


class TableSchema(BaseModel):
    """Schema for a database table."""

    properties: Dict[str, PropertySchema] = Field(
        ..., description="Dictionary of property names to their schemas"
    )
    description: str = Field(
        ...,
        description="Detailed description of what this table represents in the dataset",
    )


class RelationalDatasetSchema(BaseModel):
    """Complete schema for a relational dataset with multiple tables."""

    tables: Dict[str, TableSchema] = Field(
        ..., description="Dictionary of table names to their schemas"
    )
