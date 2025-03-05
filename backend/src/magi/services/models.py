from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Entity:
    name: str
    description: str
    embedding: Optional[List[float]] = field(default_factory=list)
    postgres_reference: Optional[str] = None

    # Define column names as class variables
    NAME_COLUMN: str = "name"
    DESCRIPTION_COLUMN: str = "description"
    EMBEDDING_COLUMN: str = "embedding"
    POSTGRES_REFERENCE_COLUMN: str = "postgres_reference"


@dataclass
class RelationshipType:
    name: str
    description: str
    embedding: Optional[List[float]] = field(default_factory=list)
    postgres_reference: Optional[str] = None

    # Define column names as class variables
    NAME_COLUMN: str = "name"
    DESCRIPTION_COLUMN: str = "description"
    EMBEDDING_COLUMN: str = "embedding"
    POSTGRES_REFERENCE_COLUMN: str = "postgres_reference"
    RELATIONSHIP_TYPE_HASH_COLUMN: str = "relationship_type_hash"


@dataclass
class Relationship:
    from_entity: Entity
    to_entity: Entity
    relationship_type: RelationshipType
    constraint_condition: Optional[str] = None
    reason: Optional[str] = None
    is_causal: bool = False
    source_document_uri: Optional[str] = None

    # Define column names as class variables
    FROM_ENTITY_COLUMN: str = "from_entity"
    TO_ENTITY_COLUMN: str = "to_entity"
    RELATIONSHIP_TYPE_COLUMN: str = "relationship_type"
    FROM_ENTITY_DESCRIPTION_COLUMN: str = "from_entity_description"
    TO_ENTITY_DESCRIPTION_COLUMN: str = "to_entity_description"
    RELATIONSHIP_DESCRIPTION_COLUMN: str = "relationship_description"
    CONSTRAINT_CONDITION_COLUMN: str = "constraint_condition"
    REASON_COLUMN: str = "reason"
    IS_CAUSAL_COLUMN: str = "is_causal"
    SOURCE_DOCUMENT_URI_COLUMN: str = "source_document_uri"
    FROM_ENTITY_HASH_COLUMN: str = "from_entity_hash"
    TO_ENTITY_HASH_COLUMN: str = "to_entity_hash"
    RELATIONSHIP_TYPE_HASH_COLUMN: str = "relationship_type_hash"
    FROM_ENTITY_REFERENCE_COLUMN: str = "from_entity_reference"
    TO_ENTITY_REFERENCE_COLUMN: str = "to_entity_reference"
    RELATIONSHIP_TYPE_REFERENCE_COLUMN: str = "relationship_type_reference"
