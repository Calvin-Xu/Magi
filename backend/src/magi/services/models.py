from dataclasses import dataclass, field
from typing import List, Optional
from pyspark.sql.types import BooleanType, StringType, StructField, StructType


@dataclass
class Entity:
    name: str
    description: str
    embedding: Optional[List[float]] = field(default_factory=list)
    postgres_reference: Optional[int] = None
    hash_key: Optional[str] = None
    from_imported_schema: bool = False

    # Define DataFrame column names as class variables
    NAME_COLUMN: str = "name"
    DESCRIPTION_COLUMN: str = "description"
    EMBEDDING_COLUMN: str = "embedding"
    POSTGRES_REFERENCE_COLUMN: str = "postgres_reference"
    FROM_IMPORTED_SCHEMA_COLUMN: str = "from_imported_schema"

    def __hash__(self):
        return hash((self.name, self.description))

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.name == other.name and self.description == other.description


@dataclass
class RelationshipType:
    name: str
    description: str
    embedding: Optional[List[float]] = field(default_factory=list)
    postgres_reference: Optional[int] = None
    hash_key: Optional[str] = None
    from_imported_schema: bool = False

    # Define DataFrame column names as class variables
    NAME_COLUMN: str = "name"
    DESCRIPTION_COLUMN: str = "description"
    EMBEDDING_COLUMN: str = "embedding"
    POSTGRES_REFERENCE_COLUMN: str = "postgres_reference"
    RELATIONSHIP_TYPE_HASH_COLUMN: str = "relationship_type_hash"
    FROM_IMPORTED_SCHEMA_COLUMN: str = "from_imported_schema"


@dataclass
class Relationship:
    from_entity: Entity
    to_entity: Entity
    relationship_type: RelationshipType
    constraint_condition: Optional[str] = None
    reason: Optional[str] = None
    is_causal: bool = False
    source_document_uri: Optional[str] = None
    from_imported_schema: bool = False
    confidence: Optional[float] = None

    # Define DataFrame column names as class variables
    FROM_ENTITY_COLUMN: str = "from_entity"
    TO_ENTITY_COLUMN: str = "to_entity"
    RELATIONSHIP_TYPE_COLUMN: str = "relationship_type"
    FROM_ENTITY_DESCRIPTION_COLUMN: str = "from_entity_description"
    TO_ENTITY_DESCRIPTION_COLUMN: str = "to_entity_description"
    RELATIONSHIP_TYPE_DESCRIPTION_COLUMN: str = "relationship_description"
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
    FROM_IMPORTED_SCHEMA_COLUMN: str = "from_imported_schema"
    CONFIDENCE_COLUMN: str = "confidence"


# Schema for relationship triples in Spark
RELATIONSHIP_SCHEMA = StructType(
    [
        StructField("from_entity", StringType(), False),
        StructField("from_entity_description", StringType(), False),
        StructField("to_entity", StringType(), False),
        StructField("to_entity_description", StringType(), False),
        StructField("relationship_type", StringType(), False),
        StructField("relationship_description", StringType(), False),
        StructField("constraint_condition", StringType(), True),
        StructField("reason", StringType(), False),
        StructField("is_causal", BooleanType(), False),
        StructField("source_document_uri", StringType(), True),
    ]
)
