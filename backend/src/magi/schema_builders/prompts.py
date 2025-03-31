"""Prompt templates for schema extraction."""

SCHEMA_EXTRACTION_PROMPT = """
You are an expert data scientist tasked with analyzing a relational dataset and creating a schema graph.

Your goal is to extract tables, properties, and relationships from the dataset to build a structured schema. Use the dataset headers and the provided context to understand the data structure.

# Dataset Files
{dataset_files}

# User Context
{user_prompt}

# Attached Support Documents
{support_documents}

# Instructions

1. Identify each table in the dataset
2. For each table, identify all properties (columns)
3. Determine which properties are foreign keys referencing other tables
4. Include detailed, globally-identifying descriptions for each table and property
5. Make sure descriptions specifically mention the dataset name they are associated with

# Output Format

Provide your analysis as a structured JSON with the following format:

```json
{
    "tables": {
        "TableName1": {
            "properties": {
                "property1": {
                    "description": "Detailed description of property1",
                    "reference": false
                },
                "property2": {
                    "description": "Detailed description of property2",
                    "reference": "ReferencedTableName"
                }
            },
            "description": "Detailed description of TableName1"
        },
        "TableName2": {
            "properties": {
                // Additional properties...
            },
            "description": "Detailed description of TableName2"
        }
    }
}
```

For each property, set "reference" to the name of the referenced table if it's a foreign key, or to false if it's not.

Do not include any explanations outside of the JSON structure.
"""
