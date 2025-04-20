"""Prompt templates for schema extraction."""

# prompts.py
SCHEMA_EXTRACTION_PROMPT = """
You are an expert data scientist tasked with analyzing a relational dataset and creating a schema graph.

You will receive information about **one table at a time**, including:
- The total number of columns in the table
- The specific subset (chunk) of columns you are to analyze in this round
- The first one or two rows for those columns
- Any user-supplied context or support documents relevant to understanding the dataset

**Important**: You do NOT see all columns at once, only the current subset. Some columns might reference others not visible in this chunk. If you need to reference them, do so generically (e.g., "potentially references a table or column not currently shown"), or if you already know the table name from user context, you can indicate that.

Your goal is to propose partial schema information for the chunk of columns shown. Specifically:
1. For each column in this chunk, provide:
   - A short but globally meaningful description
   - Data type
   - Whether it is a primary key
   - Whether it references another table (if you can infer from context)
2. You may also propose additional insights gleaned from the user context or support documents, but keep them limited to the scope of these columns.

**Output Format**:

Return **only** JSON following this structure:

```json
{{
  "properties": {{
    "column_name": {{
      "description": "Detailed description of the column",
      "type": "string|number|boolean|etc.",
      "is_primary_key": false,
      "reference": false
    }},
    "another_column": {{
      "description": "Detailed description of the column",
      "type": "string|number|boolean|etc.",
      "is_primary_key": false,
      "reference": "NameOfReferencedTableOrFalse"
    }}
  }}
}}
```

Note: You are NOT returning the entire table schema in this chunk—only for the columns shown.

If you cannot infer something, leave a best guess or mark it as false for reference, etc.

Any textual explanation outside of this JSON structure will be ignored.

Now, here is your chunk to analyze:

CHUNK INFO:
TOTAL COLUMNS: {total_columns}
CHUNK RANGE: {chunk_start_index}-{chunk_end_index}
TABLE NAME: {table_name}

COLUMNS IN THIS CHUNK (and two rows of data):
{table_chunk_sample}

{user_prompt}

{support_documents}
"""
