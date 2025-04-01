"""Prompts for knowledge graph augmentation.

This module defines the prompts used by the Perplexity augmenter to guide
the research process and structure the results.
"""

from typing import Optional


def research_system_prompt() -> str:
    """System prompt for guiding research behavior.

    Returns:
        System prompt string for the research API
    """
    return (
        "You are a research assistant that specializes in discovering relationships "
        "between entities in datasets, with a focus on causal relationships. "
        "Your task is to analyze dataset schemas and conduct deep research on diverse sources to identify "
        "meaningful connections between variables, particularly those with causal implications. "
        "Use your research to introduce NEW relevant entities that expand our understanding "
        "of the domain that are connected to the existing entities and among themselves. "
        "This creates a hybrid schema-knowledge graph that combines "
        "dataset-specific information with broader domain knowledge. "
        "Always cite your sources and provide confidence levels for your findings. "
        "You should format your response as a structured JSON object."
    )


def research_user_prompt(
    context: str,
    user_instruction: Optional[str] = None,
) -> str:
    """Generate a user prompt for research based on schema context.

    Args:
        context: Schema context describing tables, properties, and known relationships
        user_instruction: Optional user guidance for research focus

    Returns:
        Formatted user prompt for the research API
    """
    default_instruction = (
        "Conduct thorough research to identify relationships between entities in this dataset "
        "and introduce new domain-relevant entities that expand our understanding. "
        "You should identify both relationships between existing entities AND relationships "
        "that connect existing entities to new, domain-relevant entities not present in the dataset. "
        "Three categories of relationships triples are allowed:\n"
        "1. <universal -[predicate]- universal> (human - is a - mammal)\n"
        "2. <instance -[predicate]- universal> (Socrates - is a - human)\n"
        "3. <instance -[predicate]- instance> (Socrates - taught - Plato)\n"
        "Instances must be named and identifiable outside the original context; for example, 'Triple Entente - triggered - World War I' is valid and 'the alliance - triggered - war' is not. Do not extract a <universal -[predicate]- universal> triple from a quote about historical instances.\n"
        "For each relationship you identify:\n"
        "1. subject\n"
        "   - A canonical name of the subject you identify; use this name consistently for all relationships\n"
        "2. subject_description\n"
        "   - A globally unique, disambiguating description of the subject (including aliases, distinguishing attributes, etc.) that begins with subject's name; instance names must still be specifically identifiable despite the description.\n"
        "   - Compose descriptions using both your best knowledge of the subject and information from the text\n"
        "3. object\n"
        "4. object_description\n"
        "5. predicate\n"
        "   - A short label describing the relationship between the subject and object. Use established relationship types (e.g., 'is a', 'part of', 'has a', 'has property', 'created', etc.) where possible.\n"
        "   - Use the present tense only for relationships that will always be true, now or in the future.\n"
        "6. predicate_description\n"
        "   - A concise, standardized description of the abstracted type of relationship between the subject and object as indicated by the predicate.\n"
        "   - It must accurately characterize this relationship, and also be reusable by other relationships in possibly different contexts.\n"
        "7. constraint_condition\n"
        "   - A concise description of the conditions under which the statement formed by `subject-predicate-object` holds if it is not always true (in which case write 'None'); for temporal constraints, include the most datetimes available;\n"
        "8. reason\n"
        "   - A concise explanation, citing key excerpts from the text, to justify why this relationship is supported.\n"
        "9. is_causal\n"
        "   - Whether the relationship should appear in a causal graph in the context of causal inference. Answer with 'yes' or 'no'.\n"
        "10. confidence\n"
        "   - A confidence level between 0 and 1, indicating the strength of your belief in the relationship.\n"
        "11. source_uri\n"
        "   - Citation URI for the relationship.\n"
        "Additional Guidelines:\n"
        "- Semantic Clarity: Subject-predicate-object should form a coherent sentence, though elements like articles can be dropped for brevity. Indirect objects can be included in the predicate if absolutely necessary.\n"
        "  - a reference using 'the' must either be resolved to a specific named entity\n"
        "- Predicate Management: Use standard predicates instead of quoting the source text and creating new ones unnecessarily. Prefer single verbs and only use concise custom predicates when no standard term applies.\n"
        "- Confidence: Only extract relationships you can confidently infer from the text.\n"
        "- Validation: Refer to your subject description and object description to determine whether they are univeral or instance entities. Avoid triples in the form of <universal -[predicate]- instance>.\n"
        "Structure your response as a JSON object with 'relationships' and 'summary' fields."
    )

    instruction = user_instruction or default_instruction

    return f"## SCHEMA CONTEXT\n\n{context}\n\n## INSTRUCTIONS\n\n{instruction}"
