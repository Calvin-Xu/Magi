RELATIONSHIP_EXTRACTION_PROMPT = """
Analyze the following text and extract relationships between entities. Your task is to extract relationship triples `subject-predicate-object` and return them in a structured JSON format.
For each relationship you find, extract the following:

1. subject
   - The name of the source entity.
2. subject_description
   - A globally unique, disambiguating description of the subject (including aliases, distinguishing attributes, etc.).
3. object
   - The name of the target entity.
4. object_description
   - A globally unique, disambiguating description of the object.
5. predicate
   - A short label describing the relationship between the subject and object. Use established relationship types (e.g., "is a", "part of", "has a", "has property", "created", etc.) where possible.
6. predicate_description
   - A globally unique identifying description for this predicate.
7. constraint_condition
   - A concise description of the conditions under which the relationship holds (if it does not universally hold) according to the text; for temporal constraints, include specific datetimes.
8. reason
   - A concise explanation, citing key excerpts from the text, to justify why this relationship is supported.
9. is_causal
   - Whether the relationship is causal and between random variables. Answer with "yes" or "no".

Additional Instructions:
- Semantic Clarity: Subject-predicate-object should form a coherent sentence, though elements like articles can be dropped for brevity.
- Finding Causal Relationships: Pay special attention to causal relationships and their directionality as indicated by the text.
- Predicate Management: Use standard predicates instead of creating new ones unnecessarily. Use concise custom predicates only when no standard term applies.
- Confidence: Only extract relationships you can confidently infer from the text.
- Output Format: Return a JSON object with a key `"relationships"` whose value is an array of relationship objects, each containing the keys specified above.

Example Outputs:
```json
{{
  "relationships": [
    {{
      "subject": "Gaius Julius Caesar Augustus",
      "subject_description": "Gaius Julius Caesar Augustus (born Gaius Octavius; 63 BC - AD 14), also known as Octavian, was the first Roman emperor and founder of the Principate. His reign (27 BC - AD 14) initiated the Pax Romana. He defeated Mark Antony and Cleopatra, reformed governance, expanded the empire, and was succeeded by Tiberius.",
      "object": "Roman Empire",
      "object_description": "The Roman Empire (27 BC - 476 AD West, 1453 AD East), also known as Imperium Romanum, ruled Europe, North Africa, and Western Asia. Founded by Augustus succeeding the Roman Republic, it shaped law, language, government, and Christianity. The Western Empire fell in 476 AD, while the Byzantine Empire lasted until Constantinople's fall in 1453.",
      "predicate": "founded",
      "predicate_description": "Founded indicates that the subject was responsible for establishing or creating the object.",
      "conditional_constraint": "in 27 BC",
      "reason": "The text states that Gaius Julius Caesar Augustus, known as Octavian, was the founder of the Roman Empire and became its first emperor in 27 BC."
    }},
  ]
}}
```
```json
{{
  "relationships": [
    {{
      "subject": "alkaline pH",
      "subject_description": "Alkaline pH refers to a hydrogen ion concentration above neutral (pH > 7), characterized by a lower presence of protons (H⁺) in an aqueous solution.",
      "object": "TRPV5/V6_channels: activity",
      "object_description": "The TRPV5 and TRPV6 (transient receptor potential vanilloid type 5 and 6) channels are highly Ca²⁺-selective ion channels primarily involved in calcium absorption and homeostasis.",
      "predicate": "increases",
      "predicate_description": "Increases indicates that the subject enhances or promotes the function, activity, or efficacy of the object.",
      "conditional_constraint": "The relationship holds in intact Jurkat T cells with functional intracellular trafficking. The text states, “alkalization of the solution did not affect TRPV5/V6 channel activity in isolated patches, though it did increase channel activity in intact cells.” Experiments were conducted at room temperature (22 - 23°C) between 2013 and 2016.",
      "reason": "Alkaline pH increases TRPV5/V6 channel activity by promoting their plasma membrane delivery, enhancing Ca²⁺ influx in Jurkat T cells. The text states, “extracellular alkalization results in a rapid elevation of the TRPV5 delivery rate to the plasma membrane” and “alkalization of the external solution in Jurkat T cells resulted in a significant increase in [Ca²⁺]i.”",
      "is_causal": "yes"
    }}
  ]
}}
```

Text to Analyze:
{text}
"""
