RELATIONSHIP_EXTRACTION_PROMPT = """
You are an AI assistant specialized in extracting identifying specific entities and their relationships in context.
You would analyze a text, extract relationship triples `subject-predicate-object` and return them in a structured JSON format.

Three categories of relationships triples are allowed:
1. <universal -[predicate]- universal> ("human - is a - mammal")
2. <instance -[predicate]- universal> ("Socrates - is a - human")
3. <instance -[predicate]- instance> ("Socrates - taught - Plato")

Instances must be named and identifiable outside the orignal context; for example, "Triple Entente - triggered - World War I" is valid and "the alliance - triggered - war" is not. Do not extract a <universal -[predicate]- universal> triple from a quote about historical instances.

For each relationship, extract the following information:
1. subject
   - A canonical name of the subject you identify; use this name consistently for all relationships
2. subject_description
   - A globally unique, disambiguating description of the subject (including aliases, distinguishing attributes, etc.) that begins with subject's name; instance names must still be specifically identifiable despite the description.
   - Compose descriptions using both your best knowledge of the subject and information from the text
3. object
4. object_description
5. predicate
   - A short label describing the relationship between the subject and object. Use established relationship types (e.g., "is a", "part of", "has a", "has property", "created", etc.) where possible.
   - Use the present tense only for relationships that will always be true, now or in the future.
6. predicate_description
   - A concise, standardized description of the abstracted type of relationship between the subject and object as indicated by the predicate.
   - It must be accurate to this relationship, and also maximally general and reusable by other relationships in possibly different contexts.
7. constraint_condition
   - A concise description of the conditions under which the statement formed by `subject-predicate-object` holds if it is not always true (in which case write "None"); for temporal constraints, include the most datetimes available;
8. reason
   - A concise explanation, citing key excerpts from the text, to justify why this relationship is supported.
9. is_causal
   - Whether the relationship describes a causal link between random variables and is supported by quantitative evidence in the text. Answer with "yes" or "no".

Additional Guidelines:
- Thoroughness: Extract as many accurate relationships as possible.
- Semantic Clarity: Subject-predicate-object should form a coherent sentence, though elements like articles can be dropped for brevity. Indirect objects can be included in the predicate if absolutely necessary.
  - a reference using "the" must either be resolved to a specific named entity
- Predicate Management: Use standard predicates instead of quoting the source text and creating new ones unnecessarily. Prefer single verbs and only use concise custom predicates when no standard term applies.
- Confidence: Only extract relationships you can confidently infer from the text.
- Validation: Refer to your subject description and object description to determine whether they are univeral or instance entities. Avoid triples in the form of <universal -[predicate]- instance>.

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
      "reason": "Stated that \"Gaius Julius Caesar Augustus, known as Octavian, was the founder of the Roman Empire and became its first emperor in 27 BC.\""
      "is_causal": "no"
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
      "conditional_constraint": "The relationship holds in intact Jurkat T cells with functional intracellular trafficking: \"alkalization of the solution did not affect TRPV5/V6 channel activity in isolated patches, though it did increase channel activity in intact cells.\" Experiments were conducted at room temperature (22 - 23°C) between 2013 and 2016.",
      "reason": "Stated \"extracellular alkalization results in a rapid elevation of the TRPV5 delivery rate to the plasma membrane\" and \"alkalization of the external solution in Jurkat T cells resulted in a significant increase in [Ca²⁺]i.\"",
      "is_causal": "yes"
    }}
  ]
}}
```

Text to Analyze:
{text}
"""
