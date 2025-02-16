RELATIONSHIP_PROMPT = """
Analyze the following text and extract relationships between entities. Your task is to extract relationship triples and return them in a structured JSON format.
For each relationship you find, extract the following:

1. **subject**  
   - The name of the source entity.  
2. **subject_description**  
   - A globally unique, disambiguating description of the subject (including aliases, distinguishing attributes, etc.).  
3. **object**  
   - The name of the target entity.  
4. **object_description**  
   - A globally unique, disambiguating description of the object.
5. **predicate**  
   - A short label describing the relationship between the subject and object. Use established relationship types (e.g., "is a", "part of", "has a", "has property", "created", etc.) where possible.  
6. **predicate_description**  
   - A globally unique identifying description for this predicate. It should clarify the semantics of the relationship (e.g., "Indicates that the subject is a specific instance or subtype of the object.").  
7. **constraint_condition**  
   - A concise description of the conditions under which the predicate holds (if it does not universally hold), including temporal constraints
8. **reason**  
   - A concise explanation, citing key excerpts from the text, to justify why this relationship is supported.
9. **is_causal**  
   - Whether the relationship is causal and between random variables. Answer with "yes" or "no".

**Additional Instructions:**
- **Finding Causal Relationships:** Pay special attention to causal relationships and their directionality as indicated by the text.
- **Predicate Management:** Use standard predicates instead of creating new ones unnecessarily. Use concise custom predicates only when no standard term applies.
- **Confidence:** Only extract relationships you can confidently infer from the text.
- **Output Format:** Return a JSON object with a key `"relationships"` whose value is an array of relationship objects, each containing the keys specified above.

**Example Output:**
```json
{{
  "relationships": [
    {{
      "subject": "extracellular_acidic_pH",
      "subject_description": "A condition in which the pH of the extracellular environment is lower than normal, leading to an increase in proton concentration.",
      "object": "TRPV5/V6_channels: activity",
      "object_description": "The functional state of TRPV5 and TRPV6, which are calcium-selective ion channels involved in calcium homeostasis.",
      "predicate": "reduces",
      "predicate_description": "Indicates that the subject decreases the function, activity, or efficacy of the object.",
      "conditional_constraint": "",
      "reason": "According to the study, extracellular acidic pH was experimentally demonstrated to reduce the activity of TRPV5/V6 channels, suggesting an inhibitory effect under such conditions.",
      "is_causal": "yes"
    }},
    {{
      "subject": "alkaline_pH",
      "subject_description": "A condition in which the pH of the extracellular or intracellular environment is higher than neutral, indicating a lower concentration of hydrogen ions.",
      "object": "TRPV5/V6_channels: activity",
      "object_description": "The functional state of TRPV5 and TRPV6, which are calcium-selective ion channels involved in calcium homeostasis.",
      "predicate": "increases",
      "predicate_description": "Indicates that the subject enhances or promotes the function, activity, or efficacy of the object.",
      "conditional_constraint": "inside Jurkat_T_cells",
      "reason": "The study demonstrated that alkaline pH conditions specifically within Jurkat T cells increased TRPV5/V6 channel activity, suggesting a pH-dependent modulation in this cellular environment.",
      "is_causal": "yes"
    }}
  ]
}}
```

**Text to Analyze:**  
{text}
"""
