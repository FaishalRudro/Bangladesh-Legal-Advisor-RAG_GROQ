METADATA_EXTRACTION_PROMPT = """
You are analyzing a Bangladesh legal document (Act, Ordinance, Rule, or Judgment).
Read the provided text sample and extract the following metadata as JSON.

Return ONLY valid JSON, no explanation:

{
  "law_name": "<full title or Act name in original language>",
  "year": "<year the law was passed or enacted, as a string>",
  "link": "<url link if present, else empty string>",
  "repealed": "<'REPEALED' if the text indicates it is repealed, else 'ACTIVE'>",
  "language": "<primary language: 'english', 'bengali', or 'mixed'>"
}

CRITICAL RULES:
1. "year" MUST be a string (e.g., "1972", "2023"). If year is not found, use "".
2. Repealed status detection: look for patterns like [REPEALED: ...] or রহিত.
3. If a field cannot be determined, use "" — NEVER guess.
4. Return only raw JSON, no markdown code fences.
"""
