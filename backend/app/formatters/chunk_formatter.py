class ChunkFormatter:
    """
    Formats retrieved Legal chunks into a structured prompt context
    that helps Gemini write in the scholarly style.
    """

    def format_for_prompt(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            m = chunk["metadata"]

            law_name = m.get("law_title") or m.get("law_name") or m.get("source_file") or ""
            year     = m.get("year") or ""
            section  = m.get("section_title") or m.get("section") or ""

            parts.append(
                f"--- Source {i} ---\n"
                f"[Metadata] law_name: {law_name}, year: {year}, section_title: {section}\n"
                f"{chunk['document']}\n"
            )
        return "\n".join(parts)
