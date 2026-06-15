from app.schemas.chat_schema import ChatResponse, Source

class ScholarlyResponseFormatter:
    """
    Validates and enriches the LLM's scholarly response.
    Does NOT reformat the answer — preserves the scholarly prose.
    Only adds source metadata and confidence scoring.
    """

    def format(
        self,
        llm_answer: str,
        retrieved_chunks: list[dict],
        query_lang: str,
        legal_query: str,
    ) -> ChatResponse:

        sources = self._build_sources(retrieved_chunks)
        confidence = self._score_confidence(retrieved_chunks)

        return ChatResponse(
            answer=llm_answer,           # preserved as-is from Gemini
            sources=sources,
            confidence=confidence,
            query_language=query_lang,
            legal_query=legal_query,
            laws_searched=list({s.law_name for s in sources}),
            total_chunks_retrieved=len(retrieved_chunks),
        )

    def _build_sources(self, chunks) -> list[Source]:
        sources = []
        for chunk in chunks:
            m = chunk["metadata"]
            score = chunk.get("score", 0.0)
            sources.append(Source(
                law_name=m.get("law_name") or m.get("law_title") or m.get("source_file", ""),
                year=m.get("year", ""),
                link=m.get("link", ""),
                repealed=m.get("repealed", ""),
                relevance_score=round(score, 3),
                relevance_label=(
                    "high" if score >= 0.75 else
                    "medium" if score >= 0.50 else
                    "low"
                ),
                context_text=chunk.get("document", "")
            ))
        return sources

    def _score_confidence(self, chunks) -> str:
        if not chunks:
            return "not_found"
        max_score = max(c.get("score", 0.0) for c in chunks)
        if max_score >= 0.75:
            return "high"
        if max_score >= 0.50:
            return "medium"
        return "low"
