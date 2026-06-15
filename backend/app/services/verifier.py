from google import genai
from google.genai import types
from app.core.config import settings
from app.formatters.chunk_formatter import ChunkFormatter
import logging

logger = logging.getLogger("app.verifier")

class Verifier:
    """
    A dedicated verification service to check if the retrieved context 
    can actually answer the query BEFORE sending it to the heavy generation prompt.
    Mixes algorithmic checks with a lightweight LLM check to avoid complete AI dependence.
    """
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GENERATION_MODEL
        self.formatter = ChunkFormatter()

    async def verify(self, query_bn: str, query_en: str, chunks: list[dict]) -> bool:
        if not chunks:
            return False

        max_score = max((c.get("score", 0.0) for c in chunks), default=0.0)

        # Fast reject — completely off-topic
        if max_score < 0.40:
            logger.info(f"Verifier: Rejecting — max score too low ({max_score:.4f})")
            return False

        # Fast accept — clearly on-topic
        if max_score >= 0.75:
            logger.info(f"Verifier: Fast accepting ({max_score:.4f})")
            return True

        # Borderline: AI check — pass if ANY related content exists
        logger.info(f"Verifier: Borderline ({max_score:.4f}), running AI check...")
        context = self.formatter.format_for_prompt(chunks)
        prompt = (
            "You are a relevance filter for a legal RAG system.\n"
            "Read the user query and the retrieved legal context passages.\n"
            "Answer YES if the context contains ANY information related to the query topic "
            "(even partially — a relevant law, section, or concept is enough).\n"
            "Answer NO only if the context is completely unrelated to the query topic.\n"
            "Note: The generator will decide if the answer is specific enough — "
            "your job is only to check topic relevance.\n"
            "Reply with EXACTLY 'YES' or 'NO'. Do not explain.\n\n"
            f"Bengali Query: {query_bn}\n"
            f"English Query: {query_en}\n\n"
            f"Context:\n{context}"
        )

        try:
            res = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=5
                )
            )
            answer = res.text.strip().upper()
            logger.info(f"Verifier AI Result: {answer}")
            return "YES" in answer
        except Exception as e:
            logger.error(f"Verifier AI Error: {e}")
            return True
