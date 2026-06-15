import time
from google import genai
from google.genai import errors
from app.core.config import settings

class Embedder:
    """
    Handles embedding generation using Gemini API with rate-limiting support.
    """
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        batch_size = 50  # Safe batch size for Gemini API
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            retries = 6
            
            for attempt in range(retries):
                try:
                    response = self.client.models.embed_content(
                        model=self.model_name,
                        contents=batch_texts,
                    )
                    all_embeddings.extend([e.values for e in response.embeddings])
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower() or "exhausted" in err_str.lower():
                        wait_time = 10 * (2 ** attempt)
                        print(f"Rate limited (429) on batch {i//batch_size}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"Embedder error: {e}")
                        if attempt == retries - 1:
                            # Fill with zeros or handle gracefully if completely failed?
                            # For RAG, raising is better to prevent corrupted DB, but 
                            raise e
                        time.sleep(5)
                        
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
        )
        return response.embeddings[0].values
