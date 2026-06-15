import asyncio
from google import genai
import chromadb
from app.core.config import settings
from app.services.embedder import Embedder
from app.prompts.language_prompt import LANGUAGE_DETECTION_PROMPT
from app.prompts.translation_prompt import TRANSLATION_PROMPT_BN_TO_EN, TRANSLATION_PROMPT_EN_TO_BN
from app.prompts.query_expansion_prompt import QUERY_EXPANSION_PROMPT

class Retriever:
    def __init__(self, db: chromadb.Client):
        self.db = db
        self.collection = db.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = Embedder()
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GENERATION_MODEL

    async def detect_language(self, query: str) -> str:
        prompt = LANGUAGE_DETECTION_PROMPT + "\n\nQuery: " + query
        res = await self.client.aio.models.generate_content(model=self.model_name, contents=prompt)
        lang = res.text.strip().lower()
        if lang not in ["bn", "en"]:
            return "other"
        return lang

    async def translate_query(self, query: str, source_lang: str) -> str:
        prompt = TRANSLATION_PROMPT_BN_TO_EN if source_lang == "bn" else TRANSLATION_PROMPT_EN_TO_BN
        res = await self.client.aio.models.generate_content(model=self.model_name, contents=prompt + "\n" + query)
        return res.text.strip()

    async def expand_query(self, query: str) -> str:
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        res = await self.client.aio.models.generate_content(model=self.model_name, contents=prompt)
        return res.text.strip()

    async def retrieve_bilingual(self, query_bn: str, query_en: str, law_id: str = None, top_k: int = settings.TOP_K_RESULTS) -> list[dict]:
        # Expand queries concurrently
        expand_bn_task = self.expand_query(query_bn)
        expand_en_task = self.expand_query(query_en)
        
        expanded_bn, expanded_en = await asyncio.gather(expand_bn_task, expand_en_task)
        
        # Embed both expanded queries concurrently
        emb_bn_task = asyncio.to_thread(self.embedder.embed_query, expanded_bn)
        emb_en_task = asyncio.to_thread(self.embedder.embed_query, expanded_en)
        
        bn_embedding, en_embedding = await asyncio.gather(emb_bn_task, emb_en_task)
        
        where_clause = None
        if law_id:
            where_clause = {"law_id": law_id}
            
        # Query ChromaDB with both embeddings
        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[bn_embedding, en_embedding],
            n_results=top_k,
            where=where_clause
        )
        
        # Results is a dictionary of lists of lists. Each query has its own list.
        # We need to merge and deduplicate
        unique_chunks = {}
        
        for i in range(len(results.get('documents', []))):
            docs = results['documents'][i]
            metas = results['metadatas'][i]
            distances = results.get('distances', [[0.0] * len(docs)])[i]
            ids = results['ids'][i]
            
            for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
                score = 1.0 - float(dist)
                if score >= settings.MIN_SCORE_THRESHOLD:
                    if doc_id not in unique_chunks or score > unique_chunks[doc_id]['score']:
                        unique_chunks[doc_id] = {
                            "document": doc,
                            "metadata": meta,
                            "score": score
                        }
                        
        chunks = list(unique_chunks.values())
        chunks.sort(key=lambda x: x["score"], reverse=True)
        
        return chunks[:top_k]
