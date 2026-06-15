from app.services.chunker import Chunker
from app.services.embedder import Embedder
from app.core.config import settings
import chromadb
import uuid
import json
from datetime import datetime

class Ingestor:
    def __init__(self, db: chromadb.Client):
        self.db = db
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.collection = db.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest(self, law_dict: dict) -> int:
        """
        Chunks the document, embeds the chunks, and saves to ChromaDB.
        Returns the number of chunks ingested.
        """
        # 1. Chunk document
        chunks = self.chunker.chunk_law(law_dict)
        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        
        # Build metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(chunks):
            m = chunk["metadata"].copy()
            # clean None values
            m = {k: (v if v is not None else "") for k, v in m.items()}
            m["law_id"] = law_dict["law_id"]
            metadatas.append(m)

        ids = [f"{law_dict['law_id']}_{i}" for i in range(len(chunks))]

        # 2. Embed
        embeddings = self.embedder.embed_documents(texts)

        # 3. Upsert to ChromaDB in batches
        batch_size = 5000  # Chroma recommended max batch size
        for i in range(0, len(texts), batch_size):
            self.collection.upsert(
                ids=ids[i:i + batch_size],
                documents=texts[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size]
            )

        return len(chunks)

def register_law(db: chromadb.Client, meta: dict):
    registry = db.get_or_create_collection(name=settings.LAW_REGISTRY_COLLECTION)
    # clean metadata for chroma (no None values)
    clean_meta = {k: (v if v is not None else "") for k, v in meta.items()}
    
    registry.upsert(
        ids=[meta["law_id"]],
        documents=[meta.get("law_name", "law")],
        embeddings=[[0.0] * settings.EMBEDDING_DIMENSIONS],
        metadatas=[clean_meta]
    )

def registry_has_law(db: chromadb.Client, law_id: str) -> bool:
    registry = db.get_or_create_collection(name=settings.LAW_REGISTRY_COLLECTION)
    res = registry.get(ids=[law_id])
    return len(res["ids"]) > 0

def delete_law(db: chromadb.Client, law_id: str):
    # 1. Delete all chunks belonging to this law from main collection
    collection = db.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    collection.delete(where={"law_id": law_id})

    # 2. Remove from law registry
    registry = db.get_or_create_collection(name=settings.LAW_REGISTRY_COLLECTION)
    registry.delete(ids=[law_id])
