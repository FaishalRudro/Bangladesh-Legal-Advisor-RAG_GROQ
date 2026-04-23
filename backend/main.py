import os
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import BangladeshLegalRAG, Config

rag_instance: BangladeshLegalRAG = None
index_ready = False
index_error = None

def build_index_background():
    global rag_instance, index_ready, index_error
    try:
        groq_key = os.environ["GROQ_API_KEY"]
        dataset_path = os.environ["DATASET_PATH"]
        cache_path = os.environ.get("INDEX_CACHE_PATH", "./rag_index.pkl")

        config = Config()
        config.dataset_path = dataset_path
        config.index_cache_path = cache_path
        config.embed_mmap_path = "./embeddings_tmp.npy"

        rag_instance = BangladeshLegalRAG(config=config, groq_api_key=groq_key)
        rag_instance.build_index(dataset_path)
        index_ready = True
        print("✅ RAG index ready.")
    except Exception as e:
        index_error = str(e)
        print(f"❌ Index build failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=build_index_background, daemon=True)
    thread.start()
    yield

app = FastAPI(title="Bangladesh Legal Advisor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    lang: str

class StatusResponse(BaseModel):
    ready: bool
    error: str | None = None
    stats: dict | None = None

@app.get("/status", response_model=StatusResponse)
def get_status():
    if index_error:
        return StatusResponse(ready=False, error=index_error)
    if index_ready:
        return StatusResponse(ready=True, stats=rag_instance.get_stats())
    return StatusResponse(ready=False)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not index_ready:
        raise HTTPException(status_code=503, detail="Index is still loading. Please wait.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        from rag_pipeline import LanguageDetector
        lang = LanguageDetector.detect(req.query)
        answer = rag_instance.chat(req.query, stream=False, verbose=True)
        return ChatResponse(answer=answer, lang=lang)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-history")
def clear_history():
    if rag_instance:
        rag_instance.clear_history()
    return {"message": "History cleared."}

@app.get("/")
def root():
    return {"message": "Bangladesh Legal Advisor API is running."}