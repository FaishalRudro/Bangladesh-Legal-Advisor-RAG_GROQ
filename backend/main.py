import os
import threading
import pathlib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET_REPO = "RudroBoss/Bangladesh_Legal_Data"
CACHE_REPO = "RudroBoss/Bangladesh_Legal_Data"

def download_file_from_hf(filename, local_path):
    from huggingface_hub import hf_hub_download
    try:
        downloaded = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=os.path.dirname(local_path) or "."
        )
        print(f"✅ Downloaded {filename}")
        return downloaded
    except Exception as e:
        print(f"⚠️ Could not download {filename}: {e}")
        return None

def upload_file_to_hf(local_path, filename):
    from huggingface_hub import HfApi
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        print(f"✅ Uploaded {filename} to HuggingFace")
    except Exception as e:
        print(f"⚠️ Could not upload {filename}: {e}")

def setup_files():
    # Dataset
    dataset_path = "./bangladesh_laws.json"
    if not os.path.exists(dataset_path):
        print("📥 Downloading dataset...")
        download_file_from_hf("bangladesh_laws.json", dataset_path)

    # Embedding cache
    cache_path = "./rag_index.pkl"
    if not os.path.exists(cache_path):
        print("📥 Trying to download embedding cache...")
        download_file_from_hf("rag_index.pkl", cache_path)
    else:
        print("✅ Embedding cache found locally.")

from rag_pipeline import BangladeshLegalRAG, Config

rag_instance: BangladeshLegalRAG = None
index_ready = False
index_error = None

def build_index_background():
    global rag_instance, index_ready, index_error
    try:
        setup_files()

        groq_key = os.environ["GROQ_API_KEY"]
        dataset_path = "./bangladesh_laws.json"
        cache_path = "./rag_index.pkl"

        config = Config()
        config.dataset_path = dataset_path
        config.index_cache_path = cache_path
        config.embed_mmap_path = "./embeddings_tmp.npy"

        rag_instance = BangladeshLegalRAG(config=config, groq_api_key=groq_key)

        cache_existed = os.path.exists(cache_path)
        print("📂 Loading index from cache...")
        rag_instance.build_index(dataset_path)

        # If cache was newly built, upload to HuggingFace for next time
        if not cache_existed and os.path.exists(cache_path):
            print("📤 Uploading embedding cache to HuggingFace...")
            upload_file_to_hf(cache_path, "rag_index.pkl")

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
    allow_origins=["*"],
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

# Serve React frontend
static_path = pathlib.Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/assets", StaticFiles(directory=str(static_path / "assets")), name="assets")

    @app.get("/")
    def root():
        return FileResponse(str(static_path / "index.html"))
else:
    @app.get("/")
    def root():
        return {"message": "Bangladesh Legal Advisor API is running."}