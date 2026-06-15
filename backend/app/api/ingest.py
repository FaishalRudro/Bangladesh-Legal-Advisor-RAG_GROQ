from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from datetime import datetime
import chromadb
import json
from pathlib import Path

from app.core.config import settings
from app.core.database import get_db, get_registry
from app.schemas.law_schema import LawIngestResponse, LawListResponse, LawInfo, DeleteResponse
from app.services.ingestor import Ingestor, register_law, registry_has_law, delete_law
from app.models.user import User
from app.middleware.auth import require_superadmin, require_mufti_superadmin

router = APIRouter()

def process_ingestion(db: chromadb.Client, limit: int = 0):
    json_path = settings.DATA_DIR / "bangladesh_laws.json"
    if not json_path.exists():
        print("bangladesh_laws.json not found in data directory.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    laws = data if isinstance(data, list) else [data]
    if limit > 0:
        laws = laws[:limit]

    ingestor = Ingestor(db)

    for idx, law_data in enumerate(laws):
        # Build law dictionary
        law_full_text = law_data.get("law_full_text", "").strip()
        if not law_full_text:
            continue
            
        pipe_parts = [p.strip() for p in law_full_text.split(" | ") if p.strip()]
        law_name = pipe_parts[0] if pipe_parts else "Unknown Law"
        
        # Simple ID generation or assignment
        law_id = f"law_{idx + 1}"
        
        law_dict = {
            "law_id": law_id,
            "law_full_text": law_full_text,
            "law_name": law_name,
            "year": law_data.get("year", ""),
            "link": law_data.get("link", ""),
        }
        
        repealed = "REPEALED" if "[REPEALED:" in law_full_text or "[REPEAL" in law_full_text else "ACTIVE"
        law_dict["repealed"] = repealed

        if registry_has_law(db, law_id):
            print(f"Law {law_id} already exists, skipping.")
            continue

        try:
            total_chunks = ingestor.ingest(law_dict)
            if total_chunks > 0:
                register_law(db, {
                    **law_dict,
                    "total_chunks": total_chunks,
                    "ingested_at": datetime.utcnow().isoformat()
                })
                print(f"Ingested {law_name} with {total_chunks} chunks. ({idx + 1}/{len(laws)})")
        except Exception as e:
            print(f"Failed to ingest {law_name}: {e}")

@router.post(
    "/laws/ingest", 
    summary="Ingest Bangladesh Laws",
    description="Reads bangladesh_laws.json from data directory and ingests laws into Chroma."
)
async def ingest_laws(
    background_tasks: BackgroundTasks,
    limit: int = Query(10, description="Number of laws to ingest (0 for all)"),
    db: chromadb.Client = Depends(get_db),
    current_user: User = Depends(require_superadmin)
):
    json_path = settings.DATA_DIR / "bangladesh_laws.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="bangladesh_laws.json not found in data directory.")
        
    background_tasks.add_task(process_ingestion, db, limit)
    
    return {"status": "started", "message": f"Ingestion process started in background for {limit if limit > 0 else 'all'} laws."}


@router.get(
    "/laws", 
    response_model=LawListResponse,
    summary="List All Laws",
    description="Retrieves a paginated list of all laws currently ingested."
)
async def list_laws(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    db: chromadb.Client = Depends(get_db), 
    current_user: User = Depends(require_mufti_superadmin)
):
    registry = get_registry()
    total = registry.count()
    
    offset = (page - 1) * size
    res = registry.get(limit=size, offset=offset)
    
    laws = []
    if res and res.get("ids"):
        for i in range(len(res["ids"])):
            meta = res["metadatas"][i]
            
            try:
                chunks_count = int(meta.get("total_chunks")) if meta.get("total_chunks") not in ["", None] else 0
            except (ValueError, TypeError):
                chunks_count = 0
                
            laws.append(LawInfo(
                law_id=res["ids"][i],
                law_name=res["documents"][i] if res["documents"] else meta.get("law_name", ""),
                year=meta.get("year", ""),
                link=meta.get("link", ""),
                repealed=meta.get("repealed", ""),
                total_chunks=chunks_count,
                ingested_at=meta.get("ingested_at", "")
            ))
            
    return LawListResponse(total=total, page=page, size=size, laws=laws)

@router.delete(
    "/laws/{law_id}", 
    response_model=DeleteResponse,
    summary="Delete a Law",
    description="Permanently deletes a law from the RAG system."
)
async def remove_law(law_id: str, db: chromadb.Client = Depends(get_db), current_user: User = Depends(require_superadmin)):
    if not registry_has_law(db, law_id):
        raise HTTPException(status_code=404, detail="Law not found in database.")
        
    delete_law(db, law_id)
        
    return DeleteResponse(
        law_id=law_id,
        status="deleted",
        message="Law and all associated vectors successfully removed."
    )
