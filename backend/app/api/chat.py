from fastapi import APIRouter, HTTPException, Depends, Query
import chromadb
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.database import get_db
from app.core.postgres import get_pg_db
from app.schemas.chat_schema import ChatRequest, ChatResponse, ChatSessionListResponse, ChatSessionDetailResponse, FeedbackRequest, PinSessionRequest, PaginatedSessions
from app.services.retriever import Retriever
from app.services.generator import Generator
from app.formatters.response_formatter import ScholarlyResponseFormatter
from app.models.user import User
from app.middleware.auth import require_any_user, require_mufti
from app.services.chat_service import ChatService

router = APIRouter()

@router.post(
    "/chat", 
    response_model=ChatResponse,
    summary="Chat with Bangladesh Legal Advisor",
    description="""
Queries the RAG system to generate scholarly Legal answers based on ingested Law books.

### Execution Pipeline:
1. **Language Detection**: Detects if the query is in English or Bengali. Rejects unsupported languages.
2. **Translation**: Translates the query into classical Legal to perform high-fidelity semantic search against the original texts.
3. **Retrieval**: Queries the ChromaDB vector database to retrieve the top `K` most relevant semantic chunks.
4. **Verification**: A secondary LLM agent programmatically verifies if the retrieved contexts explicitly contain the answer. If not, it safely aborts to prevent hallucinations.
5. **Generation**: The primary LLM generates a Legal Advice-style essay combining the context, inline Legal citations, and translation.

### Request Body (`ChatRequest`):
- `query` (str): The user's question in English or Bengali.
- `law_id` (str, optional): If provided, strictly limits the search to a specific law.
- `top_k` (int, default=5): Number of chunks to retrieve.
- `session_id` (int, optional): The ID of the session to continue. If not provided, a new session is created.

### Error Responses:
- **400 Bad Request**: Query language not supported (e.g., Legal).
- **401 Unauthorized**: Missing or invalid Bearer token.

**Permissions:** Authenticated Users Only (Super Admin, Lawyer, User)
"""
)
async def chat(request: ChatRequest, db: chromadb.Client = Depends(get_db), pg_db: AsyncSession = Depends(get_pg_db), current_user: User = Depends(require_any_user)):
    retriever = Retriever(db)
    generator = Generator()
    formatter = ScholarlyResponseFormatter()

    # 1. Language Detection
    lang = await retriever.detect_language(request.query)
    if lang == "other":
        raise HTTPException(
            status_code=400,
            detail="Query language not supported. Please use Bengali or English. (আরবিতে প্রশ্ন করবেন না।)"
        )

    # 2. Translate Query for Bilingual Search
    if lang == "bn":
        query_bn = request.query
        query_en = await retriever.translate_query(request.query, source_lang="bn")
    else:
        query_en = request.query
        query_bn = await retriever.translate_query(request.query, source_lang="en")

    # 3. Retrieve Documents from both English and Bengali Embeddings
    target_law_id = request.law_id
    if target_law_id in ["", "string"]:
        target_law_id = None

    chunks = await retriever.retrieve_bilingual(
        query_bn=query_bn,
        query_en=query_en,
        law_id=target_law_id,
        top_k=request.top_k
    )


    # for idx, c in enumerate(chunks, 1):
    #     print(f"\nContext {idx} [Score: {c['score']:.4f}]")
    #     print(c['document'])


    # 4. Verify Context Programmatically (Hybrid Approach)
    from app.services.verifier import Verifier
    verifier = Verifier()
    
    if not await verifier.verify(query_bn, query_en, chunks):
        llm_answer = (
            "দুঃখিত, আপনার প্রশ্নের উত্তর প্রদত্ত কিতাবসমূহে পাওয়া যায়নি।" 
            if lang == "bn" else 
            "Sorry, the answer to your question was not found in the provided texts."
        )
    else:
        # 5. Generate Scholarly Essay Response
        llm_answer = await generator.generate(
            query=request.query,
            language=lang,
            chunks=chunks
        )

    # 6. Format Response with Metadata
    response = formatter.format(
        llm_answer=llm_answer,
        retrieved_chunks=chunks,
        query_lang=lang,
        legal_query=query_en if lang == "bn" else query_bn
    )

    # 7. Save to Database using ChatService
    session_id, message_id = await ChatService.save_chat_message(
        pg_db=pg_db,
        current_user=current_user,
        query=request.query,
        ai_response_dict=response.model_dump(),
        session_id=request.session_id
    )

    response.message_id = message_id
    response.session_id = session_id

    return response

@router.get(
    "/sessions", 
    response_model=PaginatedSessions,
    summary="Get User Chat Sessions",
    description="""
Retrieves a paginated list of all chat sessions for the authenticated user.
The sessions are returned in the following order:
1. Pinned sessions first (newest to oldest).
2. Unpinned sessions second (newest to oldest).

This endpoint only returns lightweight session metadata (ID, title, creation date, pin status). It does **not** return the actual chat messages. To fetch the messages, use the `/sessions/{session_id}` endpoint.

**Permissions:** Authenticated Users Only
"""
)
async def get_sessions(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    pg_db: AsyncSession = Depends(get_pg_db), 
    current_user: User = Depends(require_any_user)
):
    """Get paginated chat sessions for the current user."""
    return await ChatService.get_user_sessions(pg_db, current_user.id, page, size)

@router.get(
    "/sessions/{session_id}", 
    response_model=ChatSessionDetailResponse,
    summary="Get Chat Session History",
    description="""
Retrieves the complete history of a specific chat session with paginated messages.
This includes:
- The session details (title, pin status, etc.)
- Paginated queries and full AI responses within this session
- Any feedback left by Lawyers on specific messages, including the Lawyer's name.

**Permissions:** Authenticated Users Only (can only view their own sessions)

### Path Parameters:
- `session_id` (int): The ID of the session to retrieve.
"""
)
async def get_session(
    session_id: int, 
    page: int = Query(1, ge=1),
    size: int = Query(10,ge=1, le=100),
    pg_db: AsyncSession = Depends(get_pg_db), 
    current_user: User = Depends(require_any_user)
):
    """Get a specific chat session with its paginated messages."""
    return await ChatService.get_user_session(pg_db, session_id, current_user.id, page, size)

@router.delete(
    "/sessions/{session_id}",
    summary="Delete Chat Session",
    description="""
Permanently deletes a specific chat session and all its associated messages and feedbacks.
Users can only delete their own chat sessions.

**Permissions:** Authenticated Users Only (can only delete their own sessions)

### Path Parameters:
- `session_id` (int): The ID of the session to delete.
"""
)
async def delete_session(session_id: int, pg_db: AsyncSession = Depends(get_pg_db), current_user: User = Depends(require_any_user)):
    """Delete a chat session."""
    await ChatService.delete_user_session(pg_db, session_id, current_user.id)
    return {"detail": "Chat session deleted successfully"}

@router.patch(
    "/sessions/{session_id}/pin",
    summary="Pin/Unpin Chat Session",
    description="""
Toggles the pinned status of a specific chat session.
Pinned sessions appear at the top of the chat history list.

**Permissions:** Authenticated Users Only (can only modify their own sessions)

### Path Parameters:
- `session_id` (int): The ID of the session to pin or unpin.

### Request Body:
- `is_pinned` (bool): `true` to pin the session, `false` to unpin it.
"""
)
async def pin_session(
    session_id: int, 
    request: PinSessionRequest,
    pg_db: AsyncSession = Depends(get_pg_db), 
    current_user: User = Depends(require_any_user)
):
    """Pin or unpin a chat session."""
    await ChatService.toggle_pin_session(pg_db, session_id, current_user.id, request.is_pinned)
    status = "pinned" if request.is_pinned else "unpinned"
    return {"detail": f"Chat session {status} successfully"}

@router.post(
    "/messages/{message_id}/feedback",
    summary="Submit Lawyer Feedback",
    description="""
Allows Lawyers to submit scholarly feedback on specific AI-generated responses.
This feedback helps improve the quality and accuracy of the Legal RAG system.

### Request Body:
- `is_good` (bool, optional): `true` for thumbs up, `false` for thumbs down.
- `feedback_text` (str, optional): Detailed textual feedback regarding the response's Law accuracy or formatting.

**Permissions:** Lawyers Only. (A Lawyer can only provide feedback on their own chat sessions).

### Path Parameters:
- `message_id` (int): The ID of the message to provide feedback for.
"""
)
async def add_feedback(
    message_id: int, 
    request: FeedbackRequest, 
    pg_db: AsyncSession = Depends(get_pg_db), 
    current_user: User = Depends(require_mufti)
):
    """Allow lawyers to provide feedback on a specific message."""
    await ChatService.add_feedback(
        pg_db=pg_db, 
        message_id=message_id, 
        lawyer=current_user, 
        is_good=request.is_good, 
        feedback_text=request.feedback_text
    )
    return {"detail": "Feedback saved successfully."}
