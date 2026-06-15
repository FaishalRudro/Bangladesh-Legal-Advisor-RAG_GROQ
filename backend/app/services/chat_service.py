from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload
from fastapi import HTTPException
from app.models.chat import ChatSession, ChatMessage, Feedback
from app.models.user import User

class ChatService:
    @staticmethod
    async def save_chat_message(pg_db: AsyncSession, current_user: User, query: str, ai_response_dict: dict, session_id: int = None) -> tuple[int, int]:
        if not session_id:
            new_session = ChatSession(user_id=current_user.id, title=query[:50])
            pg_db.add(new_session)
            await pg_db.commit()
            await pg_db.refresh(new_session)
            session_id = new_session.id
        else:
            result = await pg_db.execute(select(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id))
            existing_session = result.scalars().first()
            if not existing_session:
                raise HTTPException(status_code=404, detail="Chat session not found or does not belong to you.")

        chat_message = ChatMessage(
            session_id=session_id,
            user_query=query,
            ai_response=ai_response_dict
        )
        pg_db.add(chat_message)
        await pg_db.commit()
        await pg_db.refresh(chat_message)

        return session_id, chat_message.id

    @staticmethod
    async def get_user_sessions(pg_db: AsyncSession, user_id: int, page: int = 1, size: int = 10):
        from sqlalchemy import func
        offset = (page - 1) * size
        
        total_result = await pg_db.execute(select(func.count(ChatSession.id)).filter(ChatSession.user_id == user_id))
        total = total_result.scalar()

        result = await pg_db.execute(
            select(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.is_pinned.desc(), ChatSession.created_at.desc())
            .offset(offset)
            .limit(size)
        )
        sessions = result.scalars().all()
        return {"total": total, "page": page, "size": size, "sessions": sessions}

    @staticmethod
    async def toggle_pin_session(pg_db: AsyncSession, session_id: int, user_id: int, is_pinned: bool):
        result = await pg_db.execute(select(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user_id))
        session = result.scalars().first()
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found.")
        session.is_pinned = is_pinned
        await pg_db.commit()
        return True

    @staticmethod
    async def get_user_session(pg_db: AsyncSession, session_id: int, user_id: int, page: int = 1, size: int = 10):
        from sqlalchemy import func
        result = await pg_db.execute(
            select(ChatSession)
            .filter(
                ChatSession.id == session_id, 
                ChatSession.user_id == user_id
            )
        )
        session = result.scalars().first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found.")
            
        offset = (page - 1) * size
        
        total_msg_result = await pg_db.execute(select(func.count(ChatMessage.id)).filter(ChatMessage.session_id == session_id))
        total_messages = total_msg_result.scalar()
        
        messages_result = await pg_db.execute(
            select(ChatMessage).options(joinedload(ChatMessage.feedbacks).joinedload(Feedback.lawyer))
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .offset(offset)
            .limit(size)
        )
        messages = messages_result.unique().scalars().all()
            
        # Process feedbacks to inject mufti_name
        for message in messages:
            for feedback in message.feedbacks:
                feedback.mufti_name = feedback.lawyer.name if feedback.lawyer else "Unknown Lawyer"
                
        return {
            "id": session.id,
            "user_id": session.user_id,
            "title": session.title,
            "created_at": session.created_at,
            "is_pinned": session.is_pinned,
            "total_messages": total_messages,
            "page": page,
            "size": size,
            "messages": messages
        }

    @staticmethod
    async def delete_user_session(pg_db: AsyncSession, session_id: int, user_id: int):
        result = await pg_db.execute(select(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user_id))
        session = result.scalars().first()
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found.")
        await pg_db.delete(session)
        await pg_db.commit()
        return True

    @staticmethod
    async def add_feedback(pg_db: AsyncSession, message_id: int, lawyer: User, is_good: bool, feedback_text: str):
        result_msg = await pg_db.execute(select(ChatMessage).filter(ChatMessage.id == message_id))
        message = result_msg.scalars().first()
        if not message:
            raise HTTPException(status_code=404, detail="Message not found.")
        
        result_session = await pg_db.execute(select(ChatSession).filter(ChatSession.id == message.session_id))
        session = result_session.scalars().first()
        
        if session.user_id != lawyer.id:
            raise HTTPException(status_code=403, detail="You can only provide feedback on your own messages.")

        result_fb = await pg_db.execute(select(Feedback).filter(
            Feedback.message_id == message_id,
            Feedback.lawyer_id == lawyer.id
        ))
        feedback = result_fb.scalars().first()

        if not feedback:
            feedback = Feedback(
                message_id=message_id,
                lawyer_id=lawyer.id,
                is_good=is_good,
                feedback_text=feedback_text
            )
            pg_db.add(feedback)
        else:
            if is_good is not None:
                feedback.is_good = is_good
            if feedback_text is not None:
                feedback.feedback_text = feedback_text

        await pg_db.commit()
        return True
