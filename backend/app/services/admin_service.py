from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from sqlalchemy.orm import joinedload
from fastapi import HTTPException
from app.models.user import User, UserRole
from app.models.chat import Feedback
from app.schemas.admin import UserStatusUpdate, UserRoleUpdate

class AdminService:
    @staticmethod
    async def get_all_feedbacks(pg_db: AsyncSession, page: int = 1, size: int = 10):
        offset = (page - 1) * size
        
        total_result = await pg_db.execute(select(func.count(Feedback.id)))
        total = total_result.scalar()
        
        feedbacks_result = await pg_db.execute(
            select(Feedback).options(
                joinedload(Feedback.lawyer)
            ).order_by(Feedback.created_at.desc()).offset(offset).limit(size)
        )
        feedbacks = feedbacks_result.scalars().all()
        
        # Inject mufti_name
        for fb in feedbacks:
            fb.mufti_name = fb.lawyer.name if fb.lawyer else "Unknown Lawyer"
            
        return {"total": total, "page": page, "size": size, "feedbacks": feedbacks}

    @staticmethod
    async def delete_feedback(pg_db: AsyncSession, feedback_id: int):
        result = await pg_db.execute(select(Feedback).filter(Feedback.id == feedback_id))
        feedback = result.scalars().first()
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found")
        await pg_db.delete(feedback)
        await pg_db.commit()
        return True

    @staticmethod
    async def get_all_users(pg_db: AsyncSession, page: int = 1, size: int = 10):
        offset = (page - 1) * size
        
        total_result = await pg_db.execute(select(func.count(User.id)))
        total = total_result.scalar()
        
        users_result = await pg_db.execute(select(User).order_by(User.id.asc()).offset(offset).limit(size))
        users = users_result.scalars().all()
        
        return {"total": total, "page": page, "size": size, "users": users}

    @staticmethod
    async def update_user_status(pg_db: AsyncSession, user_id: int, status_update: UserStatusUpdate):
        result = await pg_db.execute(select(User).filter(User.id == user_id))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.role == UserRole.super_admin:
            raise HTTPException(status_code=400, detail="Cannot deactivate a super admin")
        
        user.is_active = status_update.is_active
        await pg_db.commit()
        return user

    @staticmethod
    async def update_user_role(pg_db: AsyncSession, user_id: int, role_update: UserRoleUpdate):
        result = await pg_db.execute(select(User).filter(User.id == user_id))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.role == UserRole.super_admin:
            raise HTTPException(status_code=400, detail="Cannot change role of a super admin")
            
        user.role = role_update.role
        await pg_db.commit()
        return user

    @staticmethod
    async def delete_user(pg_db: AsyncSession, user_id: int):
        result = await pg_db.execute(select(User).filter(User.id == user_id))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.role == UserRole.super_admin:
            raise HTTPException(status_code=400, detail="Cannot delete a super admin")
            
        await pg_db.delete(user)
        await pg_db.commit()
        return True
