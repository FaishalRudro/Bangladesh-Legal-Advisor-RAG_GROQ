from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from fastapi import HTTPException
from datetime import datetime, timedelta
import jwt
from app.models.user import User
from app.schemas.profile import UserUpdate, ChangePasswordRequest, ChangePasswordVerify, ForgotPasswordRequest, ResetPasswordRequest
from app.core.security import verify_password, get_password_hash
from app.services.email_service import generate_otp, send_password_change_otp_email, send_reset_password_email
from app.core.config import settings

password_change_otps = {}

class ProfileService:
    @staticmethod
    async def update_profile(db: AsyncSession, current_user: User, data: UserUpdate):
        if data.name:
            current_user.name = data.name
        await db.commit()
        await db.refresh(current_user)
        return current_user

    @staticmethod
    async def change_password_request(db: AsyncSession, current_user: User, data: ChangePasswordRequest):
        if not verify_password(data.old_password, current_user.hashed_password):
            raise HTTPException(status_code=400, detail="Incorrect old password")
        
        otp = generate_otp()
        password_change_otps[current_user.email] = {
            "otp": otp,
            "expires_at": datetime.utcnow() + timedelta(minutes=10) # 10 mins expiry
        }
        
        await send_password_change_otp_email(current_user.email, otp)
        return {"detail": "OTP sent to your email to verify password change"}

    @staticmethod
    async def change_password_verify(db: AsyncSession, current_user: User, data: ChangePasswordVerify):
        otp_data = password_change_otps.get(current_user.email)
        if not otp_data:
            raise HTTPException(status_code=400, detail="No password change request found")
        
        if datetime.utcnow() > otp_data["expires_at"]:
            del password_change_otps[current_user.email]
            raise HTTPException(status_code=400, detail="OTP expired. Please request again.")
            
        if otp_data["otp"] != data.otp:
            raise HTTPException(status_code=400, detail="Invalid OTP")
            
        current_user.hashed_password = get_password_hash(data.new_password)
        await db.commit()
        
        del password_change_otps[current_user.email]
        return {"detail": "Password changed successfully"}

    @staticmethod
    async def forgot_password(db: AsyncSession, data: ForgotPasswordRequest):
        result = await db.execute(select(User).filter(User.email == data.email))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="Email not found in our database.")
            
        expire = datetime.utcnow() + timedelta(hours=1)
        reset_token = jwt.encode(
            {"sub": user.email, "exp": expire, "type": "reset"},
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        
        await send_reset_password_email(user.email, reset_token)
        return {"detail": "A password reset link has been sent to your email."}

    @staticmethod
    async def reset_password(db: AsyncSession, data: ResetPasswordRequest):
        try:
            payload = jwt.decode(data.token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            if payload.get("type") != "reset":
                raise HTTPException(status_code=400, detail="Invalid token type")
            email = payload.get("sub")
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=400, detail="Reset link has expired")
        except jwt.PyJWTError:
            raise HTTPException(status_code=400, detail="Invalid reset link")
            
        result = await db.execute(select(User).filter(User.email == email))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
            
        user.hashed_password = get_password_hash(data.new_password)
        await db.commit()
        return {"detail": "Password has been reset successfully. You can now login."}
