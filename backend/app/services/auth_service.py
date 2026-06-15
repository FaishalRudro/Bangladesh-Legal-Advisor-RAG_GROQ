from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from fastapi import HTTPException
from datetime import timedelta, datetime
from app.models.user import User, UserRole
from app.schemas.user import UserCreate, MuftiCreate, SetPassword, VerifyOTP, ResendOTP, UserLogin
from app.core.security import get_password_hash, verify_password, create_access_token
from app.services.email_service import send_otp_email, send_invitation_email, generate_otp
from app.core.config import settings

# In-memory store for pending verifications
pending_users = {}

class AuthService:
    @staticmethod
    async def signup(user_in: UserCreate, db: AsyncSession):
        result = await db.execute(select(User).filter(User.email == user_in.email))
        user = result.scalars().first()
        if user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        otp = generate_otp()
        hashed_password = get_password_hash(user_in.password)
        expires_at = datetime.utcnow() + timedelta(seconds=settings.OTP_EXPIRE_SECONDS)
        
        # Store securely in memory instead of DB
        pending_users[user_in.email] = {
            "name": user_in.name,
            "password": hashed_password,
            "otp": otp,
            "expires_at": expires_at
        }
        
        await send_otp_email(user_in.email, otp)
        return {"msg": f"OTP sent to {user_in.email}. Please verify your account."}

    @staticmethod
    async def verify_otp(data: VerifyOTP, db: AsyncSession):
        result = await db.execute(select(User).filter(User.email == data.email))
        user = result.scalars().first()
        if user:
            raise HTTPException(status_code=400, detail="User is already registered and verified.")
            
        pending_user = pending_users.get(data.email)
        if not pending_user:
            raise HTTPException(status_code=404, detail="No pending registration found for this email.")
            
        if datetime.utcnow() > pending_user["expires_at"]:
            raise HTTPException(status_code=400, detail="OTP has expired. Please request a new one.")
            
        if pending_user["otp"] != data.otp:
            raise HTTPException(status_code=400, detail="Invalid OTP")
        
        # Insert into the database ONLY after successful verification
        new_user = User(
            name=pending_user["name"],
            email=data.email,
            hashed_password=pending_user["password"],
            role=UserRole.user,
            is_verified=True,
            otp=None
        )
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        del pending_users[data.email]
        return new_user

    @staticmethod
    async def resend_otp(data: ResendOTP, db: AsyncSession):
        result = await db.execute(select(User).filter(User.email == data.email))
        user = result.scalars().first()
        if user:
            raise HTTPException(status_code=400, detail="Email is already registered. Please login.")
            
        pending_user = pending_users.get(data.email)
        if not pending_user:
            raise HTTPException(status_code=404, detail="No pending registration found. Please sign up first.")
            
        new_otp = generate_otp()
        pending_user["otp"] = new_otp
        pending_user["expires_at"] = datetime.utcnow() + timedelta(seconds=settings.OTP_EXPIRE_SECONDS)
        
        await send_otp_email(data.email, new_otp)
        return {"msg": f"A new OTP has been sent to {data.email}."}

    @staticmethod
    async def login(login_data: UserLogin, db: AsyncSession):
        result = await db.execute(select(User).filter(User.email == login_data.email))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=400, detail=f"User not found in DB: {login_data.email}")
        if not user.hashed_password:
            raise HTTPException(status_code=400, detail="User exists but has no password set.")
        if not verify_password(login_data.password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Incorrect password. The password does not match the database.")
        if not user.is_verified:
            raise HTTPException(status_code=400, detail="Account is not verified.")
            
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email, "role": user.role.value}, expires_delta=access_token_expires
        )
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": user
        }

    @staticmethod
    async def create_mufti(mufti_in: MuftiCreate, db: AsyncSession):
        result = await db.execute(select(User).filter(User.email == mufti_in.email))
        user = result.scalars().first()
        if user:
            raise HTTPException(status_code=400, detail="Email already exists")
            
        new_mufti = User(
            name=mufti_in.name,
            email=mufti_in.email,
            role=UserRole.lawyer,
            is_verified=False
        )
        db.add(new_mufti)
        await db.commit()
        await db.refresh(new_mufti)
        
        await send_invitation_email(new_mufti.email)
        return new_mufti

    @staticmethod
    async def accept_invite(data: SetPassword, db: AsyncSession):
        result = await db.execute(select(User).filter(User.email == data.email, User.role == UserRole.lawyer))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="Lawyer not found")
        if user.is_verified and user.hashed_password:
            raise HTTPException(status_code=400, detail="Account already active")
            
        user.hashed_password = get_password_hash(data.password)
        user.is_verified = True
        await db.commit()
        return {"msg": "Password set successfully. You can now login."}
