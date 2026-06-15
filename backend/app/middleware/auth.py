from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.core.postgres import get_pg_db
from app.models.user import User, UserRole
from app.core.config import settings
import jwt
from jwt.exceptions import InvalidTokenError as JWTError

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/swagger-login")

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_pg_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    result = await db.execute(select(User).filter(User.email == email))
    user = result.scalars().first()
    if user is None:
        raise credentials_exception
    return user

def require_role(roles: list[UserRole]):
    async def role_checker(request: Request, current_user: User = Depends(get_current_user)):
        if not current_user.is_active:
            if request.method != "GET":
                raise HTTPException(
                    status_code=403, 
                    detail="Your account has been deactivated. You can only view your data, but cannot perform any actions."
                )
        if current_user.role not in roles:
            raise HTTPException(status_code=403, detail="Not enough privileges")
        return current_user
    return role_checker

# Pre-configured role dependencies
require_superadmin = require_role([UserRole.super_admin])
require_mufti = require_role([UserRole.super_admin, UserRole.lawyer])
require_mufti_superadmin = require_role([UserRole.super_admin, UserRole.lawyer])
require_any_user = require_role([UserRole.super_admin, UserRole.lawyer, UserRole.user])
