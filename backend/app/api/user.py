from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.postgres import get_pg_db
from app.models.user import User
from app.schemas.user import UserResponse
from app.schemas.profile import UserUpdate, ChangePasswordRequest, ChangePasswordVerify, ForgotPasswordRequest, ResetPasswordRequest
from app.services.profile_service import ProfileService
from app.middleware.auth import require_any_user

router = APIRouter()

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get Current User Profile",
    description="Retrieves the profile information of the currently authenticated user."
)
async def get_me(current_user: User = Depends(require_any_user)):
    return current_user

@router.patch(
    "/me",
    response_model=UserResponse,
    summary="Update Profile",
    description="Updates the profile information of the currently authenticated user."
)
async def update_profile(
    data: UserUpdate,
    db: AsyncSession = Depends(get_pg_db),
    current_user: User = Depends(require_any_user)
):
    return await ProfileService.update_profile(db, current_user, data)

@router.post(
    "/change-password/request",
    summary="Request Password Change",
    description="Sends an OTP to the user's email to verify their password change request."
)
async def change_password_request(
    data: ChangePasswordRequest,
    db: AsyncSession = Depends(get_pg_db),
    current_user: User = Depends(require_any_user)
):
    return await ProfileService.change_password_request(db, current_user, data)

@router.post(
    "/change-password/verify",
    summary="Verify Password Change",
    description="Verifies the OTP and updates the password for the current user."
)
async def change_password_verify(
    data: ChangePasswordVerify,
    db: AsyncSession = Depends(get_pg_db),
    current_user: User = Depends(require_any_user)
):
    return await ProfileService.change_password_verify(db, current_user, data)

@router.post(
    "/forgot-password",
    summary="Forgot Password",
    description="Sends a password reset link to the user's email if the account exists."
)
async def forgot_password(
    data: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_pg_db)
):
    return await ProfileService.forgot_password(db, data)

@router.post(
    "/reset-password",
    summary="Reset Password",
    description="Resets the user's password using the token sent to their email."
)
async def reset_password(
    data: ResetPasswordRequest,
    db: AsyncSession = Depends(get_pg_db)
):
    return await ProfileService.reset_password(db, data)
