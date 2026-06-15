from fastapi import APIRouter, Depends, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.postgres import get_pg_db
from app.models.user import User
from app.middleware.auth import require_superadmin
from app.schemas.admin import PaginatedFeedbacks, PaginatedUsers, UserStatusUpdate, UserRoleUpdate
from app.schemas.user import UserResponse
from app.services.admin_service import AdminService

router = APIRouter()

@router.get(
    "/feedbacks",
    response_model=PaginatedFeedbacks,
    summary="Get All Feedbacks",
    description="""
Fetches a paginated list of all feedbacks provided by Lawyers.
Includes the feedback text, the lawyer who provided it, and the associated message.

**Permissions:** Super Admin Only
"""
)
async def get_all_feedbacks(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    pg_db: AsyncSession = Depends(get_pg_db),
    current_user: User = Depends(require_superadmin)
):
    return await AdminService.get_all_feedbacks(pg_db, page, size)


@router.delete(
    "/feedbacks/{feedback_id}",
    summary="Delete Feedback",
    description="""
Permanently deletes a specific feedback.

**Permissions:** Super Admin Only
"""
)
async def delete_feedback(
    feedback_id: int = Path(...),
    pg_db: AsyncSession = Depends(get_pg_db),
    current_user: User = Depends(require_superadmin)
):
    await AdminService.delete_feedback(pg_db, feedback_id)
    return {"detail": "Feedback deleted successfully"}


@router.get(
    "/users",
    response_model=PaginatedUsers,
    summary="Get All Users",
    description="""
Fetches a paginated list of all registered users, including regular users, lawyers, and super admins.

**Permissions:** Super Admin Only
"""
)
async def get_all_users(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    pg_db: AsyncSession = Depends(get_pg_db),
    current_user: User = Depends(require_superadmin)
):
    return await AdminService.get_all_users(pg_db, page, size)


@router.patch(
    "/users/{user_id}/status",
    response_model=UserResponse,
    summary="Activate/Deactivate User",
    description="""
Toggles a user's active status. Deactivated users can log in to view their profile, but are blocked from using protected APIs like `/chat` or `/sessions`.
Super Admins cannot deactivate themselves or other Super Admins.

**Permissions:** Super Admin Only
"""
)
async def update_user_status(
    status_update: UserStatusUpdate,
    user_id: int = Path(...),
    pg_db: AsyncSession = Depends(get_pg_db),
    current_user: User = Depends(require_superadmin)
):
    return await AdminService.update_user_status(pg_db, user_id, status_update)


@router.patch(
    "/users/{user_id}/role",
    response_model=UserResponse,
    summary="Change User Role",
    description="""
Upgrades or downgrades a user's role (e.g., from 'user' to 'lawyer').
Super Admins cannot change the role of another Super Admin.

**Permissions:** Super Admin Only
"""
)
async def update_user_role(
    role_update: UserRoleUpdate,
    user_id: int = Path(...),
    pg_db: AsyncSession = Depends(get_pg_db),
    current_user: User = Depends(require_superadmin)
):
    return await AdminService.update_user_role(pg_db, user_id, role_update)


@router.delete(
    "/users/{user_id}",
    summary="Delete User",
    description="""
Permanently deletes a user and all their associated data (chat sessions, messages, feedback).

**Permissions:** Super Admin Only
"""
)
async def delete_user(
    user_id: int = Path(...),
    pg_db: AsyncSession = Depends(get_pg_db),
    current_user: User = Depends(require_superadmin)
):
    await AdminService.delete_user(pg_db, user_id)
    return {"detail": "User deleted successfully"}
