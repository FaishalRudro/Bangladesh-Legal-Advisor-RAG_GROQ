from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.core.postgres import get_pg_db
from app.models.user import User, UserRole
from app.schemas.user import UserCreate, MuftiCreate, SetPassword, VerifyOTP, ResendOTP, UserLogin, UserResponse, Token
from app.services.auth_service import AuthService
from app.middleware.auth import require_superadmin

router = APIRouter()


@router.post(
    "/signup",
    summary="User Signup",
    description="""
Initiates the standard user registration process.

### Workflow:
1. Validates the password strength (length, uppercase, lowercase, numbers, special characters).
2. Checks if the email already exists in the database.
3. Generates a 6-digit OTP and saves the user data to temporary in-memory storage.
4. Sends the OTP to the provided email address via SMTP.

*Note: The user is NOT saved to PostgreSQL until the OTP is verified.*

**Permissions:** Public (Open)
"""
)
async def signup(user_in: UserCreate, db: Session = Depends(get_pg_db)):
    return await AuthService.signup(user_in, db)

@router.post(
    "/resend-otp",
    summary="Resend Verification OTP",
    description="""
Generates a new verification OTP for a pending registration.

### Behavior:
If the user's initial OTP expired or was not received, this endpoint creates a new one and dispatches it via email. It resets the expiration timer for the in-memory storage.

**Permissions:** Public (Open)
"""
)
async def resend_otp(data: ResendOTP, db: Session = Depends(get_pg_db)):
    return await AuthService.resend_otp(data, db)

@router.post(
    "/verify-otp", 
    response_model=UserResponse,
    summary="Verify OTP",
    description="""
Finalizes the registration process by validating the OTP.

### Behavior:
1. Verifies the OTP against the in-memory storage.
2. Checks if the OTP has expired.
3. If successful, hashes the password and permanently inserts the user into the PostgreSQL database.
4. Clears the temporary in-memory data.

**Permissions:** Public (Open)
"""
)
async def verify_otp(data: VerifyOTP, db: Session = Depends(get_pg_db)):
    return await AuthService.verify_otp(data, db)

@router.post(
    "/login", 
    response_model=Token,
    summary="User Login",
    description="""
Authenticates a user and returns a JSON Web Token (JWT).

### Request Payload:
Accepts a standard JSON payload containing `email` and `password`.

### Response Payload:
Returns an OAuth2-compliant token payload along with the full `user` object to populate frontend state management without requiring a secondary `/me` request.

### Error Responses:
- **400 Bad Request**: User not found, incorrect password, or account is unverified.

**Permissions:** Public (Open)
"""
)
async def login(login_data: UserLogin, db: Session = Depends(get_pg_db)):
    return await AuthService.login(login_data, db)

@router.post("/swagger-login", response_model=Token, include_in_schema=False)
async def swagger_login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_pg_db)):
    # Swagger UI strictly requires form-data. We map it to our JSON schema internally.
    login_data = UserLogin(email=form_data.username, password=form_data.password)
    return await AuthService.login(login_data, db)

@router.post(
    "/create-lawyer", 
    response_model=UserResponse,
    summary="Create Lawyer Account",
    description="""
Creates a new administrative Lawyer account.

### Workflow:
1. Accepts the name and email of the intended Lawyer.
2. Creates an unverified `User` record in PostgreSQL with the `lawyer` role.
3. Dispatches a welcome email via SMTP containing a dynamic link to the frontend application.
4. The frontend link enables the Lawyer to establish their password.

**Permissions:** Super Admin Only
"""
)
async def create_mufti(mufti_in: MuftiCreate, db: Session = Depends(get_pg_db), current_user: User = Depends(require_superadmin)):
    return await AuthService.create_mufti(mufti_in, db)

@router.post(
    "/accept-invite",
    summary="Accept Lawyer Invitation",
    description="""
Allows an invited user (Lawyer) to establish their password and activate their account.

### Workflow:
1. Validates the provided password against strength criteria.
2. Locates the pending user record by email.
3. Hashes the password and marks the account as `is_verified = True`.

**Permissions:** Public (Open via invitation link)
"""
)
async def accept_invite(data: SetPassword, db: Session = Depends(get_pg_db)):
    return await AuthService.accept_invite(data, db)
