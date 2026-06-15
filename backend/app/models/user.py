from sqlalchemy import Column, Integer, String, Boolean, Enum
import enum
from app.core.postgres import Base

class UserRole(str, enum.Enum):
    super_admin = "super_admin"
    lawyer = "lawyer"
    user = "user"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True) # nullable for admin until they set it
    role = Column(Enum(UserRole), default=UserRole.user, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    otp = Column(String, nullable=True) # for user verification
