from app.core.postgres import Base
from app.models.user import User
from app.models.chat import ChatSession, ChatMessage, Feedback

# This file is used to import all models so that Alembic can auto-generate migrations
