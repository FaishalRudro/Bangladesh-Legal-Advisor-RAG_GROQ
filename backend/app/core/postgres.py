from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.core.config import settings

# Convert synchronous URL to async URL
# Also strip ?sslmode=require — asyncpg handles SSL via connect_args instead
_base_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
ASYNC_DATABASE_URL = _base_url.split("?")[0]

# Enable SSL for hosted databases (e.g. Neon, Supabase)
_connect_args = {"ssl": True} if "neon.tech" in ASYNC_DATABASE_URL or "supabase" in ASYNC_DATABASE_URL else {}

engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    connect_args=_connect_args,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)
AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

Base = declarative_base()

async def get_pg_db():
    async with AsyncSessionLocal() as session:
        yield session
