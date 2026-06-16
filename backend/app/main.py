from fastapi import FastAPI
from app.api import ingest, chat, auth, admin, user
from app.core.postgres import engine, Base, AsyncSessionLocal
from sqlalchemy.future import select
from app.models.user import User, UserRole
from app.core.security import get_password_hash
from app.core.config import settings
import logging
import logging.config
import os
from pathlib import Path
from app.middleware.custom import CustomMiddleware

# ---------------------------------------------------------------------------
# Logging configuration
# All app loggers (verifier, generator, etc.) use INFO level and write to
# both the console (uvicorn's stream) and a rotating file in LOG_DIR.
# ---------------------------------------------------------------------------
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "default",
            "filename": str(settings.LOG_DIR / "app.log"),
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        # Root logger — catches everything not matched below
        "": {
            "handlers": ["console", "file"],
            "level": "WARNING",
            "propagate": False,
        },
        # Our application loggers
        "app": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
        # uvicorn access/error logs
        "uvicorn": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bangladesh Legal Advisor",
    description="Scholarly Law",
    version="2.0.0"
)

app.add_middleware(CustomMiddleware)

@app.on_event("startup")
async def startup_event():
    # Re-apply logging config on startup (needed when uvicorn --reload overrides it)
    logging.config.dictConfig(LOGGING_CONFIG)

    # Download bangladesh_laws.json from HuggingFace Dataset if not present
    laws_path = settings.DATA_DIR / "bangladesh_laws.json"
    if not laws_path.exists():
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        dataset_url = (
            "https://huggingface.co/datasets/RudroBoss/Bangladesh_Legal_Data"
            "/resolve/main/bangladesh_laws.json"
        )
        logger.info("bangladesh_laws.json not found locally — downloading from HuggingFace Dataset...")
        try:
            import urllib.request
            headers = {}
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"

            req = urllib.request.Request(dataset_url, headers=headers)
            laws_path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(req) as response, open(laws_path, "wb") as out_file:
                out_file.write(response.read())
            logger.info(f"Downloaded bangladesh_laws.json ({laws_path.stat().st_size // (1024*1024)} MB)")
        except Exception as e:
            logger.error(f"Failed to download bangladesh_laws.json: {e}")
    else:
        logger.info("bangladesh_laws.json already present, skipping download.")

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).filter(User.role == UserRole.super_admin))
        super_admin = result.scalars().first()
        if not super_admin:
            logger.info("Creating default super admin account...")
            hashed_pw = get_password_hash(settings.SUPER_ADMIN_PASSWORD)
            super_admin_user = User(
                name=settings.SUPER_ADMIN_NAME,
                email=settings.SUPER_ADMIN_EMAIL,
                hashed_password=hashed_pw,
                role=UserRole.super_admin,
                is_verified=True
            )
            db.add(super_admin_user)
            await db.commit()

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(user.router, prefix="/api/v1/users", tags=["users"])
app.include_router(ingest.router, prefix="/api/v1", tags=["laws"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])

@app.get("/api/v1/health")
def health_check():
    return {"status": "ok", "message": "Legal RAG API is running."}

# Trigger reload
