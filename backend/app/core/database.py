import chromadb
from app.core.config import settings

class DatabaseSessionManager:
    def __init__(self):
        self._client = None
        self._collection = None
        self._registry = None

    def init_db(self):
        # We use persistent client as it's thread-safe and suited for FastAPI
        self._client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))
        
        # Create or get collections
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        self._registry = self._client.get_or_create_collection(
            name=settings.LAW_REGISTRY_COLLECTION
        )

    def get_client(self):
        if self._client is None:
            self.init_db()
        return self._client

    def get_collection(self):
        if self._collection is None:
            self.init_db()
        return self._collection
        
    def get_registry(self):
        if self._registry is None:
            self.init_db()
        return self._registry

db_manager = DatabaseSessionManager()

def get_db():
    return db_manager.get_client()
    
def get_collection():
    return db_manager.get_collection()
    
def get_registry():
    return db_manager.get_registry()
