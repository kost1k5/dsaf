from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import settings

# Create the SQLAlchemy engine using the database URL from settings
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Dependency to get a database session.
    Ensures the session is always closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
