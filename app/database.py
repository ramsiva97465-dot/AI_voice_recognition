import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment variables from .env file
load_dotenv()

# Retrieve the database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set or empty")

# Create the SQLAlchemy engine.
# Using future=True for SQLAlchemy 2.0 style and pool_pre_ping=True to check connection health.
engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True
)

# Create a configured "SessionLocal" class
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True
)

# Create a DeclarativeMeta class for defining models
Base = declarative_base()

def get_db():
    """
    Dependency helper to yield a database session.
    Ensures that the database session is closed after the request is finished.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
