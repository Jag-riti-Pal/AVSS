"""
Module: database.py
Purpose: Persistent ORM schema and repository for CCTV telemetry and incident logs.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Use SQLite for standalone edge; replace with "postgresql://user:pass@localhost:5432/cctv" for Postgres
DATABASE_URL = "sqlite:///./surveillance.db"

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Enable Write-Ahead Logging (WAL) for SQLite to allow concurrent reads and writes
if DATABASE_URL.startswith("sqlite"):
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL;")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class KnownEntity(Base):
    __tablename__ = "entities_known"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    role = Column(String(50), default="RESIDENT")  # RESIDENT, REGULAR_VISITOR
    enrolled_at = Column(DateTime, default=datetime.utcnow)

    events = relationship("SurveillanceEvent", back_populates="entity")


class SurveillanceEvent(Base):
    __tablename__ = "surveillance_events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    event_type = Column(String(50), nullable=False)        # INGRESS, EGRESS, INTRUSION
    identity_label = Column(String(100), nullable=False)    # e.g., "STRANGER", "Mom"
    is_resident = Column(Integer, default=0)                # 1 = True, 0 = False
    confidence = Column(Float, nullable=True)
    clip_path = Column(String(255), nullable=True)          # Local path to MP4
    snapshot_path = Column(String(255), nullable=True)      # Local path to JPEG

    entity_id = Column(Integer, ForeignKey("entities_known.id"), nullable=True)
    entity = relationship("KnownEntity", back_populates="events")


def init_db():
    """Initializes tables in the database."""
    Base.metadata.create_all(bind=engine)