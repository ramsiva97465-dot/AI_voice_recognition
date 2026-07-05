import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.database import Base

class Customer(Base):
    """
    Represents a customer in the voice identity system.
    """
    __tablename__ = "customers"

    customer_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique identifier for the customer (UUID)."
    )
    customer_name = Column(
        String(100),
        nullable=False,
        doc="Name of the customer."
    )
    customer_reference = Column(
        String(100),
        nullable=True,
        doc="Optional external reference ID for the customer."
    )
    mobile_number = Column(
        String(20),
        nullable=True,
        doc="Optional mobile phone number."
    )
    status = Column(
        String(20),
        default="ACTIVE",
        nullable=False,
        doc="Status of the customer (e.g., ACTIVE, INACTIVE)."
    )
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        doc="Timestamp of registration."
    )
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
        doc="Timestamp of last modification."
    )

    # Relationships
    voice_embeddings = relationship(
        "VoiceEmbedding",
        back_populates="customer",
        cascade="all, delete-orphan",
        doc="Voice embeddings associated with this customer."
    )
    authentication_logs = relationship(
        "AuthenticationLog",
        back_populates="customer",
        cascade="all, delete-orphan",
        doc="Authentication logs associated with this customer."
    )


class VoiceEmbedding(Base):
    """
    Represents a voice embedding profile enrolled for a customer.
    """
    __tablename__ = "voice_embeddings"

    embedding_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique identifier for the voice embedding profile (UUID)."
    )
    customer_id = Column(
        UUID(as_uuid=True),
        ForeignKey("customers.customer_id"),
        nullable=False,
        doc="Foreign key pointing to the customer."
    )
    embedding = Column(
        Vector(192),
        nullable=False,
        doc="The generated 192-dimensional voice embedding vector."
    )
    sample_rate = Column(
        Integer,
        nullable=False,
        doc="Sample rate of the audio file used to generate this embedding."
    )
    audio_duration = Column(
        Float,
        nullable=False,
        doc="Duration of the audio in seconds."
    )
    model_name = Column(
        String(100),
        default="ECAPA-TDNN",
        nullable=False,
        doc="The voice identification model used to generate this embedding."
    )
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        doc="Timestamp when the embedding was generated."
    )

    # Relationships
    customer = relationship(
        "Customer",
        back_populates="voice_embeddings",
        doc="The customer this embedding belongs to."
    )


class AuthenticationLog(Base):
    """
    Logs every voice authentication attempt made by a customer.
    """
    __tablename__ = "authentication_logs"

    log_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique identifier for the authentication attempt log (UUID)."
    )
    customer_id = Column(
        UUID(as_uuid=True),
        ForeignKey("customers.customer_id"),
        nullable=False,
        doc="Foreign key pointing to the customer who attempted authentication."
    )
    similarity = Column(
        Float,
        nullable=False,
        doc="The cosine similarity score computed during authentication."
    )
    threshold = Column(
        Float,
        nullable=False,
        doc="The default minimum similarity threshold set at the time of authentication."
    )
    authenticated = Column(
        Boolean,
        nullable=False,
        doc="True if similarity met the threshold, indicating successful verification."
    )
    processing_time_ms = Column(
        Float,
        nullable=False,
        doc="Total time taken to process the request in milliseconds."
    )
    audio_duration = Column(
        Float,
        nullable=False,
        doc="Duration of the audio in seconds."
    )
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        doc="Timestamp of the authentication attempt."
    )

    # Relationships
    customer = relationship(
        "Customer",
        back_populates="authentication_logs",
        doc="The customer this log entry belongs to."
    )
