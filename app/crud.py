import uuid
import numpy as np
from sqlalchemy.orm import Session
from app.models import Customer, VoiceEmbedding, AuthenticationLog

def create_customer(
    db: Session,
    customer_name: str,
    customer_reference: str = None,
    mobile_number: str = None
) -> Customer:
    """
    Creates a new customer record.
    
    Args:
        db (Session): Database session.
        customer_name (str): Name of the customer.
        customer_reference (str, optional): External customer reference ID.
        mobile_number (str, optional): Mobile phone number.
        
    Returns:
        Customer: The newly created customer object.
    """
    new_customer = Customer(
        customer_name=customer_name,
        customer_reference=customer_reference,
        mobile_number=mobile_number
    )
    try:
        db.add(new_customer)
        db.commit()
        db.refresh(new_customer)
        return new_customer
    except Exception as e:
        db.rollback()
        raise e


def get_customer_by_id(db: Session, customer_id: uuid.UUID) -> Customer | None:
    """
    Retrieves a customer by their UUID.
    
    Args:
        db (Session): Database session.
        customer_id (uuid.UUID): Customer identifier.
        
    Returns:
        Customer | None: The matching customer object, or None if not found.
    """
    try:
        return db.query(Customer).filter(Customer.customer_id == customer_id).first()
    except Exception as e:
        db.rollback()
        raise e


def get_customer_by_name(db: Session, customer_name: str) -> Customer | None:
    """
    Retrieves a customer by their name.
    
    Args:
        db (Session): Database session.
        customer_name (str): Name of the customer.
        
    Returns:
        Customer | None: The matching customer object, or None if not found.
    """
    try:
        return db.query(Customer).filter(Customer.customer_name == customer_name).first()
    except Exception as e:
        db.rollback()
        raise e


def save_embedding(
    db: Session,
    customer_id: uuid.UUID,
    embedding: np.ndarray,
    sample_rate: int,
    audio_duration: float
) -> VoiceEmbedding:
    """
    Saves a new voice embedding vector to the database for a customer.
    
    Args:
        db (Session): Database session.
        customer_id (uuid.UUID): ID of the customer.
        embedding (np.ndarray): NumPy array representing the voice embedding vector.
        sample_rate (int): Sample rate of the audio file.
        audio_duration (float): Duration of the audio in seconds.
        
    Returns:
        VoiceEmbedding: The newly created voice embedding object.
    """
    # Convert numpy array to list representation for pgvector compatibility
    embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
    
    new_embedding = VoiceEmbedding(
        customer_id=customer_id,
        embedding=embedding_list,
        sample_rate=sample_rate,
        audio_duration=audio_duration
    )
    try:
        db.add(new_embedding)
        db.commit()
        db.refresh(new_embedding)
        return new_embedding
    except Exception as e:
        db.rollback()
        raise e


def get_all_embeddings(db: Session):
    """
    Retrieves all voice embeddings mapped with their customer details.
    
    Args:
        db (Session): Database session.
        
    Returns:
        List[Row]: Result rows containing customer_id, customer_name, embedding, 
                   sample_rate, and audio_duration.
    """
    try:
        return db.query(
            VoiceEmbedding.customer_id,
            Customer.customer_name,
            VoiceEmbedding.embedding,
            VoiceEmbedding.sample_rate,
            VoiceEmbedding.audio_duration
        ).join(Customer, VoiceEmbedding.customer_id == Customer.customer_id).all()
    except Exception as e:
        db.rollback()
        raise e


def save_authentication_log(
    db: Session,
    customer_id: uuid.UUID,
    similarity: float,
    threshold: float,
    authenticated: bool,
    processing_time_ms: float,
    audio_duration: float
) -> AuthenticationLog:
    """
    Saves a voice authentication log record.
    
    Args:
        db (Session): Database session.
        customer_id (uuid.UUID): ID of the customer.
        similarity (float): Cosine similarity score.
        threshold (float): Similarity threshold.
        authenticated (bool): True if verification succeeded, False otherwise.
        processing_time_ms (float): Request processing time.
        audio_duration (float): Duration of the audio file.
        
    Returns:
        AuthenticationLog: The created authentication log entry.
    """
    new_log = AuthenticationLog(
        customer_id=customer_id,
        similarity=similarity,
        threshold=threshold,
        authenticated=authenticated,
        processing_time_ms=processing_time_ms,
        audio_duration=audio_duration
    )
    try:
        db.add(new_log)
        db.commit()
        db.refresh(new_log)
        return new_log
    except Exception as e:
        db.rollback()
        raise e


def get_authentication_history(db: Session, customer_id: uuid.UUID):
    """
    Retrieves all authentication logs for a customer, ordered by latest first.
    
    Args:
        db (Session): Database session.
        customer_id (uuid.UUID): Customer identifier.
        
    Returns:
        List[AuthenticationLog]: List of authentication logs.
    """
    try:
        return db.query(AuthenticationLog)\
            .filter(AuthenticationLog.customer_id == customer_id)\
            .order_by(AuthenticationLog.created_at.desc())\
            .all()
    except Exception as e:
        db.rollback()
        raise e


def delete_customer(db: Session, customer_id: uuid.UUID) -> bool:
    """
    Deletes a customer and all their associated voice embeddings and logs.
    
    Args:
        db (Session): Database session.
        customer_id (uuid.UUID): ID of the customer to delete.
        
    Returns:
        bool: True if customer was deleted, False if customer was not found.
    """
    try:
        customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
        if customer:
            db.delete(customer)
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        raise e


def get_all_customers(db: Session):
    """
    Retrieves all customers with their voice embedding counts.
    
    Args:
        db (Session): Database session.
        
    Returns:
        List[Tuple[Customer, int]]: List of tuples containing Customer object and their voice embedding count.
    """
    from sqlalchemy import func
    try:
        return db.query(
            Customer,
            func.count(VoiceEmbedding.embedding_id).label("embedding_count")
        ).outerjoin(
            VoiceEmbedding, Customer.customer_id == VoiceEmbedding.customer_id
        ).group_by(
            Customer.customer_id
        ).all()
    except Exception as e:
        db.rollback()
        raise e

