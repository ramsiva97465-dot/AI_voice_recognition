#!/usr/bin/env python
"""FastAPI entry point for the Voice Authentication service.

Provides a minimal health‑check endpoint and endpoints for enrolling,
verifying, identifying, and authenticating speakers.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import numpy as np
import time
from datetime import datetime
import soundfile as sf
from app import models
from app.embedding import generate_embedding, generate_embedding_from_waveform
from constants import DEFAULT_THRESHOLD

from fastapi.middleware.cors import CORSMiddleware
import vad_processor
from app import customer_manager

# Database Integration
from sqlalchemy.orm import Session
from app.database import Base, engine, get_db
from app import crud

app = FastAPI(title="Voice Authentication API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure required directories exist
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    # Automatically create database tables during FastAPI startup
    Base.metadata.create_all(bind=engine)

@app.get("/", response_class=JSONResponse)
async def root() -> dict:
    """Root health‑check endpoint.

    Returns a simple JSON payload confirming that the API is running.
    """
    return {"message": "Voice Authentication API Running"}

@app.post("/enroll", response_class=JSONResponse)
async def enroll(
    name: str = Form(...),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> dict:
    """Enroll a new speaker.

    Saves the uploaded audio temporarily, generates an embedding, stores it
    into PostgreSQL database, and cleans up the temporary file.
    """
    total_start = time.perf_counter()
    temp_path = TEMP_DIR / audio.filename
    try:
        # Save uploaded file to temporary location
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        # Get sample rate and audio duration
        audio_np, sr = sf.read(str(temp_path))
        audio_duration = len(audio_np) / sr
        
        # Generate embedding
        embedding = generate_embedding(str(temp_path))
        
        # Create Customer if not exists
        customer = crud.get_customer_by_name(db, name)
        if not customer:
            customer = crud.create_customer(db, name)
            
        # Save embedding into PostgreSQL
        crud.save_embedding(
            db=db,
            customer_id=customer.customer_id,
            embedding=embedding,
            sample_rate=sr,
            audio_duration=audio_duration
        )
        
        processing_time_ms = (time.perf_counter() - total_start) * 1000.0
        
        # Cleanup temporary file
        temp_path.unlink(missing_ok=True)
        
        return {
            "status": "success",
            "customer_id": str(customer.customer_id),
            "customer_name": customer.customer_name,
            "embedding_saved": True,
            "embedding_dimension": int(embedding.shape[0]),
            "sample_rate": int(sr),
            "audio_duration": round(audio_duration, 2),
            "processing_time_ms": round(processing_time_ms, 2),
            "message": "Speaker enrolled successfully."
        }
    except Exception as exc:
        import traceback
        with open("error.log", "a") as f:
            f.write(f"Error in /enroll: {exc}\n{traceback.format_exc()}\n")
        # Ensure temp file is removed on error
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))

# Verification endpoint
@app.post("/verify", response_class=JSONResponse)
async def verify(
    name: str = Form(...),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> dict:
    """Verify a speaker by comparing an uploaded audio embedding against the enrolled embedding."""
    # Save uploaded audio to temporary file
    temp_path = TEMP_DIR / audio.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        # Generate embedding from the uploaded audio
        embedding = generate_embedding(str(temp_path))
        
        # Load enrolled speaker embeddings from DB
        customer = crud.get_customer_by_name(db, name)
        if not customer or not customer.voice_embeddings:
            raise HTTPException(status_code=404, detail=f"Speaker '{name}' not found")
            
        best_sim = 0.0
        for ve in customer.voice_embeddings:
            emb = np.array(ve.embedding)
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(emb)
            sim = float(np.dot(embedding, emb) / (norm_a * norm_b)) if norm_a and norm_b else 0.0
            if sim > best_sim:
                best_sim = sim
                
        threshold = DEFAULT_THRESHOLD
        verified = best_sim >= threshold
        return {
            "speaker": name,
            "similarity": round(best_sim * 100, 2),
            "threshold": threshold * 100,
            "verified": verified,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        temp_path.unlink(missing_ok=True)

# Identification endpoint
@app.post("/identify", response_class=JSONResponse)
async def identify(
    audio: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> dict:
    """Identify the most similar enrolled speaker."""
    temp_path = TEMP_DIR / audio.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        # Generate embedding from uploaded audio
        embedding = generate_embedding(str(temp_path))
        
        # Fetch all embeddings from PostgreSQL
        all_embeddings = crud.get_all_embeddings(db)
        if not all_embeddings:
            raise HTTPException(status_code=404, detail="No enrolled speakers found")
            
        best_name = "Unknown"
        best_sim = 0.0
        for row in all_embeddings:
            emb = np.array(row.embedding)
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(emb)
            sim = float(np.dot(embedding, emb) / (norm_a * norm_b)) if norm_a and norm_b else 0.0
            if sim > best_sim:
                best_sim = sim
                best_name = row.customer_name
                
        threshold = DEFAULT_THRESHOLD
        identified = best_sim >= threshold
        return {
            "predicted_speaker": best_name if identified else "Unknown",
            "similarity": round(best_sim * 100, 2),
            "identified": identified,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        temp_path.unlink(missing_ok=True)

# Authenticate endpoint (Orchestration)
@app.post("/authenticate", response_class=JSONResponse)
async def authenticate(
    audio: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> dict:
    """Automatic Customer Voice Identity Service endpoint."""
    total_start = time.perf_counter()
    temp_path = TEMP_DIR / audio.filename

    try:
        # Save the uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        # Fetch all embeddings from PostgreSQL
        all_embeddings = crud.get_all_embeddings(db)
        
        # 1. Check if we have any enrolled speakers at all
        if not all_embeddings:
            processing_time_ms = (time.perf_counter() - total_start) * 1000.0
            return {
                "status": "warning",
                "existing_customer": False,
                "authentication_result": "NO_REGISTERED_SPEAKERS",
                "similarity": 0.0,
                "processing_time_ms": round(processing_time_ms, 2),
                "audio_duration": 0.0,
                "message": "No enrolled speakers found in the system."
            }

        # 2. VAD & Audio Loading
        waveform, sr = vad_processor.load_audio(temp_path)
        speech_waveform = vad_processor.remove_silence(waveform, sr)
        
        duration_seconds = len(speech_waveform) / sr
        if duration_seconds < 15.0:
            processing_time_ms = (time.perf_counter() - total_start) * 1000.0
            return {
                "status": "error",
                "authentication_result": "INSUFFICIENT_AUDIO",
                "required_duration_seconds": 15.0,
                "received_duration_seconds": round(duration_seconds, 2),
                "processing_time_ms": round(processing_time_ms, 2),
                "message": "Need at least 15 seconds of clear speech."
            }

        # 3. Generate the embedding
        embedding = generate_embedding_from_waveform(speech_waveform, sr)

        # 4. Cosine similarity search
        best_name = None
        best_customer_id = None
        best_sim = 0.0
        
        for row in all_embeddings:
            emb = np.array(row.embedding)
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(emb)
            sim = (
                float(np.dot(embedding, emb) / (norm_a * norm_b))
                if norm_a and norm_b
                else 0.0
            )
            if sim > best_sim:
                best_sim = sim
                best_customer_id = row.customer_id
                best_name = row.customer_name
                
        threshold = DEFAULT_THRESHOLD
        
        # Determine authentication result
        authenticated = best_customer_id is not None and best_sim >= threshold
        
        # Save authentication log into PostgreSQL
        if best_customer_id is not None:
            crud.save_authentication_log(
                db=db,
                customer_id=best_customer_id,
                similarity=best_sim,
                threshold=threshold,
                authenticated=authenticated,
                processing_time_ms=(time.perf_counter() - total_start) * 1000.0,
                audio_duration=duration_seconds
            )
            
            # Fetch matched customer to get matched_embedding_count
            matched_customer = crud.get_customer_by_id(db, best_customer_id)
            matched_embedding_count = len(matched_customer.voice_embeddings) if matched_customer else 0
        else:
            matched_embedding_count = 0

        processing_time_ms = (time.perf_counter() - total_start) * 1000.0
        current_timestamp = datetime.utcnow().isoformat()

        if authenticated:
            return {
                "status": "success",
                "existing_customer": True,
                "customer_id": str(best_customer_id),
                "customer_name": best_name,
                "authentication_result": "AUTHENTICATED",
                "similarity": round(best_sim * 100, 2),
                "threshold": threshold,
                "processing_time_ms": round(processing_time_ms, 2),
                "audio_duration": round(duration_seconds, 2),
                "matched_embedding_count": matched_embedding_count,
                "model": "ECAPA-TDNN",
                "timestamp": current_timestamp,
                "message": "Speaker authenticated successfully."
            }
        else:
            return {
                "status": "warning",
                "existing_customer": False,
                "authentication_result": "UNKNOWN_SPEAKER",
                "similarity": round(best_sim * 100, 2),
                "threshold": threshold,
                "processing_time_ms": round(processing_time_ms, 2),
                "audio_duration": round(duration_seconds, 2),
                "matched_embedding_count": matched_embedding_count,
                "model": "ECAPA-TDNN",
                "timestamp": current_timestamp,
                "message": "Speaker not registered."
            }

    except Exception as exc:
        processing_time_ms = (time.perf_counter() - total_start) * 1000.0
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "authentication_result": "INTERNAL_SERVER_ERROR",
            "processing_time_ms": round(processing_time_ms, 2),
            "message": str(exc)
        }
    finally:
        temp_path.unlink(missing_ok=True)
