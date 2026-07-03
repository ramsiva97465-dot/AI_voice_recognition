#!/usr/bin/env python
"""FastAPI entry point for the Voice Authentication service.

Provides a minimal health‑check endpoint. Additional routes will be added later.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import numpy as np

from app.embedding import generate_embedding, generate_embedding_from_waveform
from constants import DEFAULT_THRESHOLD

from fastapi.middleware.cors import CORSMiddleware
import vad_processor
from app import customer_manager

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
SPEAKER_DB_DIR = BASE_DIR / "database" / "speakers"
SPEAKER_DB_DIR.mkdir(parents=True, exist_ok=True)

speaker_embeddings_cache: dict[str, np.ndarray] = {}

def reload_cache():
    """Helper to reload all speaker embeddings into the in-memory cache."""
    global speaker_embeddings_cache
    new_cache = {}
    for npy_path in SPEAKER_DB_DIR.glob("*.npy"):
        new_cache[npy_path.stem] = np.load(npy_path)
    speaker_embeddings_cache.clear()
    speaker_embeddings_cache.update(new_cache)

@app.on_event("startup")
async def startup_event():
    reload_cache()

@app.get("/", response_class=JSONResponse)
async def root() -> dict:
    """Root health‑check endpoint.

    Returns a simple JSON payload confirming that the API is running.
    """
    return {"message": "Voice Authentication API Running"}

@app.post("/enroll", response_class=JSONResponse)
async def enroll(
    name: str = Form(...),
    audio: UploadFile = File(...)
) -> dict:
    """Enroll a new speaker.

    Saves the uploaded audio temporarily, generates an embedding, stores it,
    and cleans up the temporary file. Returns JSON with status and details.
    """
    temp_path = TEMP_DIR / audio.filename
    try:
        # Save uploaded file to temporary location
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        # Generate embedding
        embedding = generate_embedding(str(temp_path))
        # Save embedding to database
        save_path = SPEAKER_DB_DIR / f"{name}.npy"
        np.save(save_path, embedding)
        # Update in-memory cache
        speaker_embeddings_cache[name] = embedding
        # Cleanup temporary file
        temp_path.unlink(missing_ok=True)
        return {
            "status": "success",
            "speaker": name,
            "embedding_dimension": embedding.shape[0]
        }
    except Exception as exc:
        import traceback
        with open("error.log", "a") as f:
            f.write(f"Error in /enroll: {exc}\\n{traceback.format_exc()}\\n")
        # Ensure temp file is removed on error
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))

# Verification endpoint
@app.post("/verify", response_class=JSONResponse)
async def verify(
    name: str = Form(...),
    audio: UploadFile = File(...)
) -> dict:
    """Verify a speaker by comparing an uploaded audio embedding against the enrolled embedding."""
    # Save uploaded audio to temporary file
    temp_path = TEMP_DIR / audio.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        # Generate embedding from the uploaded audio
        embedding = generate_embedding(str(temp_path))
        # Load enrolled speaker embedding
        enrolled_path = SPEAKER_DB_DIR / f"{name}.npy"
        if not enrolled_path.is_file():
            raise HTTPException(status_code=404, detail=f"Speaker '{name}' not found")
        enrolled_emb = np.load(enrolled_path)
        # Cosine similarity
        norm_a = np.linalg.norm(embedding)
        norm_b = np.linalg.norm(enrolled_emb)
        similarity = float(np.dot(embedding, enrolled_emb) / (norm_a * norm_b)) if norm_a and norm_b else 0.0
        threshold = DEFAULT_THRESHOLD
        verified = similarity >= threshold
        return {
            "speaker": name,
            "similarity": round(similarity * 100, 2),
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
    audio: UploadFile = File(...)
) -> dict:
    """Identify the most similar enrolled speaker."""
    temp_path = TEMP_DIR / audio.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        # Generate embedding from uploaded audio
        embedding = generate_embedding(str(temp_path))
        if not speaker_embeddings_cache:
            raise HTTPException(status_code=404, detail="No enrolled speakers found")
        # Compute similarities
        best_name = "Unknown"
        best_sim = 0.0
        for name, emb in speaker_embeddings_cache.items():
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(emb)
            sim = float(np.dot(embedding, emb) / (norm_a * norm_b)) if norm_a and norm_b else 0.0
            if sim > best_sim:
                best_sim = sim
                best_name = name
        threshold = DEFAULT_THRESHOLD
        identified = best_sim >= threshold
        result = {
            "predicted_speaker": best_name if identified else "Unknown",
            "similarity": round(best_sim * 100, 2),
            "identified": identified,
        }
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        temp_path.unlink(missing_ok=True)

# Authenticate endpoint (Orchestration)
@app.post("/authenticate", response_class=JSONResponse)
async def authenticate(
    audio: UploadFile = File(...)
) -> dict:
    """Automatic Customer Voice Identity Service endpoint."""
    import time
    import traceback

    print("========== START /authenticate ===========")
    print("[DEBUG] Request received")
    total_start = time.perf_counter()
    temp_path = TEMP_DIR / audio.filename

    # ---------- 1️⃣ Save uploaded file ----------
    try:
        start = time.perf_counter()
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        elapsed = time.perf_counter() - start
        print(f"[PERF] Saving uploaded file: {elapsed:.2f}s")
    except Exception as exc:
        print("[ERROR] Saving uploaded file")
        traceback.print_exc()
        raise

    # ---------- 2️⃣ Load audio (VAD) ----------
    try:
        start = time.perf_counter()
        waveform, sr = vad_processor.load_audio(temp_path)
        elapsed = time.perf_counter() - start
        print(f"[PERF] Load audio: {elapsed:.2f}s")
    except Exception as exc:
        print("[ERROR] Load audio")
        traceback.print_exc()
        raise

    # ---------- 3️⃣ Voice Activity Detection ----------
    try:
        start = time.perf_counter()
        speech_waveform = vad_processor.remove_silence(waveform, sr)
        elapsed = time.perf_counter() - start
        print(f"[PERF] Voice activity detection: {elapsed:.2f}s")
    except Exception as exc:
        print("[ERROR] Voice activity detection")
        traceback.print_exc()
        raise

    duration_seconds = len(speech_waveform) / sr
    if duration_seconds < 15.0:
        total_elapsed = time.perf_counter() - total_start
        print(f"[PERF] total endpoint execution: {total_elapsed:.2f}s")
        return {
            "status": "insufficient_audio",
            "message": "Need at least 15 seconds of clear speech.",
        }

    # Debug info for the processed waveform
    print(f"type(speech_waveform): {type(speech_waveform)}")
    print(f"speech_waveform.shape: {speech_waveform.shape}")
    print(f"speech_waveform.dtype: {speech_waveform.dtype}")
    print(f"sr: {sr}")

    # ---------- 4️⃣ Generate embedding ----------
    try:
        start = time.perf_counter()
        embedding = generate_embedding_from_waveform(speech_waveform, sr)
        elapsed = time.perf_counter() - start
        print(f"[PERF] Embedding: {elapsed:.2f}s")
    except Exception as exc:
        print("[ERROR] Embedding generation")
        traceback.print_exc()
        raise

    # ---------- 5️⃣ Cosine similarity search ----------
    try:
        start = time.perf_counter()
        best_name = None
        best_sim = 0.0
        if speaker_embeddings_cache:
            for name, emb in speaker_embeddings_cache.items():
                norm_a = np.linalg.norm(embedding)
                norm_b = np.linalg.norm(emb)
                sim = (
                    float(np.dot(embedding, emb) / (norm_a * norm_b))
                    if norm_a and norm_b
                    else 0.0
                )
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
        threshold = DEFAULT_THRESHOLD
        elapsed = time.perf_counter() - start
        print(f"[PERF] Cosine similarity search: {elapsed:.2f}s")
    except Exception as exc:
        print("[ERROR] Cosine similarity search")
        traceback.print_exc()
        raise

    # ---------- 6️⃣ Customer lookup / creation ----------
    try:
        start = time.perf_counter()
        if best_name and best_sim >= threshold:
            # Existing Customer
            customer = customer_manager.update_customer(best_name)
            response = {
                "customer_id": best_name,
                "existing_customer": True,
                "similarity": round(best_sim * 100, 2),
                "call_count": customer.get("call_count", 1),
                "status": "existing_customer",
            }
        else:
            # New Customer
            new_id = customer_manager.generate_customer_id()
            save_path = SPEAKER_DB_DIR / f"{new_id}.npy"
            np.save(save_path, embedding)
            speaker_embeddings_cache[new_id] = embedding
            customer = customer_manager.create_customer(new_id)
            response = {
                "customer_id": new_id,
                "existing_customer": False,
                "similarity": round(best_sim * 100, 2) if best_name else 0.0,
                "call_count": customer.get("call_count", 1),
                "status": "new_customer",
            }
        elapsed = time.perf_counter() - start
        print(f"[PERF] Customer creation/update: {elapsed:.2f}s")
    except Exception as exc:
        print("[ERROR] Customer handling")
        traceback.print_exc()
        raise

    # ---------- Return response ----------
    try:
        start = time.perf_counter()
        # response already prepared
        elapsed = time.perf_counter() - start
        print(f"[PERF] Returning response: {elapsed:.2f}s")
    except Exception as exc:
        print("[ERROR] Returning response")
        traceback.print_exc()
        raise

    total_elapsed = time.perf_counter() - total_start
    print(f"========== END /authenticate (Total: {total_elapsed:.2f}s) ===========")
    temp_path.unlink(missing_ok=True)
    return response
