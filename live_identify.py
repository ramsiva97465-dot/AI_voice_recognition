#!/usr/bin/env python
"""
live_identify.py

Live microphone speaker identification (Phase 5).

Records audio from the default microphone, generates a speaker embedding using the
existing ``app.embedding.generate_embedding`` function, compares it against all
enrolled speakers in ``database/speakers/`` and reports the most likely speaker.

Usage
-----
    python live_identify.py [--duration DURATION] [--threshold THRESHOLD]

Options
-------
    --duration   Recording length in seconds (default: 5)
    --threshold  Similarity threshold (0‑1) for a confident identification
                 (default: 0.75)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# External libraries for microphone capture
import sounddevice as sd
import soundfile as sf

# Re‑use the embedding generator from the existing code base.
from app.embedding import generate_embedding
from constants import DEFAULT_THRESHOLD

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DURATION: float = 10.0          # seconds
# Using imported DEFAULT_THRESHOLD from constants
SAMPLE_RATE: int = 16000                # Hz – required by the embedding model
TEMP_WAV: Path = Path("temp_recording.wav")

# ---------------------------------------------------------------------------
# Helper functions (mirroring identify.py)
# ---------------------------------------------------------------------------
def load_database(db_dir: Path) -> Dict[str, np.ndarray]:
    """Load all ``.npy`` speaker embeddings from ``db_dir``.

    Parameters
    ----------
    db_dir: Path
        Directory ``database/speakers`` containing ``<Speaker>.npy`` files.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping of speaker name → embedding vector.
    """
    if not db_dir.is_dir():
        raise FileNotFoundError(f"Speaker database folder not found: {db_dir}")

    embeddings: Dict[str, np.ndarray] = {}
    for npy_path in db_dir.glob("*.npy"):
        speaker_name = npy_path.stem
        try:
            embeddings[speaker_name] = np.load(npy_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load embedding {npy_path}: {exc}") from exc

    if not embeddings:
        raise RuntimeError(f"No enrolled speaker embeddings found in {db_dir}")

    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two 1‑D vectors.

    The result is in the range ``[-1, 1]``; ``0`` is returned if either vector has
    zero norm.
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Both inputs must be 1‑D vectors.")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def identify_speaker(
    test_emb: np.ndarray,
    database: Dict[str, np.ndarray],
    threshold: float,
) -> Tuple[str, float, bool]:
    """Find the most similar enrolled speaker.

    Returns a tuple ``(predicted_name, best_similarity, is_identified)``.
    """
    similarities = {name: cosine_similarity(test_emb, emb) for name, emb in database.items()}
    # Sort by similarity descending for display
    sorted_sims = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)

    print("\nSearching speaker database...\n")
    for name, sim in sorted_sims:
        status = "PASS" if sim >= threshold else "FAIL"
        print(f"{name:<10}: {sim * 100:.2f}%   [{status}]")

    best_name, best_sim = sorted_sims[0]
    identified = best_sim >= threshold
    predicted_name = best_name if identified else "Unknown"
    return predicted_name, best_sim, identified

# ---------------------------------------------------------------------------
# Recording utilities
# ---------------------------------------------------------------------------
def record_audio(duration: float, filename: Path) -> None:
    """Record ``duration`` seconds from the default microphone and write to ``filename``.

    Parameters
    ----------
    duration: float
        Recording length in seconds.
    filename: Path
        Destination WAV file (overwrites if it already exists).
    """
    try:
        print("Recording... (press Ctrl+C to abort)")
        # Ensure samplerate and channels are set
        sd.default.samplerate = SAMPLE_RATE
        sd.default.channels = 1
        # Record audio
        audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()  # Block until recording finishes
        # Diagnostic information about the raw audio array
        print("--- Recording diagnostics ---")
        print(f"Sample rate        : {SAMPLE_RATE} Hz")
        actual_duration = audio.shape[0] / SAMPLE_RATE
        print(f"Duration (samples) : {audio.shape[0]} samples")
        print(f"Duration (seconds) : {actual_duration:.2f} s")
        print(f"Array shape        : {audio.shape}")
        print(f"Array dtype        : {audio.dtype}")
        print(f"Min value          : {audio.min():.6f}")
        print(f"Max value          : {audio.max():.6f}")
        rms = (audio ** 2).mean() ** 0.5
        print(f"RMS volume         : {rms:.6f}")
        # Write to WAV file
        sf.write(str(filename), audio, SAMPLE_RATE)
        # File info after saving
        abs_path = filename.resolve()
        file_size = abs_path.stat().st_size
        print("--- File info ---")
        print(f"Saved to           : {abs_path}")
        print(f"File size (bytes)  : {file_size}")
        print("Recording complete.")
    except KeyboardInterrupt:
        print("\nRecording aborted by user.")
        sys.exit(1)
    except Exception as exc:
        raise RuntimeError(f"Microphone recording failed: {exc}") from exc

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Live microphone speaker identification (1:N).")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help=f"Recording length in seconds (default: {DEFAULT_DURATION}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Similarity threshold (0‑1) for a confident identification "
            f"(default: {DEFAULT_THRESHOLD})."
        ),
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # Step 1 – Record audio from the microphone
    # -------------------------------------------------------------------
    try:
        record_audio(args.duration, TEMP_WAV)
    except Exception as exc:
        print(f"Error during recording: {exc}", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------
    # Step 2 – Generate embedding for the recorded audio
    # -------------------------------------------------------------------
    try:
        print("Generating embedding from temp_recording.wav...")
        test_embedding = generate_embedding(str(TEMP_WAV))
    except Exception as exc:
        print(f"Failed to generate embedding from recording: {exc}", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------
    # Step 3 – Load enrolled speaker database
    # -------------------------------------------------------------------
    db_path = Path("database") / "speakers"
    try:
        database = load_database(db_path)
    except Exception as exc:
        print(f"Error loading speaker database: {exc}", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------
    # Step 4 – Identify the most similar speaker
    # -------------------------------------------------------------------
    predicted, best_sim, is_identified = identify_speaker(
        test_embedding, database, args.threshold
    )

    # -------------------------------------------------------------------
    # Step 5 – Report result
    # -------------------------------------------------------------------
    print("\n--------------------------------")
    if is_identified:
        print(f"Predicted Speaker : {predicted}")
        print(f"Similarity        : {best_sim * 100:.2f}%")
        print("Result            : IDENTIFIED")
    else:
        print("Predicted Speaker : Unknown")
        print(f"Highest Similarity: {best_sim * 100:.2f}%")
        print("Result            : UNKNOWN SPEAKER")
    print("--------------------------------\n")

    # -------------------------------------------------------------------
    # Cleanup temporary file
    # -------------------------------------------------------------------
    try:
        # TEMP_WAV.unlink(missing_ok=True)  # disabled for debugging
        pass
    except Exception:
        pass

if __name__ == "__main__":
    main()
