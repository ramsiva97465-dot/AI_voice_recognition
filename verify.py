#!/usr/bin/env python
# ---------------------------------------------------------------
# verify.py
# ---------------------------------------------------------------
# Command‑line tool to verify a speaker by comparing a test recording
# against an enrolled embedding using cosine similarity.
#
# Usage example:
#   python verify.py audio/test/siva2.wav.wav --name Siva
#   python verify.py audio/test/siva2.wav.wav --name Siva --threshold 0.80
# ---------------------------------------------------------------

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

# Reuse the shared embedding generation function
from app.embedding import generate_embedding

# Project-wide constant
from constants import DEFAULT_THRESHOLD


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a speaker by comparing a test audio file to an enrolled embedding."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the test audio file (wav, flac, etc.).",
    )
    parser.add_argument(
        "--name",
        required=True,
        type=str,
        help="Name of the enrolled speaker (used to locate the stored embedding).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold (between 0 and 1) for verification. Default is {DEFAULT_THRESHOLD}.",
    )
    return parser.parse_args()


def _load_enrolled_embedding(speaker_name: str) -> np.ndarray:
    """Load the previously stored embedding for *speaker_name*.

    The file is expected at ``database/speakers/<SpeakerName>.npy``.
    Raises ``FileNotFoundError`` if the file does not exist.
    """
    speakers_dir = pathlib.Path("database") / "speakers"
    embed_path = speakers_dir / f"{speaker_name}.npy"
    if not embed_path.is_file():
        raise FileNotFoundError(f"Enrolled embedding not found for speaker '{speaker_name}' at {embed_path}")
    return np.load(embed_path)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two 1‑D NumPy arrays.

    The result is in the range ``[-1, 1]``. No external libraries are used.
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Cosine similarity expects 1‑D vectors.")
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Zero‑norm vector encountered in cosine similarity computation.")
    return dot / (norm_a * norm_b)


def main() -> None:
    args = _parse_args()

    speaker_name = args.name.strip()
    threshold = args.threshold

    # -------------------------------------------------------------
    # Load enrolled embedding
    # -------------------------------------------------------------
    print("Loading enrolled speaker...")
    try:
        enrolled = _load_enrolled_embedding(speaker_name)
    except Exception as exc:
        print(f"Error loading enrolled embedding: {exc}")
        sys.exit(1)

    # -------------------------------------------------------------
    # Generate embedding for the test audio
    # -------------------------------------------------------------
    print("Generating test embedding...")
    try:
        test_embedding = generate_embedding(args.audio_path)
    except Exception as exc:
        print(f"Error generating test embedding: {exc}")
        sys.exit(1)

    # -------------------------------------------------------------
    # Compute cosine similarity
    # -------------------------------------------------------------
    print("Calculating cosine similarity...")
    try:
        similarity = _cosine_similarity(enrolled, test_embedding)
    except Exception as exc:
        print(f"Error computing similarity: {exc}")
        sys.exit(1)

    similarity_pct = similarity * 100.0
    threshold_pct = threshold * 100.0
    verified = similarity >= threshold

    # -------------------------------------------------------------
    # Output results
    # -------------------------------------------------------------
    print("--------------------------------")
    print(f"Speaker : {speaker_name}")
    print(f"Similarity : {similarity_pct:.2f} %")
    print(f"Threshold : {threshold_pct:.0f} %")
    print(f"Result : {'VERIFIED' if verified else 'NOT VERIFIED'}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
