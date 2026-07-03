#!/usr/bin/env python
"""
identify.py

Identify the most likely speaker for a given audio file by comparing its
embedding against all enrolled speaker embeddings stored in
`database/speakers/`.

Usage
-----
    python identify.py <audio_file> [--threshold <value>]

Example
-------
    python identify.py audio/test/person.wav
    python identify.py audio/test/person.wav --threshold 0.80

The script prints a sorted list of similarity scores and a final decision:
either the identified speaker (if similarity ≥ threshold) or "Unknown".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Re‑use the production‑grade embedding generator.
# **Do not modify** this import – it encapsulates all audio handling.
from app.embedding import generate_embedding
from constants import DEFAULT_THRESHOLD
# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# (Removed duplicate constant; using imported DEFAULT_THRESHOLD)


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def load_database(db_dir: Path) -> Dict[str, np.ndarray]:
    """Load all ``.npy`` speaker embeddings from the database directory.

    Parameters
    ----------
    db_dir: Path
        Path to ``database/speakers`` folder.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping of speaker name (file stem) → embedding vector.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    RuntimeError
        If no ``.npy`` files are found.
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
    """Compute cosine similarity between two 1‑D vectors.

    Returns a float in the range ``[-1, 1]``.  Zero is returned if either
    vector has zero norm.
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

    Parameters
    ----------
    test_emb: np.ndarray
        Embedding for the input audio.
    database: Dict[str, np.ndarray]
        Mapping of speaker names to their embeddings.
    threshold: float
        Verification threshold (0‑1).

    Returns
    -------
    Tuple[str, float, bool]
        * predicted speaker name (or ``"Unknown"``)
        * similarity of the best match (0‑1)
        * boolean indicating whether the match meets the threshold
    """
    # Compute similarities for every enrolled speaker.
    similarities = {name: cosine_similarity(test_emb, emb) for name, emb in database.items()}

    # Sort by similarity descending.
    sorted_sims = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)

    # Print the sorted list (percentage with two decimals).
    print("\nSearching speaker database...\n")
    for name, sim in sorted_sims:
        status = "PASS" if sim >= threshold else "FAIL"
        print(f"{name:<10}: {sim * 100:.2f}%   [{status}]")

    # Best match.
    best_name, best_sim = sorted_sims[0]
    identified = best_sim >= threshold
    predicted_name = best_name if identified else "Unknown"
    return predicted_name, best_sim, identified


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify a speaker by comparing a test audio file against "
        "all enrolled speaker embeddings."
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to the WAV file to be identified.",
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

    # ------------------------------------------------------------------- #
    # Step 1 – Load the speaker database
    # ------------------------------------------------------------------- #
    db_path = Path("database") / "speakers"
    try:
        database = load_database(db_path)
    except Exception as exc:
        print(f"Error loading database: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------- #
    # Step 2 – Generate embedding for the test audio
    # ------------------------------------------------------------------- #
    try:
        test_embedding = generate_embedding(str(args.audio_file))
    except Exception as exc:
        print(f"Failed to generate embedding for '{args.audio_file}': {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------- #
    # Step 3 – Identify the most similar speaker
    # ------------------------------------------------------------------- #
    predicted, best_sim, is_identified = identify_speaker(
        test_embedding, database, args.threshold
    )

    # ------------------------------------------------------------------- #
    # Step 4 – Print final result block
    # ------------------------------------------------------------------- #
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


if __name__ == "__main__":
    main()
