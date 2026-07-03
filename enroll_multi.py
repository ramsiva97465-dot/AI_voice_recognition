#!/usr/bin/env python
"""enroll_multi.py

Multi-sample enrollment script for a single speaker.

Usage:
    python enroll_multi.py <folder_path> --name <SpeakerName>

The script reads all *.wav files in the provided folder, generates embeddings
using the shared ``generate_embedding`` function, normalizes each embedding,
averages them, normalizes the average, and saves the final embedding to
``database/speakers/<SpeakerName>.npy``.

All error conditions (missing folder, no wav files, embedding generation
failure, or save failure) are reported with clear messages and cause the
script to exit with a non‑zero status.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

# Reuse the project's embedding generation logic.
from app.embedding import generate_embedding

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command‑line arguments.

    Returns
    -------
    argparse.Namespace
        Namespace with ``folder_path`` (Path) and ``name`` (str).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Enroll a speaker using multiple WAV files. "
            "All .wav files in the folder are processed and the average "
            "embedding is stored."
        )
    )
    parser.add_argument(
        "folder_path",
        type=Path,
        help="Path to a folder containing .wav files for enrollment.",
    )
    parser.add_argument(
        "--name",
        required=True,
        type=str,
        help="Speaker name – used for the saved embedding file.",
    )
    return parser.parse_args()


def _load_wav_paths(folder: Path) -> List[Path]:
    """Return a list of ``.wav`` files inside *folder*.

    Parameters
    ----------
    folder: Path
        Directory to search.

    Returns
    -------
    List[Path]
        Sorted list of wav files.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    wav_paths = sorted(folder.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in {folder}")
    return wav_paths


def _normalize_embedding(emb: np.ndarray) -> np.ndarray:
    """L2‑normalize *emb* unless its norm is zero.

    Returns a new normalized array (original is not modified).
    """
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm


def _save_embedding(speaker_name: str, embedding: np.ndarray) -> Path:
    """Save *embedding* to ``database/speakers/<speaker_name>.npy``.

    The target directory is created if it does not exist.
    """
    db_dir = Path("database") / "speakers"
    db_dir.mkdir(parents=True, exist_ok=True)
    save_path = db_dir / f"{speaker_name}.npy"
    np.save(save_path, embedding)
    return save_path


def main() -> None:
    args = _parse_args()
    folder = args.folder_path
    speaker_name = args.name.strip()

    try:
        wav_files = _load_wav_paths(folder)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    embeddings: List[np.ndarray] = []
    for wav_path in wav_files:
        try:
            emb = generate_embedding(str(wav_path))
            emb = _normalize_embedding(emb)
            embeddings.append(emb)
        except Exception as exc:
            print(
                f"Failed to generate embedding for '{wav_path.name}': {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Compute average and final normalization.
    try:
        avg_emb = np.mean(np.stack(embeddings), axis=0)
        avg_emb = _normalize_embedding(avg_emb)
    except Exception as exc:
        print(f"Error while averaging embeddings: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        saved_path = _save_embedding(speaker_name, avg_emb)
    except Exception as exc:
        print(f"Failed to save embedding: {exc}", file=sys.stderr)
        sys.exit(1)

    # Reporting
    print("--------------------------------")
    print(f"Speaker          : {speaker_name}")
    print(f"Audio Files      : {len(wav_files)}")
    print(f"Embedding Size   : {avg_emb.shape[0]}")
    print(f"Saved To         : {saved_path}")
    print("Enrollment       : SUCCESS")
    print("--------------------------------")


if __name__ == "__main__":
    main()
