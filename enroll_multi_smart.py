#!/usr/bin/env python
"""
enroll_multi_smart.py

Smart multi‑sample enrollment script that automatically discards poor‑quality
recordings before building a final speaker embedding.

Usage:
    python enroll_multi_smart.py <folder_path> --name <SpeakerName>

The script:
  1. Loads every *.wav file in the supplied folder.
  2. Generates an embedding for each file (using the shared generate_embedding).
  3. L2‑normalises each embedding.
  4. Computes a cosine‑similarity matrix between all embeddings.
  5. Calculates the average similarity of each recording to the others.
  6. Rejects recordings whose average similarity falls below a configurable
     threshold (default 0.80).
  7. Averages the *accepted* embeddings, normalises the result and saves it
     to ``database/speakers/<SpeakerName>.npy``.
  8. Prints detailed quality and enrollment reports.

All error conditions (missing folder, no wav files, embedding failures,
or complete rejection) are reported with clear messages and cause a
non‑zero exit status.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Reuse the project's embedding generation function.
from app.embedding import generate_embedding

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    """Parse command‑line arguments.

    Returns
    -------
    argparse.Namespace
        Namespace with ``folder_path`` (Path) and ``name`` (str).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Smart enrollment for a speaker using multiple WAV files. "
            "Low‑quality recordings are automatically rejected based on "
            "their average cosine similarity to the rest."
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Average‑similarity threshold for keeping a sample (default: 0.80).",
    )
    return parser.parse_args()


def _load_wav_paths(folder: Path) -> List[Path]:
    """Return a sorted list of ``.wav`` files inside *folder*.

    Parameters
    ----------
    folder : Path
        Directory to search.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist or contains no wav files.

    Returns
    -------
    List[Path]
        Sorted list of wav file paths.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    wav_paths = sorted(folder.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in {folder}")
    return wav_paths


def _normalize_embedding(emb: np.ndarray) -> np.ndarray:
    """L2‑normalize ``emb`` unless its norm is zero."""
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm


def _cosine_similarity_matrix(embs: np.ndarray) -> np.ndarray:
    """Return a cosine‑similarity matrix for the rows of ``embs``.

    Parameters
    ----------
    embs : np.ndarray
        Shape (N, D) where N is the number of embeddings.

    Returns
    -------
    np.ndarray
        Shape (N, N) with values in [-1, 1].
    """
    # Normalize rows to unit length (already done, but safe).
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1, norms)
    unit_embs = embs / safe_norms

    # Cosine similarity = dot product of unit vectors.
    return unit_embs @ unit_embs.T


def _evaluate_quality(
    similarities: np.ndarray, threshold: float
) -> Tuple[List[int], List[float], List[str]]:
    """Determine which samples to keep based on average similarity.

    Parameters
    ----------
    similarities : np.ndarray
        Cosine similarity matrix (N×N).
    threshold : float
        Minimum average similarity required to keep a sample.

    Returns
    -------
    Tuple[List[int], List[float], List[str]]
        *keep_indices*: indices of samples to retain.
        *avg_sims*: average similarity for each sample (percentage).
        *decisions*: list of strings "KEEP" or "REJECT" for each sample.
    """
    n = similarities.shape[0]
    # Exclude self‑similarity by masking the diagonal.
    mask = ~np.eye(n, dtype=bool)
    avg_sims = np.sum(similarities * mask, axis=1) / (n - 1)
    decisions = ["KEEP" if avg >= threshold else "REJECT" for avg in avg_sims]
    keep_indices = [i for i, d in enumerate(decisions) if d == "KEEP"]
    # Convert to percentages for reporting.
    avg_sims_pct = (avg_sims * 100).tolist()
    return keep_indices, avg_sims_pct, decisions


def _save_embedding(speaker_name: str, embedding: np.ndarray) -> Path:
    """Save *embedding* to ``database/speakers/<speaker_name>.npy``.

    The target directory is created if missing.
    """
    db_dir = Path("database") / "speakers"
    db_dir.mkdir(parents=True, exist_ok=True)
    save_path = db_dir / f"{speaker_name}.npy"
    np.save(save_path, embedding)
    return save_path


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #


def main() -> None:
    args = _parse_args()
    folder = args.folder_path
    speaker_name = args.name.strip()
    threshold = args.threshold

    # ------------------------------------------------------------------- #
    # Step 1 – Load WAV files
    # ------------------------------------------------------------------- #
    try:
        wav_files = _load_wav_paths(folder)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------- #
    # Step 2 – Generate & normalise embeddings
    # ------------------------------------------------------------------- #
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

    emb_matrix = np.stack(embeddings)  # shape (N, D)

    # ------------------------------------------------------------------- #
    # Step 3 – Compute similarity matrix & evaluate quality
    # ------------------------------------------------------------------- #
    similarity_matrix = _cosine_similarity_matrix(emb_matrix)
    keep_idx, avg_sims_pct, decisions = _evaluate_quality(
        similarity_matrix, threshold
    )

    # ------------------------------------------------------------------- #
    # Step 4 – Report per‑file quality
    # ------------------------------------------------------------------- #
    print("\n--------------------------------")
    print("Enrollment Quality Report")
    print()
    for wav_path, avg, decision in zip(wav_files, avg_sims_pct, decisions):
        print(f"{wav_path.name:<15} {avg:6.2f}%   {decision}")
    print()
    print(f"Accepted : {len(keep_idx)}")
    print(f"Rejected : {len(wav_files) - len(keep_idx)}")
    print("--------------------------------\n")

    if not keep_idx:
        print("Error: All recordings were rejected. No enrollment performed.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------- #
    # Step 5 – Average accepted embeddings, final normalisation & save
    # ------------------------------------------------------------------- #
    accepted_embeddings = emb_matrix[keep_idx]
    avg_emb = np.mean(accepted_embeddings, axis=0)
    avg_emb = _normalize_embedding(avg_emb)

    try:
        saved_path = _save_embedding(speaker_name, avg_emb)
    except Exception as exc:
        print(f"Failed to save embedding: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------- #
    # Step 6 – Final enrollment report
    # ------------------------------------------------------------------- #
    print("--------------------------------")
    print(f"Speaker          : {speaker_name}")
    print(f"Accepted Samples : {len(keep_idx)}")
    print(f"Rejected Samples : {len(wav_files) - len(keep_idx)}")
    print(f"Embedding Size   : {avg_emb.shape[0]}")
    print(f"Saved To         : {saved_path}")
    print("Enrollment       : SUCCESS")
    print("--------------------------------")


if __name__ == "__main__":
    main()
