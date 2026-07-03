#!/usr/bin/env python
"""
Utility functions for cosine‑similarity calculations used across the
Voice‑Biometrics FastAPI service layer.

Functions
---------
- cosine_similarity(a, b) → float
- identify_best_match(embeddings, query_embedding) → Tuple[str, float]
- verify_match(enrolled, test, threshold=0.80) → Tuple[bool, float]

All functions operate on 1‑D NumPy arrays and raise clear exceptions on
invalid input.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two 1‑D vectors.

    Parameters
    ----------
    a, b : np.ndarray
        Input vectors. Must be one‑dimensional and of the same length.

    Returns
    -------
    float
        Cosine similarity in the interval ``[-1, 1]``.

    Raises
    ------
    ValueError
        If the inputs are not 1‑D, have mismatched shapes, or contain a
        zero‑norm vector.
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("cosine_similarity expects 1‑D vectors.")
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same shape.")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Zero‑norm vector encountered in cosine similarity.")
    return float(np.dot(a, b) / (norm_a * norm_b))


def identify_best_match(
    embeddings: Dict[str, np.ndarray],
    query_embedding: np.ndarray,
) -> Tuple[str, float]:
    """
    Find the speaker ID whose stored embedding is most similar to the
    query embedding.

    Parameters
    ----------
    embeddings : dict[str, np.ndarray]
        Mapping of speaker IDs to their enrolled embeddings.
    query_embedding : np.ndarray
        Embedding generated from the test audio.

    Returns
    -------
    Tuple[str, float]
        ``(best_speaker_id, similarity_score)`` where ``similarity_score``
        is the cosine similarity between the best match and the query.

    Raises
    ------
    ValueError
        If ``embeddings`` is empty or the query embedding has an invalid
        shape.
    """
    if not embeddings:
        raise ValueError("No enrolled speaker embeddings provided.")
    if query_embedding.ndim != 1:
        raise ValueError("Query embedding must be a 1‑D vector.")

    best_id: str | None = None
    best_score = -np.inf

    for speaker_id, emb in embeddings.items():
        try:
            score = cosine_similarity(emb, query_embedding)
        except ValueError:
            # Skip malformed vectors; they should not exist in a healthy DB.
            continue
        if score > best_score:
            best_score = score
            best_id = speaker_id

    if best_id is None:
        raise RuntimeError("Failed to compute a valid similarity for any speaker.")
    return best_id, float(best_score)


def verify_match(
    enrolled: np.ndarray,
    test: np.ndarray,
    threshold: float = 0.80,
) -> Tuple[bool, float]:
    """
    Verify whether a test embedding matches an enrolled embedding.

    Parameters
    ----------
    enrolled : np.ndarray
        The stored speaker embedding.
    test : np.ndarray
        Embedding generated from the verification audio.
    threshold : float, optional
        Minimum cosine similarity required for a positive verification
        (default ``0.80``).

    Returns
    -------
    Tuple[bool, float]
        ``(verified, similarity)`` where ``verified`` is ``True`` when the
        similarity meets or exceeds ``threshold``.

    Raises
    ------
    ValueError
        If the inputs are not compatible 1‑D vectors.
    """
    similarity = cosine_similarity(enrolled, test)
    return similarity >= threshold, similarity
