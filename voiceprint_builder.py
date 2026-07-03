#!/usr/bin/env python
"""
voiceprint_builder.py

Build a speaker "voiceprint" (embedding) from a set of pre‑processed audio
chunks. Each chunk is turned into an embedding using the project's
`generate_embedding` routine, quality‑filtered via pairwise cosine similarity,
and the accepted embeddings are averaged and saved.

Typical usage
-------------
    python voiceprint_builder.py <customer_id>
    # optional overrides
    python voiceprint_builder.py <customer_id> --chunks-dir temp/chunks \
        --threshold 0.70
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

# --------------------------------------------------------------------------- #
# Project import – the embedding generator lives in the app package.
# --------------------------------------------------------------------------- #
try:
    from app.embedding import generate_embedding
except Exception as exc:  # pragma: no cover
    print(f"Error importing generate_embedding: {exc}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #

class ChunkDiscoveryError(Exception):
    """Raised when no chunk files can be found."""
    pass


class EmbeddingError(Exception):
    """Raised when an embedding cannot be generated for a chunk."""
    pass


class VoiceprintSaveError(Exception):
    """Raised when the final voiceprint cannot be written to disk."""
    pass


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def discover_chunks(chunks_dir: Path) -> List[Path]:
    """
    Locate all WAV files in *chunks_dir* and return them sorted alphabetically.
    """
    if not chunks_dir.is_dir():
        raise ChunkDiscoveryError(f"Chunks directory does not exist: {chunks_dir}")

    wav_paths = sorted(chunks_dir.glob("*.wav"))
    if not wav_paths:
        raise ChunkDiscoveryError(f"No WAV files found in {chunks_dir}")

    return wav_paths


def load_waveform(path: Path) -> Tuple[np.ndarray, int]:
    """Load a WAV file and return a mono waveform with its sample rate."""
    waveform, sr = sf.read(str(path))
    waveform = np.asarray(waveform)

    # Convert to mono if necessary.
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    return waveform, int(sr)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Return the L2‑normalized version of *v* (handles zero‑norm safely)."""
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def compute_cosine_similarity_matrix(embs: np.ndarray) -> np.ndarray:
    """Given L2‑normalized embeddings (N×D), compute the N×N cosine matrix."""
    return embs @ embs.T


def average_similarity_per_embedding(sim_matrix: np.ndarray) -> np.ndarray:
    """Average cosine similarity of each embedding to all *other* embeddings."""
    n = sim_matrix.shape[0]
    sum_without_diag = sim_matrix.sum(axis=1) - 1.0
    return sum_without_diag / max(n - 1, 1)


def select_embeddings(
    embeddings: np.ndarray,
    avg_sims: np.ndarray,
    threshold: float,
    min_keep: int = 3,
) -> Tuple[np.ndarray, List[bool]]:
    """
    Keep embeddings with average similarity >= *threshold*.
    If fewer than *min_keep* survive, keep the top *min_keep* by score.
    """
    keep_mask = avg_sims >= threshold
    if keep_mask.sum() < min_keep:
        top_indices = np.argsort(avg_sims)[-min_keep:][::-1]
        keep_mask[:] = False
        keep_mask[top_indices] = True

    kept_embeddings = embeddings[keep_mask]
    return kept_embeddings, keep_mask.tolist()


def save_voiceprint(
    embedding: np.ndarray,
    customer_id: str,
    out_root: Path = Path("database") / "speakers",
) -> Path:
    """Persist the final normalized voiceprint to disk."""
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{customer_id}.npy"
    try:
        np.save(out_path, embedding)
        return out_path
    except Exception as exc:
        raise VoiceprintSaveError(f"Failed to write voiceprint: {exc}") from exc


def print_report(
    chunk_paths: List[Path],
    avg_sims: np.ndarray,
    keep_mask: List[bool],
) -> None:
    """Print a human‑readable report of chunk quality decisions."""
    print("-" * 32)
    print("Chunk Report")
    for path, sim, keep in zip(chunk_paths, avg_sims, keep_mask):
        status = "KEEP" if keep else "REJECT"
        print(f"{path.name:<20} {sim * 100:5.1f}%   {status}")
    kept = sum(keep_mask)
    rejected = len(keep_mask) - kept
    print()
    print(f"Accepted Chunks : {kept}")
    print(f"Rejected Chunks : {rejected}")
    print("-" * 32)


def print_voiceprint_info(customer_id: str, embedding: np.ndarray, out_path: Path) -> None:
    """Print a summary of the saved voiceprint."""
    print("-" * 32)
    print("Voiceprint Saved")
    print()
    print(f"Customer ID   : {customer_id}")
    print(f"Embedding Size: {embedding.shape[0]}")
    print(f"Database Path : {out_path.as_posix()}")
    print("-" * 32)


# --------------------------------------------------------------------------- #
# CLI handling
# --------------------------------------------------------------------------- #

def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a normalized voiceprint from chunk embeddings."
    )
    parser.add_argument(
        "customer_id",
        type=str,
        help="Identifier for the speaker (used in the output filename).",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("temp") / "chunks",
        help="Directory containing chunk WAV files (default: temp/chunks).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum average similarity to keep a chunk (default: 0.70).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_cli()

    # Discover chunk files
    try:
        chunk_paths = discover_chunks(args.chunks_dir)
    except ChunkDiscoveryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Generate embeddings for each chunk
    raw_embeddings: List[np.ndarray] = []
    for path in chunk_paths:
        try:
            emb = generate_embedding(str(path))
            raw_embeddings.append(l2_normalize(emb))
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed {path.name}: {exc}") from exc

    if not raw_embeddings:
        print("Error: No embeddings could be generated.", file=sys.stderr)
        sys.exit(1)

    embeddings = np.stack(raw_embeddings)  # shape (N, D)

    # Similar```python
#!/usr/bin/env python
"""
voiceprint_builder.py

Build a speaker "voiceprint" (embedding) from a set of pre‑processed audio
chunks. Each chunk is turned into an embedding using the project's
`generate_embedding` routine, quality‑filtered via pairwise cosine similarity,
and the accepted embeddings are averaged and saved.

Typical usage
-------------
    python voiceprint_builder.py <customer_id>
    # optional overrides
    python voiceprint_builder.py <customer_id> --chunks-dir temp/chunks \
        --threshold 0.70
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

# --------------------------------------------------------------------------- #
# Project import – the embedding generator lives in the app package.
# --------------------------------------------------------------------------- #
try:
    from app.embedding import generate_embedding
except Exception as exc:  # pragma: no cover
    print(f"Error importing generate_embedding: {exc}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #

class ChunkDiscoveryError(Exception):
    """Raised when no chunk files can be found."""
    pass


class EmbeddingError(Exception):
    """Raised when an embedding cannot be generated for a chunk."""
    pass


class VoiceprintSaveError(Exception):
    """Raised when the final voiceprint cannot be written to disk."""
    pass


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def discover_chunks(chunks_dir: Path) -> List[Path]:
    """
    Locate all WAV files in *chunks_dir* and return them sorted alphabetically.
    """
    if not chunks_dir.is_dir():
        raise ChunkDiscoveryError(f"Chunks directory does not exist: {chunks_dir}")

    wav_paths = sorted(chunks_dir.glob("*.wav"))
    if not wav_paths:
        raise ChunkDiscoveryError(f"No WAV files found in {chunks_dir}")

    return wav_paths


def load_waveform(path: Path) -> Tuple[np.ndarray, int]:
    """Load a WAV file and return a mono waveform with its sample rate."""
    waveform, sr = sf.read(str(path))
    waveform = np.asarray(waveform)

    # Convert to mono if necessary.
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    return waveform, int(sr)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Return the L2‑normalized version of *v* (handles zero‑norm safely)."""
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def compute_cosine_similarity_matrix(embs: np.ndarray) -> np.ndarray:
    """Given L2‑normalized embeddings (N×D), compute the N×N cosine matrix."""
    return embs @ embs.T


def average_similarity_per_embedding(sim_matrix: np.ndarray) -> np.ndarray:
    """Average cosine similarity of each embedding to all *other* embeddings."""
    n = sim_matrix.shape[0]
    sum_without_diag = sim_matrix.sum(axis=1) - 1.0
    return sum_without_diag / max(n - 1, 1)


def select_embeddings(
    embeddings: np.ndarray,
    avg_sims: np.ndarray,
    threshold: float,
    min_keep: int = 3,
) -> Tuple[np.ndarray, List[bool]]:
    """
    Keep embeddings with average similarity >= *threshold*.
    If fewer than *min_keep* survive, keep the top *min_keep* by score.
    """
    keep_mask = avg_sims >= threshold
    if keep_mask.sum() < min_keep:
        top_indices = np.argsort(avg_sims)[-min_keep:][::-1]
        keep_mask[:] = False
        keep_mask[top_indices] = True

    kept_embeddings = embeddings[keep_mask]
    return kept_embeddings, keep_mask.tolist()


def save_voiceprint(
    embedding: np.ndarray,
    customer_id: str,
    out_root: Path = Path("database") / "speakers",
) -> Path:
    """Persist the final normalized voiceprint to disk."""
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{customer_id}.npy"
    try:
        np.save(out_path, embedding)
        return out_path
    except Exception as exc:
        raise VoiceprintSaveError(f"Failed to write voiceprint: {exc}") from exc


def print_report(
    chunk_paths: List[Path],
    avg_sims: np.ndarray,
    keep_mask: List[bool],
) -> None:
    """Print a human‑readable report of chunk quality decisions."""
    print("-" * 32)
    print("Chunk Report")
    for path, sim, keep in zip(chunk_paths, avg_sims, keep_mask):
        status = "KEEP" if keep else "REJECT"
        print(f"{path.name:<20} {sim * 100:5.1f}%   {status}")
    kept = sum(keep_mask)
    rejected = len(keep_mask) - kept
    print()
    print(f"Accepted Chunks : {kept}")
    print(f"Rejected Chunks : {rejected}")
    print("-" * 32)


def print_voiceprint_info(customer_id: str, embedding: np.ndarray, out_path: Path) -> None:
    """Print a summary of the saved voiceprint."""
    print("-" * 32)
    print("Voiceprint Saved")
    print()
    print(f"Customer ID   : {customer_id}")
    print(f"Embedding Size: {embedding.shape[0]}")
    print(f"Database Path : {out_path.as_posix()}")
    print("-" * 32)


# --------------------------------------------------------------------------- #
# CLI handling
# --------------------------------------------------------------------------- #

def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a normalized voiceprint from chunk embeddings."
    )
    parser.add_argument(
        "customer_id",
        type=str,
        help="Identifier for the speaker (used in the output filename).",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("temp") / "chunks",
        help="Directory containing chunk WAV files (default: temp/chunks).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum average similarity to keep a chunk (default: 0.70).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_cli()

    # Discover chunk files
    try:
        chunk_paths = discover_chunks(args.chunks_dir)
    except ChunkDiscoveryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Generate embeddings for each chunk
    raw_embeddings: List[np.ndarray] = []
    for path in chunk_paths:
        try:
            emb = generate_embedding(str(path))
            raw_embeddings.append(l2_normalize(emb))
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed {path.name}: {exc}") from exc

    if not raw_embeddings:
        print("Error: No embeddings could be generated.", file=sys.stderr)
        sys.exit(1)

    embeddings = np.stack(raw_embeddings)  # shape (N, D)

    # Similarity analysis
    sim_matrix = compute_cosine_similarity_matrix(embeddings)
    avg_sims = average_similarity_per_embedding(sim_matrix)

    # Select high‑quality embeddings
    kept_embeddings, keep_mask = select_embeddings(
        embeddings, avg_sims, threshold=args.threshold, min_keep=3
    )

    # Average and final‑normalize
    final_embedding = l2_normalize(np.mean(kept_embeddings, axis=0))

    # Persist voiceprint
    try:
        out_path = save_voiceprint(final_embedding, args.customer_id)
    except VoiceprintSaveError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Reporting
    print_report(chunk_paths, avg_sims, keep_mask)
    print_voiceprint_info(args.customer_id, final_embedding, out_path)


if __name__ == "__main__":
    main()
